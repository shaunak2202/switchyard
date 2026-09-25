"""Exact and semantic response caches in Redis, behind one ``ResponseCache`` facade.

* **Exact:** ``<prefix>:cache:exact:<sha256(normalised request)>`` holds the completion JSON,
  with a TTL. One GET per lookup.
* **Semantic:** every entry is a hash ``<prefix>:cache:sem:<id>`` with the namespace (a TAG),
  the prompt embedding (a FLOAT32 VECTOR in an HNSW index, cosine distance) and the completion.
  A lookup is one ``FT.SEARCH`` KNN query filtered to the namespace. Expired entries leave the
  index automatically.

Lookups are exact first (cheap and always correct), then semantic. Cache failures degrade to a
miss and are logged: the cache must never be the reason a request fails.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from switchyard.cache.embeddings import Embedder, Vector, Verifier
from switchyard.cache.normalize import (
    CacheDirective,
    SemanticQuery,
    cacheable,
    exact_key,
    semantic_query,
)
from switchyard.schemas import ChatCompletion, ChatCompletionRequest

logger = logging.getLogger(__name__)

CacheKind = Literal["exact", "semantic"]


async def _command(redis: Redis, *args: Any) -> Any:
    """``execute_command`` for the FT.* commands, which redis-py's typed API doesn't cover."""
    return await redis.execute_command(*args)  # type: ignore[no-untyped-call]


@dataclass(frozen=True, slots=True)
class CacheHit:
    kind: CacheKind
    completion: ChatCompletion
    age_s: int
    similarity: float | None = None
    verifier_score: float | None = None


@dataclass(slots=True)
class CacheLookup:
    """The outcome of a lookup, plus what a later ``store`` needs so work isn't repeated."""

    eligible: bool
    hit: CacheHit | None = None
    exact_key: str | None = None
    semantic: SemanticQuery | None = None
    vector: Vector | None = None

    @property
    def status(self) -> str:
        if not self.eligible:
            return "bypass"
        return f"hit-{self.hit.kind}" if self.hit else "miss"


class ExactCache:
    def __init__(self, redis: Redis, prefix: str, ttl_s: int) -> None:
        self.redis = redis
        self.prefix = prefix
        self.ttl_s = ttl_s

    def _key(self, digest: str) -> str:
        return f"{self.prefix}:cache:exact:{digest}"

    async def get(self, digest: str) -> CacheHit | None:
        pipe = self.redis.pipeline(transaction=False)
        pipe.get(self._key(digest))
        pipe.ttl(self._key(digest))
        raw, ttl = await pipe.execute()
        if raw is None:
            return None
        age = max(self.ttl_s - int(ttl), 0) if ttl and ttl > 0 else 0
        return CacheHit("exact", ChatCompletion.model_validate_json(raw), age_s=age)

    async def set(self, digest: str, completion: ChatCompletion) -> None:
        await self.redis.set(
            self._key(digest), completion.model_dump_json(exclude_unset=True), ex=self.ttl_s
        )


class SemanticCache:
    def __init__(
        self,
        redis: Redis,
        prefix: str,
        embedder: Embedder,
        *,
        candidate_threshold: float,
        verifier: Verifier | None = None,
        verifier_threshold: float = 1.0,
        ttl_s: int,
    ) -> None:
        self.redis = redis
        self.prefix = prefix
        self.embedder = embedder
        self.candidate_threshold = candidate_threshold
        self.verifier = verifier
        self.verifier_threshold = verifier_threshold
        self.ttl_s = ttl_s
        # Vectors from different models are not comparable: each model gets its own index
        # and key space.
        model = f"{embedder.name.replace('/', '_')}:{embedder.dim}"
        self.index = f"{prefix}:cache:semidx:{model}"
        self.key_prefix = f"{prefix}:cache:sem:{model}:"

    async def ensure_index(self) -> None:
        try:
            await _command(
                self.redis,
                "FT.CREATE", self.index, "ON", "HASH", "PREFIX", "1", self.key_prefix,
                "SCHEMA",
                "ns", "TAG",
                "vec", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32", "DIM", str(self.embedder.dim), "DISTANCE_METRIC", "COSINE",
            )  # fmt: skip
        except ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise

    async def search(self, namespace: str, text: str, vector: Vector) -> CacheHit | None:
        """Nearest cached prompt in the namespace, if it passes both stages."""
        raw = await _command(
            self.redis,
            "FT.SEARCH", self.index,
            f"(@ns:{{{namespace}}})=>[KNN 1 @vec $v AS dist]",
            "PARAMS", "2", "v", vector.astype(np.float32).tobytes(),
            "SORTBY", "dist",
            "RETURN", "4", "dist", "response", "created", "prompt",
            "DIALECT", "2",
        )  # fmt: skip
        values = _first_search_result(raw)
        if values is None:
            return None
        similarity = 1.0 - float(values["dist"])
        if similarity < self.candidate_threshold:
            return None
        verifier_score = None
        if self.verifier is not None:
            cached_prompt = _key_str(values["prompt"])
            verifier_score = await self.verifier.score(text, cached_prompt)
            if verifier_score < self.verifier_threshold:
                logger.debug(
                    "semantic candidate rejected by verifier",
                    extra={"similarity": round(similarity, 4), "score": round(verifier_score, 4)},
                )
                return None
        return CacheHit(
            "semantic",
            ChatCompletion.model_validate_json(values["response"]),
            age_s=max(int(time.time()) - int(values["created"]), 0),
            similarity=round(similarity, 4),
            verifier_score=None if verifier_score is None else round(verifier_score, 4),
        )

    async def store(
        self, namespace: str, text: str, vector: Vector, completion: ChatCompletion
    ) -> None:
        key = f"{self.key_prefix}{uuid.uuid4().hex}"
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(
            key,
            mapping={
                "ns": namespace,
                "vec": vector.astype(np.float32).tobytes(),
                "prompt": text,
                "response": completion.model_dump_json(exclude_unset=True),
                "created": int(time.time()),
            },
        )
        pipe.expire(key, self.ttl_s)
        await pipe.execute()


def _key_str(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _first_search_result(raw: Any) -> dict[str, Any] | None:
    """Fields of the top FT.SEARCH hit, whichever reply shape the client produced.

    Raw RESP2 is ``[total, key, [field, value, ...], ...]``. redis-py >= 8 parses search
    replies into ``{"results": [{"extra_attributes": {...}}], ...}`` even through
    ``execute_command``.
    """
    if isinstance(raw, dict):
        results = raw.get(b"results", raw.get("results")) or []
        if not results:
            return None
        attrs = results[0].get(b"extra_attributes", results[0].get("extra_attributes")) or {}
        return {_key_str(k): v for k, v in attrs.items()}
    if not raw or raw[0] == 0 or len(raw) < 3:
        return None
    fields = raw[2]
    return {_key_str(k): v for k, v in zip(fields[::2], fields[1::2], strict=True)}


class ResponseCache:
    def __init__(
        self,
        exact: ExactCache | None,
        semantic: SemanticCache | None,
        *,
        max_semantic_chars: int = 2000,
    ) -> None:
        self.exact = exact
        self.semantic = semantic
        self.max_semantic_chars = max_semantic_chars

    @property
    def enabled(self) -> bool:
        return self.exact is not None or self.semantic is not None

    async def lookup(
        self, request: ChatCompletionRequest, directive: CacheDirective
    ) -> CacheLookup:
        if not self.enabled or not cacheable(request, directive):
            return CacheLookup(eligible=False)
        digest = exact_key(request)
        result = CacheLookup(eligible=True, exact_key=digest)
        if self.semantic is not None and directive.semantic:
            result.semantic = semantic_query(request, self.max_semantic_chars)
        if not directive.read:
            return result

        try:
            if self.exact is not None:
                result.hit = await self.exact.get(digest)
                if result.hit:
                    return result
            if self.semantic is not None and result.semantic is not None:
                result.vector = await self.semantic.embedder.embed(result.semantic.text)
                result.hit = await self.semantic.search(
                    result.semantic.namespace, result.semantic.text, result.vector
                )
        except (RedisError, ValueError) as exc:
            logger.warning("cache lookup failed; treating as miss", extra={"error": repr(exc)})
            result.hit = None
        return result

    async def store(
        self, lookup: CacheLookup, completion: ChatCompletion, directive: CacheDirective
    ) -> None:
        if not lookup.eligible or lookup.hit is not None or not directive.write:
            return
        if not _complete_answer(completion):
            return
        try:
            if self.exact is not None and lookup.exact_key:
                await self.exact.set(lookup.exact_key, completion)
            if self.semantic is not None and lookup.semantic is not None:
                vector = lookup.vector
                if vector is None:
                    vector = await self.semantic.embedder.embed(lookup.semantic.text)
                await self.semantic.store(
                    lookup.semantic.namespace, lookup.semantic.text, vector, completion
                )
        except (RedisError, ValueError) as exc:
            logger.warning("cache store failed", extra={"error": repr(exc)})


def _complete_answer(completion: ChatCompletion) -> bool:
    """Only cache answers that ended normally. ``length`` counts: truncation at ``max_tokens`` is
    deterministic, and ``max_tokens`` is part of the key. An answer cut off by a content
    filter or with no finish reason is not one to replay to other callers."""
    if not completion.choices:
        return False
    for choice in completion.choices:
        if choice.get("finish_reason") not in ("stop", "length", "tool_calls"):
            return False
    return True
