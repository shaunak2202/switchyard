"""Models behind the semantic cache.

* An ``Embedder`` (bi-encoder) turns a prompt into a vector. It is cheap (one forward pass per
  prompt) and finds *candidates* by vector search, but its similarity cannot tell "5 km to
  miles" from "5 miles to km".
* A ``Verifier`` (cross-encoder) reads the new prompt and the cached prompt *together* and
  scores whether they ask the same thing. It is more expensive (one pass per pair), so it only
  runs when a candidate exists.

The two-stage design and its thresholds come from ``scripts/eval_semantic_threshold.py``
(ADR-016).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol

import anyio
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, text: str) -> Vector:
        """Unit-length float32 vector, so cosine similarity is a dot product."""
        ...


class Verifier(Protocol):
    name: str

    async def score(self, query: str, candidate: str) -> float:
        """Probability-like score in [0, 1] that both prompts ask the same thing."""
        ...


class SentenceTransformerEmbedder:
    """A local sentence-transformers model on CPU.

    Inference runs on one dedicated thread. That keeps it off the event loop, and a single
    thread avoids oversubscribing cores: torch already parallelises inside one encode, and
    concurrent encodes would only contend for the same cores.
    """

    def __init__(self, model: SentenceTransformer, name: str) -> None:
        self._model = model
        self.name = name
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"model {name} has no fixed embedding dimension")
        self.dim = int(dim)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")

    @classmethod
    async def load(cls, name: str) -> SentenceTransformerEmbedder:
        from sentence_transformers import SentenceTransformer

        model = await anyio.to_thread.run_sync(lambda: SentenceTransformer(name, device="cpu"))
        embedder = cls(model, name)
        await embedder.embed("warm up")  # first call pays one-off initialisation cost
        return embedder

    def _encode(self, text: str) -> Vector:
        vector = self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vector, dtype=np.float32)

    async def embed(self, text: str) -> Vector:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._encode, text)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class CrossEncoderVerifier:
    """A local sentence-transformers cross-encoder on CPU (same threading model as the
    embedder)."""

    def __init__(self, model: CrossEncoder, name: str) -> None:
        self._model = model
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verify")

    @classmethod
    async def load(cls, name: str) -> CrossEncoderVerifier:
        from sentence_transformers import CrossEncoder

        model = await anyio.to_thread.run_sync(lambda: CrossEncoder(name, device="cpu"))
        verifier = cls(model, name)
        await verifier.score("warm up", "warm up")
        return verifier

    def _predict(self, query: str, candidate: str) -> float:
        # Same call as the offline evaluation, so the chosen threshold means the same thing.
        return float(self._model.predict([(query, candidate)], show_progress_bar=False)[0])

    async def score(self, query: str, candidate: str) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._predict, query, candidate)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests: shared words ⇒ high cosine similarity."""

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, text: str) -> Vector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            bucket = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest())
            vector[bucket % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class ExactTextVerifier:
    """Test verifier: accepts a candidate only if the prompts match ignoring case and
    punctuation."""

    name = "exact-text"

    async def score(self, query: str, candidate: str) -> float:
        def canon(text: str) -> list[str]:
            return re.findall(r"\w+", text.lower())

        return 1.0 if canon(query) == canon(candidate) else 0.0
