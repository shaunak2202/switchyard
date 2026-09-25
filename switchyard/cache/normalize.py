"""Request normalisation, cache keys and caching policy.

Two requests share an exact-cache entry iff every input that can change the output is equal
after normalisation. Normalisation is deliberately conservative: it removes differences that
cannot change a model's answer (JSON key order, ``\\r\\n`` vs ``\\n``, surrounding whitespace, a
single text part vs a plain string, the order of stop sequences) and nothing else. Collapsing
internal whitespace, for example, would conflate two different code snippets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from starlette.datastructures import Headers

from switchyard.schemas import ChatCompletionRequest, ChatMessage

# Fields that cannot change the generated output.
_NON_SEMANTIC_FIELDS = frozenset({"stream", "stream_options", "user", "metadata", "store"})


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def normalize_content(content: str | list[dict[str, Any]] | None) -> Any:
    if isinstance(content, str):
        return normalize_text(content)
    if isinstance(content, list):
        if all(part.get("type") == "text" for part in content):
            return normalize_text("".join(str(part.get("text", "")) for part in content))
        return content  # multimodal parts: keep verbatim
    return content


def normalize_message(message: ChatMessage) -> dict[str, Any]:
    data = message.model_dump(mode="json", exclude_none=True)
    if "content" in data:
        data["content"] = normalize_content(message.content)
    return data


def normalized_params(request: ChatCompletionRequest) -> dict[str, Any]:
    """Everything except the messages that influences generation."""
    data = request.model_dump(mode="json", exclude_unset=True, exclude={"messages"})
    params = {k: v for k, v in data.items() if k not in _NON_SEMANTIC_FIELDS and v is not None}
    if isinstance(params.get("stop"), list):
        params["stop"] = sorted(params["stop"])
    elif isinstance(params.get("stop"), str):
        params["stop"] = [params["stop"]]
    if "max_completion_tokens" in params and "max_tokens" not in params:
        params["max_tokens"] = params.pop("max_completion_tokens")
    return params


def _digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def exact_key(request: ChatCompletionRequest) -> str:
    return _digest(
        {
            "params": normalized_params(request),
            "messages": [normalize_message(m) for m in request.messages],
        }
    )


@dataclass(frozen=True, slots=True)
class SemanticQuery:
    namespace: str
    text: str


def semantic_query(request: ChatCompletionRequest, max_chars: int) -> SemanticQuery | None:
    """What a semantic lookup compares, or ``None`` if this request is not a safe candidate.

    Only single-turn prompts qualify: an optional system/developer message plus one user
    message. The user text is what gets embedded. Everything else (model, parameters, system
    prompt) must match *exactly* and forms the namespace, so "similar" never crosses a model or
    a system prompt. Tool use and multi-choice requests are excluded, because a near-duplicate
    prompt there does not imply an interchangeable answer.
    """
    if request.n not in (None, 1):
        return None
    extra = request.model_extra or {}
    if extra.get("tools") or extra.get("functions") or extra.get("response_format"):
        return None
    *preamble, last = request.messages
    if last.role != "user" or not isinstance(normalize_content(last.content), str):
        return None
    if any(m.role not in ("system", "developer") for m in preamble):
        return None
    text = normalize_content(last.content)
    if not text or len(text) > max_chars:
        return None
    namespace = _digest(
        {
            "params": normalized_params(request),
            "preamble": [normalize_message(m) for m in preamble],
        }
    )
    return SemanticQuery(namespace=namespace, text=text)


@dataclass(frozen=True, slots=True)
class CacheDirective:
    """How the caller wants the cache used, from request headers.

    * ``Cache-Control: no-cache``: don't serve from cache (a fresh answer may still be stored).
    * ``Cache-Control: no-store``: don't store this response.
    * ``X-Switchyard-Cache: allow``: opt in to caching even though temperature > 0.
    * ``X-Switchyard-Cache: no-semantic``: exact matches only.
    """

    read: bool = True
    write: bool = True
    opt_in: bool = False
    semantic: bool = True

    @classmethod
    def from_headers(cls, headers: Headers) -> CacheDirective:
        cache_control = {
            token.strip().lower() for token in headers.get("cache-control", "").split(",")
        }
        switchyard = {
            token.strip().lower() for token in headers.get("x-switchyard-cache", "").split(",")
        }
        return cls(
            read="no-cache" not in cache_control,
            write="no-store" not in cache_control,
            opt_in="allow" in switchyard,
            semantic="no-semantic" not in switchyard,
        )


def cacheable(request: ChatCompletionRequest, directive: CacheDirective) -> bool:
    """Only deterministic requests are cached unless the caller opts in.

    With temperature > 0 (or unset, which means the provider default, usually 1) the caller
    asked for a *sample*; replaying one sample to everyone would silently change the
    semantics of the API.
    """
    return request.temperature == 0 or directive.opt_in
