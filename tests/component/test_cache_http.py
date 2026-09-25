"""Exact and semantic caching through the gateway, backed by real Redis (vector index)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from switchyard.cache.embeddings import ExactTextVerifier, HashingEmbedder
from switchyard.cache.normalize import CacheDirective
from switchyard.cache.store import ExactCache, ResponseCache
from switchyard.schemas import ChatCompletionRequest
from tests.conftest import MockServer

pytestmark = pytest.mark.usefixtures("mocks")

CACHE_ON = {
    "exact": {"enabled": True, "ttl_s": 60},
    "semantic": {
        "enabled": True,
        "candidate_threshold": 0.6,
        "verifier_threshold": 0.5,
        "ttl_s": 60,
    },
}


def body(question: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "only-a",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": question},
        ],
    } | extra


@pytest.fixture
def gateway(make_gateway: Callable[..., str]) -> str:
    return make_gateway(embedder=HashingEmbedder(), verifier=ExactTextVerifier(), cache=CACHE_ON)


@pytest.fixture
def send(api_key: str, gateway: str) -> Callable[..., httpx.Response]:
    def _send(payload: dict[str, Any], **headers: str) -> httpx.Response:
        return httpx.post(
            f"{gateway}/v1/chat/completions",
            json=payload,
            headers={"authorization": f"Bearer {api_key}"} | headers,
            timeout=10,
        )

    return _send


def _unique(text: str) -> str:
    """Cache entries outlive a test, so every test asks its own questions."""
    return f"{text} [{time.monotonic_ns()}]"


def _wait_for_store() -> None:
    time.sleep(0.15)  # cache writes happen in the background after the response


def _content(resp: httpx.Response) -> str:
    return str(resp.json()["choices"][0]["message"]["content"])


def test_exact_hit_skips_the_provider(
    send: Callable[..., httpx.Response], mocks: tuple[MockServer, MockServer]
) -> None:
    payload = body(_unique("What is the capital of France?"))
    first = send(payload)
    assert first.headers["x-switchyard-cache"] == "miss"
    _wait_for_store()
    second = send(payload)
    assert second.headers["x-switchyard-cache"] == "hit-exact"
    assert _content(second) == _content(first)
    assert second.json()["id"] != first.json()["id"]
    assert int(second.headers["age"]) >= 0
    assert mocks[0].stats()["requests"] == 1


def test_sampled_requests_bypass_unless_opted_in(
    send: Callable[..., httpx.Response], mocks: tuple[MockServer, MockServer]
) -> None:
    payload = body(_unique("Write a poem"), temperature=0.8)
    assert send(payload).headers["x-switchyard-cache"] == "bypass"
    _wait_for_store()
    assert send(payload).headers["x-switchyard-cache"] == "bypass"
    assert mocks[0].stats()["requests"] == 2

    send(payload, **{"x-switchyard-cache": "allow"})
    _wait_for_store()
    assert send(payload, **{"x-switchyard-cache": "allow"}).headers["x-switchyard-cache"] == (
        "hit-exact"
    )


def test_cache_control_no_cache_and_no_store(
    send: Callable[..., httpx.Response], mocks: tuple[MockServer, MockServer]
) -> None:
    payload = body(_unique("Explain TCP"))
    send(payload, **{"cache-control": "no-store"})
    _wait_for_store()
    assert send(payload).headers["x-switchyard-cache"] == "miss"  # nothing was stored
    _wait_for_store()
    assert send(payload, **{"cache-control": "no-cache"}).headers["x-switchyard-cache"] == "miss"
    assert send(payload).headers["x-switchyard-cache"] == "hit-exact"
    assert mocks[0].stats()["requests"] == 3


def test_semantic_hit_on_rephrased_prompt(
    send: Callable[..., httpx.Response], mocks: tuple[MockServer, MockServer]
) -> None:
    tag = time.monotonic_ns()
    first = send(body(f"What is the capital city of Japan {tag}?"))
    _wait_for_store()
    second = send(body(f"what is the capital city of japan {tag}"))
    assert second.headers["x-switchyard-cache"] == "hit-semantic"
    assert float(second.headers["x-switchyard-cache-similarity"]) >= 0.6
    assert float(second.headers["x-switchyard-cache-verifier-score"]) == 1.0
    assert _content(second) == _content(first)
    assert mocks[0].stats()["requests"] == 1


def test_semantic_never_crosses_system_prompts_or_unrelated_questions(
    send: Callable[..., httpx.Response],
) -> None:
    tag = time.monotonic_ns()
    send(body(f"How tall is Mount Everest {tag}?"))
    _wait_for_store()
    other_system = body(f"How tall is Mount Everest {tag}?")
    other_system["messages"][0]["content"] = "Answer in French."
    assert send(other_system).headers["x-switchyard-cache"] == "miss"
    assert send(body(f"Who wrote Hamlet {tag}?")).headers["x-switchyard-cache"] == "miss"
    no_semantic = send(
        body(f"how tall is mount everest {tag}"), **{"x-switchyard-cache": "no-semantic"}
    )
    assert no_semantic.headers["x-switchyard-cache"] == "miss"


def test_verifier_rejects_a_near_miss_the_embedding_accepts(
    send: Callable[..., httpx.Response], mocks: tuple[MockServer, MockServer]
) -> None:
    """ "5 km to miles" vs "5 miles to km": same words, so the bag-of-words embedding scores 1.0.
    The verifier must veto it."""
    tag = time.monotonic_ns()
    send(body(f"Convert 5 km to miles {tag}"))
    _wait_for_store()
    reversed_ = send(body(f"Convert 5 miles to km {tag}"))
    assert reversed_.headers["x-switchyard-cache"] == "miss"
    assert mocks[0].stats()["requests"] == 2


def test_stream_miss_is_stored_and_serves_both_shapes(
    send: Callable[..., httpx.Response], mocks: tuple[MockServer, MockServer]
) -> None:
    payload = body(_unique("Name three rivers"))
    streamed = send(payload | {"stream": True})
    assert streamed.headers["x-switchyard-cache"] == "miss"
    events = [ln[6:] for ln in streamed.text.split("\n") if ln.startswith("data: ")]
    text = "".join(
        json.loads(e)["choices"][0]["delta"].get("content", "")
        for e in events[:-1]
        if json.loads(e)["choices"]
    )
    _wait_for_store()

    as_json = send(payload)
    assert as_json.headers["x-switchyard-cache"] == "hit-exact"
    assert _content(as_json) == text

    replayed = send(payload | {"stream": True})
    assert replayed.headers["x-switchyard-cache"] == "hit-exact"
    replay_events = [ln[6:] for ln in replayed.text.split("\n") if ln.startswith("data: ")]
    assert replay_events[-1] == "[DONE]"
    replay_text = "".join(
        json.loads(e)["choices"][0]["delta"].get("content", "")
        for e in replay_events[:-1]
        if json.loads(e)["choices"]
    )
    assert replay_text == text
    assert mocks[0].stats()["requests"] == 1


def test_failures_are_never_cached(
    send: Callable[..., httpx.Response], mocks: tuple[MockServer, MockServer]
) -> None:
    mock_a, _ = mocks
    payload = body(_unique("Tell me a fact"))
    mock_a.configure(stream_abort_rate=1.0, fail_after_tokens=2)
    aborted = send(payload | {"stream": True})
    assert "error" in aborted.text
    mock_a.configure(stream_abort_rate=0.0, error_rate=1.0, error_status=500)
    assert send(payload).status_code == 502
    mock_a.configure(error_rate=0.0)
    _wait_for_store()
    assert send(payload).headers["x-switchyard-cache"] == "miss"


def test_cache_hit_refunds_the_token_estimate(api_key: str, gateway: str) -> None:
    from tests.conftest import create_key

    key = create_key(rpm=100, tpm=100_000)
    payload = body(_unique("Summarise Hamlet"), max_tokens=500)
    headers = {"authorization": f"Bearer {key}"}
    url = f"{gateway}/v1/chat/completions"
    httpx.post(url, json=payload, headers=headers, timeout=10)
    _wait_for_store()
    hit = httpx.post(url, json=payload, headers=headers, timeout=10)
    assert hit.headers["x-switchyard-cache"] == "hit-exact"
    _wait_for_store()
    after = httpx.post(url, json=body(_unique("x"), max_tokens=1), headers=headers, timeout=10)
    # Only the first (real) request's usage is still charged; the hit was refunded.
    assert int(after.headers["x-ratelimit-remaining-tokens"]) > 100_000 - 100


async def test_redis_failure_degrades_to_a_miss() -> None:
    from redis.asyncio import Redis

    dead = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    cache = ResponseCache(ExactCache(dead, "x", 60), None)
    request = ChatCompletionRequest.model_validate(body("hello"))
    lookup = await cache.lookup(request, CacheDirective())
    assert lookup.eligible and lookup.hit is None
    await dead.aclose()
