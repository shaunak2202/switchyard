"""Authentication and rate limiting through the HTTP gateway."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest

from tests.conftest import create_key

pytestmark = pytest.mark.usefixtures("mocks")

BODY = {"model": "mock", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}


def _auth(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


@pytest.mark.parametrize(
    "headers",
    [{}, {"authorization": "Bearer sk-sy-not-a-real-key"}, {"authorization": "Basic abc"}],
)
async def test_missing_or_invalid_key_is_401(gateway_url: str, headers: dict[str, str]) -> None:
    async with httpx.AsyncClient(base_url=gateway_url) as client:
        resp = await client.post("/v1/chat/completions", json=BODY, headers=headers)
        models = await client.get("/v1/models", headers=headers)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"
    assert resp.headers["www-authenticate"] == "Bearer"
    assert models.status_code == 401


async def test_success_carries_rate_limit_headers(gateway_url: str) -> None:
    key = create_key(rpm=100, tpm=100_000)
    async with httpx.AsyncClient(base_url=gateway_url) as client:
        resp = await client.post("/v1/chat/completions", json=BODY, headers=_auth(key))
    assert resp.status_code == 200
    assert resp.headers["x-ratelimit-limit-requests"] == "100"
    assert resp.headers["x-ratelimit-remaining-requests"] == "99"
    assert resp.headers["x-ratelimit-limit-tokens"] == "100000"


async def test_concurrent_http_burst_admits_exactly_the_limit(gateway_url: str) -> None:
    """The hard requirement: fire many requests in parallel, the limit holds exactly."""
    limit, total = 20, 150
    key = create_key(rpm=limit, tpm=10_000_000, name="burst")
    started = time.monotonic()
    limits = httpx.Limits(max_connections=total)
    async with httpx.AsyncClient(base_url=gateway_url, timeout=30, limits=limits) as client:
        responses = await asyncio.gather(
            *(
                client.post("/v1/chat/completions", json=BODY, headers=_auth(key))
                for _ in range(total)
            )
        )
    assert time.monotonic() - started < 60 / limit  # refill < 1 request during the burst
    codes = [r.status_code for r in responses]
    assert codes.count(200) == limit
    assert codes.count(429) == total - limit
    rejected = next(r for r in responses if r.status_code == 429)
    assert rejected.json()["error"]["type"] == "rate_limit_error"
    assert int(rejected.headers["retry-after"]) >= 1
    assert rejected.headers["x-ratelimit-remaining-requests"] == "0"


async def test_openai_sdk_sees_rate_limit_error(gateway_url: str) -> None:
    import openai

    key = create_key(rpm=1, tpm=10_000)
    client = openai.AsyncOpenAI(base_url=f"{gateway_url}/v1", api_key=key, max_retries=0)
    await client.chat.completions.create(model="mock", messages=[{"role": "user", "content": "a"}])
    with pytest.raises(openai.RateLimitError):
        await client.chat.completions.create(
            model="mock", messages=[{"role": "user", "content": "a"}]
        )


async def test_request_larger_than_tpm_is_400(gateway_url: str) -> None:
    key = create_key(rpm=100, tpm=50)
    async with httpx.AsyncClient(base_url=gateway_url) as client:
        resp = await client.post(
            "/v1/chat/completions", json=BODY | {"max_tokens": 1000}, headers=_auth(key)
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tokens_exceed_limit"


async def _remaining_tokens_after(client: httpx.AsyncClient, key: str, body: dict[str, Any]) -> int:
    resp = await client.post("/v1/chat/completions", json=body, headers=_auth(key))
    assert resp.status_code == 200, resp.text
    if body.get("stream"):
        _ = resp.text
    return int(resp.headers["x-ratelimit-remaining-tokens"])


@pytest.mark.parametrize("stream", [False, True])
async def test_token_estimate_is_reconciled_with_real_usage(gateway_url: str, stream: bool) -> None:
    """max_tokens=500 is pre-charged, but the mock only generates 8 tokens: the rest comes back."""
    key = create_key(rpm=1000, tpm=100_000)
    body = BODY | {"max_tokens": 500, "stream": stream}
    async with httpx.AsyncClient(base_url=gateway_url, timeout=10) as client:
        first = await _remaining_tokens_after(client, key, body)
        estimate = 100_000 - first
        assert estimate > 500
        await asyncio.sleep(0.1)  # settlement runs after the response is sent
        second = await _remaining_tokens_after(client, key, body)
    # Without reconciliation: second == 100_000 - 2 * estimate.
    # With it, the first request cost only its real usage (~10 tokens).
    assert second >= 100_000 - estimate - 20


async def test_failed_upstream_call_refunds_tokens(gateway_url: str, mocks: Any) -> None:
    mocks[0].configure(error_rate=1.0, error_status=400)  # non-retryable, no failover
    key = create_key(rpm=1000, tpm=10_000)
    async with httpx.AsyncClient(base_url=gateway_url) as client:
        failed = await client.post("/v1/chat/completions", json=BODY, headers=_auth(key))
        assert failed.status_code == 400
        assert "x-ratelimit-remaining-tokens" in failed.headers
        mocks[0].configure(error_rate=0.0)
        ok = await client.post("/v1/chat/completions", json=BODY, headers=_auth(key))
    cost = 10_000 - int(failed.headers["x-ratelimit-remaining-tokens"])
    assert int(ok.headers["x-ratelimit-remaining-tokens"]) >= 10_000 - cost - 1


async def test_readiness_reports_redis(gateway_url: str) -> None:
    async with httpx.AsyncClient(base_url=gateway_url) as client:
        body = (await client.get("/readyz")).json()
    assert body["checks"]["redis"] is True
