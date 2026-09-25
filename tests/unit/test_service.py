"""Retry / failover / breaker orchestration against scripted fake providers."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from switchyard.config import GatewayConfig, ProviderConfig
from switchyard.errors import (
    FailureKind,
    GatewayError,
    InvalidRequestError,
    ProviderError,
    ServiceUnavailableError,
    UpstreamError,
)
from switchyard.providers.base import Provider
from switchyard.reliability.backoff import RetryPolicy
from switchyard.reliability.breaker import BreakerSettings, BreakerState, CircuitBreaker
from switchyard.router import Router
from switchyard.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest
from switchyard.service import ChatService

OK = "ok"
Script = list[Any]  # each item: OK, a FailureKind, ("hang",), or ("mid", n) for streams


def _completion(provider: str) -> ChatCompletion:
    return ChatCompletion(
        id="c",
        created=1,
        model=provider,
        choices=[{"index": 0, "message": {"role": "assistant", "content": provider}}],
    )


def _chunk(text: str, finish: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c",
        created=1,
        model="m",
        choices=[{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
    )


class FakeProvider(Provider):
    def __init__(self, name: str, script: Script) -> None:
        super().__init__(name, ProviderConfig(type="mock", base_url="http://fake"))
        self.script = list(script)
        self.calls = 0
        self.stream_closed = 0

    def _next(self) -> Any:
        self.calls += 1
        return self.script.pop(0) if self.script else OK

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        step = self._next()
        if step == ("hang",):
            await asyncio.sleep(3600)
        if isinstance(step, FailureKind):
            retry_after = 60.0 if step is FailureKind.RATE_LIMITED else None
            raise ProviderError(self.name, step, "scripted", retry_after_s=retry_after)
        return _completion(self.name)

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        step = self._next()
        try:
            if isinstance(step, FailureKind):
                raise ProviderError(self.name, step, "scripted")
            for i in range(3):
                if isinstance(step, tuple) and step[0] == "mid" and i == step[1]:
                    raise ProviderError(self.name, FailureKind.CONNECTION_DROPPED, "mid")
                yield _chunk(self.name, "stop" if i == 2 else None)
        finally:
            self.stream_closed += 1

    async def health(self) -> bool:
        return True


REQUEST = ChatCompletionRequest.model_validate(
    {"model": "r", "messages": [{"role": "user", "content": "hi"}]}
)


class Harness:
    def __init__(
        self,
        scripts: dict[str, Script],
        *,
        max_attempts: int = 2,
        breaker: BreakerSettings | None = None,
        request_timeout_s: float = 5.0,
    ) -> None:
        self.providers = {name: FakeProvider(name, script) for name, script in scripts.items()}
        config = GatewayConfig.model_validate(
            {
                "providers": {n: {"type": "mock", "base_url": "http://x"} for n in scripts},
                "routes": [
                    {"model": "r", "targets": [{"provider": n, "model": "m"} for n in scripts]}
                ],
            }
        )
        self.breakers = {
            n: CircuitBreaker(n, breaker or BreakerSettings(window_size=100, min_calls=100))
            for n in scripts
        }
        self.sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        self.service = ChatService(
            Router(config, dict(self.providers)),
            self.breakers,
            retry=RetryPolicy(max_attempts, 0.1, 1.0, rng=random.Random(0)),
            request_timeout_s=request_timeout_s,
            sleep=fake_sleep,
        )

    def calls(self) -> dict[str, int]:
        return {name: p.calls for name, p in self.providers.items()}


async def test_success_first_try() -> None:
    h = Harness({"a": [OK], "b": []})
    result = await h.service.complete(REQUEST)
    assert result.provider == "a"
    assert h.calls() == {"a": 1, "b": 0}
    assert result.trace.failovers == 0


async def test_retryable_failure_is_retried_on_same_provider_with_backoff() -> None:
    h = Harness({"a": [FailureKind.UPSTREAM_5XX, OK], "b": []})
    result = await h.service.complete(REQUEST)
    assert result.provider == "a"
    assert h.calls() == {"a": 2, "b": 0}
    assert len(h.sleeps) == 1 and 0 <= h.sleeps[0] <= 0.1


async def test_retries_exhausted_then_failover() -> None:
    h = Harness({"a": [FailureKind.TIMEOUT, FailureKind.TIMEOUT], "b": [OK]})
    result = await h.service.complete(REQUEST)
    assert result.provider == "b"
    assert h.calls() == {"a": 2, "b": 1}
    assert result.trace.failovers == 1
    assert [a.failure for a in result.trace.attempts] == [
        FailureKind.TIMEOUT,
        FailureKind.TIMEOUT,
        None,
    ]


async def test_non_retryable_failure_fails_over_immediately() -> None:
    h = Harness({"a": [FailureKind.AUTH], "b": [OK]})
    result = await h.service.complete(REQUEST)
    assert result.provider == "b"
    assert h.calls() == {"a": 1, "b": 1}
    assert h.sleeps == []


async def test_bad_request_is_final() -> None:
    h = Harness({"a": [FailureKind.BAD_REQUEST], "b": [OK]})
    with pytest.raises(InvalidRequestError):
        await h.service.complete(REQUEST)
    assert h.calls() == {"a": 1, "b": 0}


async def test_retry_after_beyond_budget_fails_over_without_sleeping() -> None:
    h = Harness({"a": [FailureKind.RATE_LIMITED], "b": [OK]})  # scripted Retry-After: 60s
    result = await h.service.complete(REQUEST)
    assert result.provider == "b"
    assert h.sleeps == []


async def test_all_fail_reports_last_real_error() -> None:
    h = Harness({"a": [FailureKind.UPSTREAM_5XX] * 2, "b": [FailureKind.CONNECT] * 2})
    with pytest.raises(UpstreamError) as exc:
        await h.service.complete(REQUEST)
    assert exc.value.status_code == 502
    assert exc.value.code == "connect"


async def test_open_breaker_is_skipped_without_calling_provider() -> None:
    h = Harness({"a": [], "b": []})
    for _ in range(100):
        h.breakers["a"].record_failure()
    assert h.breakers["a"].state is BreakerState.OPEN
    result = await h.service.complete(REQUEST)
    assert result.provider == "b"
    assert h.calls() == {"a": 0, "b": 1}
    assert result.trace.attempts[0].failure is FailureKind.CIRCUIT_OPEN


async def test_all_breakers_open_is_503_with_retry_after() -> None:
    h = Harness({"a": [], "b": []}, breaker=BreakerSettings(window_size=1, min_calls=1, open_s=7))
    for b in h.breakers.values():
        b.record_failure()
    with pytest.raises(ServiceUnavailableError) as exc:
        await h.service.complete(REQUEST)
    assert exc.value.status_code == 503
    assert exc.value.headers["retry-after"] == "7"
    assert h.calls() == {"a": 0, "b": 0}


async def test_breaker_trips_from_repeated_failures_then_short_circuits() -> None:
    settings = BreakerSettings(window_size=4, min_calls=4, failure_rate_threshold=0.5, open_s=60)
    h = Harness({"a": [FailureKind.UPSTREAM_5XX] * 4, "b": []}, breaker=settings)
    for _ in range(2):
        assert (await h.service.complete(REQUEST)).provider == "b"
    assert h.breakers["a"].state is BreakerState.OPEN
    assert (await h.service.complete(REQUEST)).provider == "b"
    assert h.calls()["a"] == 4  # the third request never touched "a"


async def test_bad_requests_do_not_trip_the_breaker() -> None:
    settings = BreakerSettings(window_size=2, min_calls=2)
    h = Harness({"a": [FailureKind.BAD_REQUEST] * 5}, breaker=settings)
    for _ in range(5):
        with pytest.raises(InvalidRequestError):
            await h.service.complete(REQUEST)
    assert h.breakers["a"].state is BreakerState.CLOSED


async def test_request_budget_bounds_total_time() -> None:
    h = Harness({"a": [("hang",)], "b": []}, request_timeout_s=0.1)
    with pytest.raises(GatewayError) as exc:
        await h.service.complete(REQUEST)
    assert exc.value.status_code == 504
    assert exc.value.code == "request_timeout"
    assert h.calls() == {"a": 1, "b": 0}


async def test_cancellation_releases_half_open_permit() -> None:
    settings = BreakerSettings(window_size=1, min_calls=1, open_s=0.0, half_open_max_calls=1)
    h = Harness({"a": [("hang",)]}, breaker=settings)
    h.breakers["a"].record_failure()
    assert h.breakers["a"].state is BreakerState.HALF_OPEN
    task = asyncio.create_task(h.service.complete(REQUEST))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.breakers["a"].allow()  # the permit came back


# -- streaming ---------------------------------------------------------------------------


async def _read(chunks: AsyncGenerator[ChatCompletionChunk, None]) -> list[str]:
    return [c.choices[0]["delta"]["content"] async for c in chunks]


async def test_stream_fails_over_before_first_chunk() -> None:
    h = Harness({"a": [FailureKind.CONNECT, FailureKind.CONNECT], "b": [OK]})
    result = await h.service.stream(REQUEST)
    assert result.provider == "b"
    assert await _read(result.chunks) == ["b", "b", "b"]
    assert h.providers["a"].stream_closed == 2  # failed attempts released their connections


async def test_mid_stream_failure_is_not_retried_but_counts_against_breaker() -> None:
    settings = BreakerSettings(window_size=1, min_calls=1)
    h = Harness({"a": [("mid", 1)], "b": []}, breaker=settings)
    result = await h.service.stream(REQUEST)
    with pytest.raises(ProviderError):
        await _read(result.chunks)
    assert h.calls() == {"a": 1, "b": 0}
    assert h.breakers["a"].state is BreakerState.OPEN


async def test_completed_stream_records_success() -> None:
    settings = BreakerSettings(window_size=1, min_calls=1, open_s=0.0, half_open_max_calls=1)
    h = Harness({"a": [OK]}, breaker=settings)
    h.breakers["a"].record_failure()
    result = await h.service.stream(REQUEST)
    assert h.breakers["a"].state is BreakerState.HALF_OPEN  # not decided until the end
    await _read(result.chunks)
    assert h.breakers["a"].state is BreakerState.CLOSED


async def test_client_disconnect_mid_stream_releases_and_closes_upstream() -> None:
    settings = BreakerSettings(window_size=1, min_calls=1, open_s=0.0, half_open_max_calls=1)
    h = Harness({"a": [OK]}, breaker=settings)
    h.breakers["a"].record_failure()
    result = await h.service.stream(REQUEST)
    await anext(result.chunks)
    await result.chunks.aclose()  # what Starlette does when the client goes away
    assert h.providers["a"].stream_closed == 1
    assert h.breakers["a"].state is BreakerState.HALF_OPEN
    assert h.breakers["a"].allow()
