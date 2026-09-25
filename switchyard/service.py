"""Chat orchestration: routing, retries with backoff, circuit breaking and failover.

For each target of a route, in priority order:

1. Ask the provider's circuit breaker for permission. If it is open, skip to the next target
   right away; that is the point of a breaker, since it saves a timeout.
2. Call the provider. On a *retryable* failure, back off (full jitter, honouring
   ``Retry-After``) and retry the same provider up to ``max_attempts`` times.
3. On a failure that is *failover-able* but not retryable, or once retries are used up, move to
   the next target. A non-failover-able failure (a bad request) is returned to the caller
   immediately.

The whole loop runs under one request deadline, so retries times timeouts can never exceed
``request_timeout_s``. For streams, the loop ends once the first chunk arrives; after that the
response is committed and a failure is reported in-band (ADR-003).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from switchyard.errors import (
    FailureKind,
    ProviderError,
    ServiceUnavailableError,
    UpstreamError,
    gateway_error_from_provider,
)
from switchyard.reliability.backoff import RetryPolicy
from switchyard.reliability.breaker import CircuitBreaker
from switchyard.router import Router, Target
from switchyard.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest

logger = logging.getLogger(__name__)

T = TypeVar("T")
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Attempt:
    provider: str
    failure: FailureKind | None  # None means the attempt succeeded


@dataclass(slots=True)
class CallTrace:
    """What happened while serving one request. Surfaced as headers, logs and metrics."""

    attempts: list[Attempt] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    @property
    def failovers(self) -> int:
        """How many times we moved on to a different provider."""
        providers = [a.provider for a in self.attempts]
        return sum(1 for a, b in itertools.pairwise(providers) if a != b)


@dataclass(slots=True)
class CompletionResult:
    response: ChatCompletion
    provider: str
    model: str
    upstream_ms: float
    trace: CallTrace


@dataclass(slots=True)
class StreamResult:
    """A stream whose first chunk has already been received (see ADR-003)."""

    chunks: AsyncGenerator[ChatCompletionChunk, None]
    provider: str
    model: str
    trace: CallTrace


@dataclass(slots=True)
class _Opened:
    first: ChatCompletionChunk | None
    rest: AsyncGenerator[ChatCompletionChunk, None]


class ChatService:
    def __init__(
        self,
        router: Router,
        breakers: Mapping[str, CircuitBreaker],
        *,
        retry: RetryPolicy | None = None,
        request_timeout_s: float = 60.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.router = router
        self.breakers = breakers
        self.retry = retry or RetryPolicy()
        self.request_timeout_s = request_timeout_s
        self._sleep = sleep

    # -- public API ------------------------------------------------------------------------

    async def complete(self, request: ChatCompletionRequest) -> CompletionResult:
        trace = CallTrace()

        async def call(target: Target) -> tuple[ChatCompletion, float]:
            started = time.perf_counter()
            response = await target.provider.complete(request, target.model)
            return response, (time.perf_counter() - started) * 1000

        (response, upstream_ms), target = await self._execute(request, call, trace)
        self.breakers[target.provider.name].record_success()
        return CompletionResult(
            response=response,
            provider=target.provider.name,
            model=target.model,
            upstream_ms=upstream_ms,
            trace=trace,
        )

    async def stream(self, request: ChatCompletionRequest) -> StreamResult:
        trace = CallTrace()

        async def call(target: Target) -> _Opened:
            chunks = target.provider.stream(request, target.model)
            try:
                return _Opened(await anext(chunks), chunks)
            except StopAsyncIteration:
                return _Opened(None, chunks)
            except BaseException:
                await chunks.aclose()
                raise

        opened, target = await self._execute(request, call, trace)
        breaker = self.breakers[target.provider.name]
        return StreamResult(
            chunks=self._guard_stream(opened, breaker),
            provider=target.provider.name,
            model=target.model,
            trace=trace,
        )

    # -- internals -------------------------------------------------------------------------

    async def _execute(
        self,
        request: ChatCompletionRequest,
        call: Callable[[Target], Awaitable[T]],
        trace: CallTrace,
    ) -> tuple[T, Target]:
        """Run ``call`` against the route's targets with retries and failover.

        On success, the winning provider's breaker permit is still held: the caller must record
        the outcome (``record_success`` now, or when the stream ends).
        """
        targets = self.router.resolve(request.model)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.request_timeout_s
        last_real_error: ProviderError | None = None

        for index, target in enumerate(targets):
            name = target.provider.name
            breaker = self.breakers[name]
            if index > 0 and trace.attempts:
                logger.warning(
                    "failing over",
                    extra={
                        "from_provider": trace.attempts[-1].provider,
                        "to_provider": name,
                        "reason": str(trace.attempts[-1].failure),
                    },
                )

            for attempt in range(1, self.retry.max_attempts + 1):
                if not breaker.allow():
                    trace.attempts.append(Attempt(name, FailureKind.CIRCUIT_OPEN))
                    break
                try:
                    async with asyncio.timeout_at(deadline):
                        result = await call(target)
                except ProviderError as err:
                    self._record_failure(breaker, err)
                    trace.attempts.append(Attempt(name, err.kind))
                    last_real_error = err
                    logger.warning(
                        "provider call failed",
                        extra={
                            "provider": name,
                            "upstream_model": target.model,
                            "attempt": attempt,
                            "failure": err.kind.value,
                            "status": err.status,
                            "detail": err.message,
                        },
                    )
                    if not err.kind.failover:
                        raise gateway_error_from_provider(err) from err
                    if not err.kind.retryable or attempt == self.retry.max_attempts:
                        break
                    delay = self.retry.delay(attempt, err.retry_after_s)
                    if delay is None or loop.time() + delay >= deadline:
                        break
                    await self._sleep(delay)
                except TimeoutError:
                    breaker.release()
                    trace.attempts.append(Attempt(name, FailureKind.TIMEOUT))
                    raise UpstreamError(
                        f"request exceeded the gateway budget of {self.request_timeout_s:g}s",
                        code="request_timeout",
                        status_code=504,
                    ) from None
                except BaseException:
                    breaker.release()  # cancelled (client went away) or a bug; not evidence
                    raise
                else:
                    trace.attempts.append(Attempt(name, None))
                    return result, target

        if last_real_error is not None:
            raise gateway_error_from_provider(last_real_error)
        retry_after = min(self.breakers[t.provider.name].retry_after_s() for t in targets)
        raise ServiceUnavailableError(
            "all providers for this model are unavailable (circuit open)",
            code="all_circuits_open",
            headers={"retry-after": str(max(1, round(retry_after)))},
        )

    @staticmethod
    def _record_failure(breaker: CircuitBreaker, err: ProviderError) -> None:
        if err.kind.counts_against_provider:
            breaker.record_failure()
        else:
            breaker.release()

    async def _guard_stream(
        self, opened: _Opened, breaker: CircuitBreaker
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Relay the rest of the stream and report its outcome to the circuit breaker."""
        settled = False
        try:
            if opened.first is not None:
                yield opened.first
                async for chunk in opened.rest:
                    yield chunk
            breaker.record_success()
            settled = True
        except ProviderError as err:
            self._record_failure(breaker, err)
            settled = True
            raise
        finally:
            if not settled:
                breaker.release()  # client disconnected mid-stream
            await opened.rest.aclose()
