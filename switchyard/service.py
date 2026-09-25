"""Chat orchestration: resolve a route and call a provider.

The HTTP layer depends only on ``ChatService``; reliability (Phase 2), caching (Phase 4) and
metering plug in here without touching handlers or adapters.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from switchyard.errors import ProviderError, gateway_error_from_provider
from switchyard.router import Router, Target
from switchyard.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompletionResult:
    response: ChatCompletion
    provider: str
    model: str
    upstream_ms: float


@dataclass(slots=True)
class StreamResult:
    """A stream whose first chunk has already been received.

    Pulling the first chunk before returning lets the handler send a real HTTP error status
    (and, in Phase 2, fail over) when a provider fails before producing anything. Once the
    first byte has gone to the client the status line is committed.
    """

    chunks: AsyncGenerator[ChatCompletionChunk, None]
    provider: str
    model: str


async def _prepend(
    first: ChatCompletionChunk, rest: AsyncGenerator[ChatCompletionChunk, None]
) -> AsyncGenerator[ChatCompletionChunk, None]:
    try:
        yield first
        async for chunk in rest:
            yield chunk
    finally:
        await rest.aclose()


class ChatService:
    def __init__(self, router: Router) -> None:
        self.router = router

    async def complete(self, request: ChatCompletionRequest) -> CompletionResult:
        target = self._primary(request)
        started = time.perf_counter()
        try:
            response = await target.provider.complete(request, target.model)
        except ProviderError as err:
            self._log_failure(err, target)
            raise gateway_error_from_provider(err) from err
        return CompletionResult(
            response=response,
            provider=target.provider.name,
            model=target.model,
            upstream_ms=(time.perf_counter() - started) * 1000,
        )

    async def stream(self, request: ChatCompletionRequest) -> StreamResult:
        target = self._primary(request)
        chunks = target.provider.stream(request, target.model)
        try:
            first = await anext(chunks)
        except StopAsyncIteration:
            first = None
        except ProviderError as err:
            await chunks.aclose()
            self._log_failure(err, target)
            raise gateway_error_from_provider(err) from err

        async def empty() -> AsyncGenerator[ChatCompletionChunk, None]:
            return
            yield  # pragma: no cover

        stream = _prepend(first, chunks) if first is not None else empty()
        return StreamResult(chunks=stream, provider=target.provider.name, model=target.model)

    def _primary(self, request: ChatCompletionRequest) -> Target:
        return self.router.resolve(request.model)[0]

    @staticmethod
    def _log_failure(err: ProviderError, target: Target) -> None:
        logger.warning(
            "provider call failed",
            extra={
                "provider": err.provider,
                "upstream_model": target.model,
                "failure": err.kind.value,
                "status": err.status,
                "detail": err.message,
            },
        )
