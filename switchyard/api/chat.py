"""``POST /v1/chat/completions`` and ``GET /v1/models``."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from switchyard.api.sse import DONE_EVENT, encode_event
from switchyard.errors import ProviderError, UpstreamError, gateway_error_from_provider
from switchyard.schemas import ChatCompletionRequest
from switchyard.service import ChatService, StreamResult

logger = logging.getLogger(__name__)

router = APIRouter()

PROVIDER_HEADER = "x-switchyard-provider"
MODEL_HEADER = "x-switchyard-model"
UPSTREAM_MS_HEADER = "x-switchyard-upstream-ms"
ATTEMPTS_HEADER = "x-switchyard-attempts"
FAILOVERS_HEADER = "x-switchyard-failovers"


def _service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
    service = _service(request)
    if body.stream:
        result = await service.stream(body)
        return StreamingResponse(
            _sse(result, include_usage=body.wants_stream_usage),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
                PROVIDER_HEADER: result.provider,
                MODEL_HEADER: result.model,
                ATTEMPTS_HEADER: str(len(result.trace.attempts)),
                FAILOVERS_HEADER: str(result.trace.failovers),
            },
        )

    completion = await service.complete(body)
    return JSONResponse(
        completion.response.model_dump(mode="json", exclude_unset=True),
        headers={
            PROVIDER_HEADER: completion.provider,
            MODEL_HEADER: completion.model,
            UPSTREAM_MS_HEADER: f"{completion.upstream_ms:.1f}",
            ATTEMPTS_HEADER: str(len(completion.trace.attempts)),
            FAILOVERS_HEADER: str(completion.trace.failovers),
        },
    )


async def _sse(result: StreamResult, *, include_usage: bool) -> AsyncIterator[bytes]:
    """Relay chunks as SSE.

    Headers (200) are already on the wire when this runs, so a mid-stream provider failure
    cannot change the status code. Instead we emit an OpenAI-style ``error`` event and end the
    stream without ``[DONE]``; the OpenAI SDKs raise ``APIError`` on such an event, so clients
    fail fast instead of hanging.
    """
    started = time.perf_counter()
    chunk_count = 0
    try:
        async for chunk in result.chunks:
            if chunk.is_usage_only and not include_usage:
                continue
            chunk_count += 1
            yield encode_event(chunk)
    except ProviderError as err:
        logger.warning(
            "stream failed after first byte",
            extra={
                "provider": err.provider,
                "failure": err.kind.value,
                "chunks_sent": chunk_count,
                "detail": err.message,
            },
        )
        error = gateway_error_from_provider(err)
        if not isinstance(error, UpstreamError):
            error = UpstreamError(error.message, code=error.code)
        yield encode_event(error.to_body())
        return
    finally:
        await result.chunks.aclose()
    logger.debug(
        "stream complete",
        extra={
            "provider": result.provider,
            "chunks": chunk_count,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    yield DONE_EVENT


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, object]:
    service = _service(request)
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "switchyard"}
            for model in service.router.models()
        ],
    }
