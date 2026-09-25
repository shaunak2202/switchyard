"""``POST /v1/chat/completions`` and ``GET /v1/models``.

Request flow: admit (auth + rate limit) → cache lookup → [hit: replay] / [miss: providers with
retries and failover → relay → store in cache in the background] → settle token usage.

The handler leaves route, cache status, provider and upstream time in ``request.state``; the
middleware turns them into request-level metrics and the access log line.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.datastructures import State

from switchyard import metrics
from switchyard.api.guard import Guard
from switchyard.api.sse import DONE_EVENT, encode_event
from switchyard.cache.normalize import CacheDirective
from switchyard.cache.store import CacheHit, CacheLookup, ResponseCache
from switchyard.cache.streaming import StreamAssembler, replay_as_chunks
from switchyard.errors import (
    GatewayError,
    ModelNotFoundError,
    ProviderError,
    UpstreamError,
    gateway_error_from_provider,
)
from switchyard.ratelimit.limiter import Decision
from switchyard.ratelimit.tokens import estimate_prompt_tokens, estimate_tokens_from_chars
from switchyard.schemas import ChatCompletion, ChatCompletionRequest, Usage
from switchyard.service import ChatService, StreamResult
from switchyard.tasks import BackgroundTasks

logger = logging.getLogger(__name__)

router = APIRouter()

PROVIDER_HEADER = "x-switchyard-provider"
MODEL_HEADER = "x-switchyard-model"
UPSTREAM_MS_HEADER = "x-switchyard-upstream-ms"
ATTEMPTS_HEADER = "x-switchyard-attempts"
FAILOVERS_HEADER = "x-switchyard-failovers"
CACHE_HEADER = "x-switchyard-cache"
SIMILARITY_HEADER = "x-switchyard-cache-similarity"
VERIFIER_HEADER = "x-switchyard-cache-verifier-score"

SSE_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}


def _state(request: Request) -> tuple[Guard, ChatService, ResponseCache, BackgroundTasks]:
    state = request.app.state
    return state.guard, state.chat_service, state.cache, state.tasks


def _route_label(service: ChatService, model: str) -> str:
    """A bounded metrics label: configured aliases by name, anything else collapsed."""
    if model in service.router.models():
        return model
    try:
        [target] = service.router.resolve(model)
    except (ModelNotFoundError, ValueError):
        return "unknown"
    return f"{target.provider.name}/*"


def _record_usage(provider: str, usage: Usage | None) -> None:
    if usage is not None:
        metrics.TOKENS.labels(provider, "prompt").inc(usage.prompt_tokens)
        metrics.TOKENS.labels(provider, "completion").inc(usage.completion_tokens)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
    guard, service, cache, tasks = _state(request)
    request.state.stream = body.stream
    request.state.route = _route_label(service, body.model)
    decision = await guard.admit(request, body)
    limit_headers = decision.headers() if decision else {}

    directive = CacheDirective.from_headers(request.headers)
    lookup = await cache.lookup(body, directive)
    request.state.cache_status = lookup.status
    if lookup.hit is not None:
        return _serve_hit(lookup.hit, body, request.state, guard, decision, limit_headers)

    try:
        if body.stream:
            result = await service.stream(body)
        else:
            completion = await service.complete(body)
    except GatewayError as exc:
        await guard.settle(decision, 0)  # nothing was generated: refund the estimate
        exc.headers = limit_headers | {CACHE_HEADER: lookup.status} | exc.headers
        raise

    if body.stream:
        request.state.provider = result.provider
        request.state.upstream_s = result.trace.upstream_s
        relay = _Relay(
            result=result,
            body=body,
            state=request.state,
            guard=guard,
            decision=decision,
            cache=cache,
            tasks=tasks,
            lookup=lookup,
            directive=directive,
        )
        return StreamingResponse(
            relay.run(),
            media_type="text/event-stream",
            headers=limit_headers
            | SSE_HEADERS
            | {
                PROVIDER_HEADER: result.provider,
                MODEL_HEADER: result.model,
                ATTEMPTS_HEADER: str(len(result.trace.attempts)),
                FAILOVERS_HEADER: str(result.trace.failovers),
                CACHE_HEADER: lookup.status,
            },
        )

    request.state.provider = completion.provider
    request.state.upstream_s = completion.trace.upstream_s
    if lookup.eligible:
        tasks.spawn(cache.store(lookup, completion.response, directive), name="cache-store")
    usage = completion.response.usage
    _record_usage(completion.provider, usage)
    actual = usage.total_tokens if usage else estimate_prompt_tokens(body)
    return JSONResponse(
        completion.response.model_dump(mode="json", exclude_unset=True),
        headers=limit_headers
        | {
            PROVIDER_HEADER: completion.provider,
            MODEL_HEADER: completion.model,
            UPSTREAM_MS_HEADER: f"{completion.upstream_ms:.1f}",
            ATTEMPTS_HEADER: str(len(completion.trace.attempts)),
            FAILOVERS_HEADER: str(completion.trace.failovers),
            CACHE_HEADER: lookup.status,
        },
        # Settle after the response is sent: reconciliation is off the latency path.
        background=BackgroundTask(guard.settle, decision, actual),
    )


def _serve_hit(
    hit: CacheHit,
    body: ChatCompletionRequest,
    state: State,
    guard: Guard,
    decision: Decision | None,
    limit_headers: dict[str, str],
) -> Response:
    """A cache hit consumes no provider tokens, so the whole token estimate is refunded (the
    request still counts against requests/min)."""
    completion = hit.completion.model_copy(update={"id": f"chatcmpl-cache-{uuid.uuid4().hex}"})
    headers = limit_headers | {CACHE_HEADER: f"hit-{hit.kind}", "age": str(hit.age_s)}
    if hit.similarity is not None:
        headers[SIMILARITY_HEADER] = f"{hit.similarity:.4f}"
    if hit.verifier_score is not None:
        headers[VERIFIER_HEADER] = f"{hit.verifier_score:.4f}"
    background = BackgroundTask(guard.settle, decision, 0)
    if body.stream:
        return StreamingResponse(
            _replay(completion, state, include_usage=body.wants_stream_usage),
            media_type="text/event-stream",
            headers=headers | SSE_HEADERS,
            background=background,
        )
    return JSONResponse(
        completion.model_dump(mode="json", exclude_unset=True),
        headers=headers,
        background=background,
    )


def _observe_ttft(state: State) -> None:
    metrics.TTFT.labels(state.route, state.cache_status).observe(
        time.perf_counter() - state.started
    )


async def _replay(
    completion: ChatCompletion, state: State, *, include_usage: bool
) -> AsyncIterator[bytes]:
    first = True
    for chunk in replay_as_chunks(completion, include_usage=include_usage):
        yield encode_event(chunk)
        if first and chunk.has_content():
            _observe_ttft(state)
            first = False
    yield DONE_EVENT


@dataclass(slots=True)
class _Relay:
    """Relays one upstream stream to the client as SSE, then accounts for it.

    Headers (200) are already on the wire when this runs, so a mid-stream provider failure
    cannot change the status code. Instead we emit an OpenAI-style ``error`` event and end the
    stream without ``[DONE]``; the OpenAI SDKs raise ``APIError`` on such an event, so clients
    fail fast instead of hanging.
    """

    result: StreamResult
    body: ChatCompletionRequest
    state: State
    guard: Guard
    decision: Decision | None
    cache: ResponseCache
    tasks: BackgroundTasks
    lookup: CacheLookup
    directive: CacheDirective

    async def run(self) -> AsyncIterator[bytes]:
        body, result = self.body, self.result
        reported: Usage | None = None
        streamed_chars = 0
        sent_first = sent_content = False
        store = self.lookup.eligible and self.directive.write
        assembler = StreamAssembler() if store else None
        try:
            async for chunk in result.chunks:
                if chunk.usage is not None:
                    reported = chunk.usage
                for choice in chunk.choices:
                    streamed_chars += len((choice.get("delta") or {}).get("content") or "")
                if assembler is not None:
                    assembler.add(chunk)
                if chunk.is_usage_only and not body.wants_stream_usage:
                    continue
                yield encode_event(chunk)
                if not sent_first:
                    sent_first = True
                    # Overhead up to the first byte: what the gateway added before the client
                    # saw anything, beyond the provider's own time to first chunk.
                    since_start = time.perf_counter() - self.state.started
                    metrics.OVERHEAD.labels("true").observe(
                        max(since_start - result.trace.upstream_s, 0.0)
                    )
                if not sent_content and chunk.has_content():
                    sent_content = True
                    _observe_ttft(self.state)
        except ProviderError as err:
            logger.warning(
                "stream failed after first byte",
                extra={"provider": err.provider, "failure": err.kind.value, "detail": err.message},
            )
            error = gateway_error_from_provider(err)
            if not isinstance(error, UpstreamError):
                error = UpstreamError(error.message, code=error.code)
            yield encode_event(error.to_body())
            return
        finally:
            await result.chunks.aclose()
            _record_usage(result.provider, reported)
            # Charge what was actually produced, including for streams the client abandoned.
            actual = reported.total_tokens if reported else None
            if actual is None:
                actual = estimate_prompt_tokens(body) + estimate_tokens_from_chars(streamed_chars)
            await self.guard.settle(self.decision, actual)

        # Only a stream that completed normally is worth caching.
        if assembler is not None and (assembled := assembler.build()) is not None:
            self.tasks.spawn(
                self.cache.store(self.lookup, assembled, self.directive), name="cache-store"
            )
        yield DONE_EVENT


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, object]:
    guard, service, _, _ = _state(request)
    await guard.authenticate(request)
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "switchyard"}
            for model in service.router.models()
        ],
    }
