"""Request-context middleware (pure ASGI).

Written as raw ASGI rather than ``BaseHTTPMiddleware`` because the latter buffers/wraps
streaming responses and runs the app in a separate task, which breaks contextvar propagation
and makes SSE latency measurements lie. Here the access log is emitted when the *last* body
byte is sent, so a stream's logged duration is its real duration.
"""

from __future__ import annotations

import logging
import time

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from switchyard import metrics
from switchyard.context import accept_or_create_request_id, request_id_var

access_logger = logging.getLogger("switchyard.access")

REQUEST_ID_HEADER = "x-request-id"
CHAT_PATH = "/v1/chat/completions"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for key, value in scope["headers"]:
            if key == REQUEST_ID_HEADER.encode():
                incoming = value.decode("latin-1")
                break
        request_id = accept_or_create_request_id(incoming)
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["started"] = started
        status = 500
        finished_at: float | None = None
        is_chat = scope["path"] == CHAT_PATH and scope["method"] == "POST"
        if is_chat:
            metrics.IN_FLIGHT.inc()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status, finished_at
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                # The client has the whole response. Background work Starlette runs after
                # this point (token settlement, for example) must not count as latency.
                finished_at = time.perf_counter()

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration = (finished_at or time.perf_counter()) - started
            if is_chat:
                metrics.IN_FLIGHT.dec()
                _record_chat(state, status, duration)
            if scope["path"] not in ("/healthz", "/readyz", "/metrics"):
                access_logger.info(
                    "request",
                    extra={
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status,
                        "duration_ms": round(duration * 1000, 2),
                        "key_id": state.get("key_id"),
                        "route": state.get("route"),
                        "cache": state.get("cache_status"),
                        "provider": state.get("provider"),
                    },
                )
            request_id_var.reset(token)


def _record_chat(state: dict[str, object], status: int, duration: float) -> None:
    """Request-level metrics. The handler leaves route/cache/stream/upstream time in the
    request state; for requests rejected before that (401, 429, 400) sensible defaults apply."""
    route = str(state.get("route", "none"))
    stream = "true" if state.get("stream") else "false"
    cache = str(state.get("cache_status", "none"))
    metrics.REQUESTS.labels(route, stream, str(status), cache).inc()
    if status < 400:
        metrics.REQUEST_DURATION.labels(route, stream, cache).observe(duration)
        upstream_s = state.get("upstream_s")
        if stream == "false" and isinstance(upstream_s, float):
            metrics.OVERHEAD.labels("false").observe(max(duration - upstream_s, 0.0))
