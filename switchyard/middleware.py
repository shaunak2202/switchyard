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

from switchyard.context import accept_or_create_request_id, request_id_var

access_logger = logging.getLogger("switchyard.access")

REQUEST_ID_HEADER = "x-request-id"


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
        scope.setdefault("state", {})["request_id"] = request_id

        started = time.perf_counter()
        status = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            if scope["path"] not in ("/healthz", "/readyz", "/metrics"):
                access_logger.info(
                    "request",
                    extra={
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
            request_id_var.reset(token)
