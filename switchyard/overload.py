"""Overload protection: shed new work when the event loop is saturated (ADR-020).

An asyncio server has no thread pool to exhaust. When CPU-bound, work piles up in the event
loop's ready queue and *every* request slows down, including ones already admitted. A
concurrency cap is the wrong signal for an LLM gateway, since thousands of idle streams cost
almost nothing while a few hundred busy requests can pin a core.

Event-loop lag measures the saturation itself: a probe asks to wake after ``interval_s`` and
records how late it actually ran. A smoothed lag above ``max_lag_s`` means the loop can't keep
up, so new chat requests are refused with a cheap 503 + Retry-After before any real work
happens. Requests already in flight finish with bounded latency instead of everyone timing out
together.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

from starlette.types import ASGIApp, Receive, Scope, Send

from switchyard import metrics

logger = logging.getLogger(__name__)


class LoopLagMonitor:
    # Defaults give a ~0.5 s time constant: a single stall (a GC pause, a first-use import, a
    # burst of simultaneous arrivals) moves the average by a tenth of its size, while
    # saturation lasting a few hundred ms trips the threshold. An earlier, faster setting
    # (20 ms probes, smoothing 0.3) shed requests on transients during load tests (ADR-020).
    def __init__(self, interval_s: float = 0.05, smoothing: float = 0.1) -> None:
        self.interval_s = interval_s
        self.smoothing = smoothing
        self.lag_s = 0.0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="loop-lag-monitor")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def observe(self, lag_s: float) -> None:
        # Exponentially weighted: reacts within a few probes but ignores a single GC pause.
        self.lag_s += self.smoothing * (max(lag_s, 0.0) - self.lag_s)
        metrics.LOOP_LAG.set(self.lag_s)

    async def _run(self) -> None:
        while True:
            started = time.perf_counter()
            await asyncio.sleep(self.interval_s)
            self.observe(time.perf_counter() - started - self.interval_s)


class LoadShedMiddleware:
    """Refuses new requests on ``paths`` while the monitor reports overload."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        monitor: LoopLagMonitor,
        max_lag_s: float,
        paths: frozenset[str] = frozenset({"/v1/chat/completions"}),
    ) -> None:
        self.app = app
        self.monitor = monitor
        self.max_lag_s = max_lag_s
        self.paths = paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["path"] in self.paths
            and self.monitor.lag_s > self.max_lag_s
        ):
            metrics.SHED.inc()
            body = json.dumps(
                {
                    "error": {
                        "message": "The gateway is overloaded. Retry shortly.",
                        "type": "service_unavailable",
                        "param": None,
                        "code": "gateway_overloaded",
                    }
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"retry-after", b"1"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
