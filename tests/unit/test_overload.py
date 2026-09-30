from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from switchyard.overload import LoadShedMiddleware, LoopLagMonitor


def test_smoothing_ignores_a_single_spike_but_follows_sustained_lag() -> None:
    monitor = LoopLagMonitor(smoothing=0.3)
    monitor.observe(0.5)  # one GC pause
    monitor.observe(0.0)
    monitor.observe(0.0)
    assert monitor.lag_s < 0.1
    for _ in range(15):
        monitor.observe(0.05)
    assert monitor.lag_s == pytest.approx(0.05, abs=0.005)


async def test_monitor_detects_a_blocked_loop() -> None:
    monitor = LoopLagMonitor(interval_s=0.05, smoothing=1.0)
    monitor.start()
    await asyncio.sleep(0.01)
    time.sleep(0.2)  # noqa: ASYNC251 - deliberately block the loop, as CPU-bound work would
    await asyncio.sleep(0.01)  # the overdue probe runs; the next one is not due yet
    assert monitor.lag_s > 0.1
    await monitor.stop()


async def _client(lag: float) -> httpx.AsyncClient:
    monitor = LoopLagMonitor()
    monitor.lag_s = lag
    inner = Starlette(
        routes=[
            Route("/v1/chat/completions", lambda _: PlainTextResponse("ok"), methods=["POST"]),
            Route("/healthz", lambda _: PlainTextResponse("ok")),
        ]
    )
    app = LoadShedMiddleware(inner, monitor=monitor, max_lag_s=0.025)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_sheds_chat_requests_only_while_overloaded() -> None:
    async with await _client(lag=0.1) as client:
        shed = await client.post("/v1/chat/completions")
        assert shed.status_code == 503
        assert shed.headers["retry-after"] == "1"
        assert shed.json()["error"]["code"] == "gateway_overloaded"
        assert (await client.get("/healthz")).status_code == 200  # probes are never shed
    async with await _client(lag=0.001) as client:
        assert (await client.post("/v1/chat/completions")).status_code == 200
