"""Liveness, readiness and provider status.

* ``/healthz``: the process is up and the event loop is responsive. Never checks dependencies,
  so a Redis blip does not get the gateway restarted.
* ``/readyz``: the gateway can serve traffic, i.e. its *own* dependencies are available.
  Upstream providers are deliberately excluded: a Groq outage is what failover is for, and
  marking every replica unready because of it would turn a partial outage into a total one.
* ``/status/providers``: an on-demand probe of each provider for humans and dashboards.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.exceptions import RedisError

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    checks: dict[str, bool] = {"config": True, "redis": await _redis_ok(request)}
    ready = all(checks.values())
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "checks": checks},
        status_code=200 if ready else 503,
    )


async def _redis_ok(request: Request) -> bool:
    try:
        async with asyncio.timeout(1.0):
            return bool(await request.app.state.redis.ping())
    except (RedisError, OSError, TimeoutError):
        return False


@router.get("/status/providers")
async def provider_status(request: Request) -> dict[str, dict[str, object]]:
    providers = request.app.state.providers
    breakers = request.app.state.breakers
    names = list(providers)
    results = await asyncio.gather(*(providers[name].health() for name in names))
    return {
        name: {
            "healthy": healthy,
            "circuit": breakers[name].state.name.lower(),
            "failure_rate": round(breakers[name].failure_rate(), 3),
        }
        for name, healthy in zip(names, results, strict=True)
    }


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
