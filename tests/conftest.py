from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import uvicorn
from redis import Redis as SyncRedis
from redis.asyncio import Redis
from starlette.types import ASGIApp

from mock_provider.app import MockSettings
from mock_provider.app import create_app as create_mock_app
from switchyard.auth.keys import KeyStore
from switchyard.config import GatewayConfig
from switchyard.main import create_app as create_gateway_app

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# Every test session gets its own key namespace, so tests never touch a developer's real keys
# and parallel CI jobs sharing one Redis cannot interfere.
REDIS_PREFIX = f"sy-test-{uuid.uuid4().hex[:8]}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


@contextmanager
def run_server(app: ASGIApp) -> Iterator[str]:
    """Serve ``app`` with a real uvicorn server on a background thread."""
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("test server failed to start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@dataclass
class MockServer:
    url: str

    def configure(self, **settings: Any) -> None:
        httpx.patch(f"{self.url}/admin/config", json=settings).raise_for_status()

    def reset(self) -> None:
        httpx.post(f"{self.url}/admin/reset").raise_for_status()

    def stats(self) -> dict[str, int]:
        data: dict[str, int] = httpx.get(f"{self.url}/admin/stats").json()
        return data


FAST_MOCK = MockSettings(ttft_ms=5, inter_token_ms=1, output_tokens=8, hang_s=30, seed=7)


@pytest.fixture(scope="session")
def mock_a() -> Iterator[MockServer]:
    with run_server(create_mock_app(FAST_MOCK)) as url:
        yield MockServer(url)


@pytest.fixture(scope="session")
def mock_b() -> Iterator[MockServer]:
    with run_server(create_mock_app(FAST_MOCK)) as url:
        yield MockServer(url)


# The shared gateway's breaker never trips, so one test's injected failures can't leak into
# the next. Reliability tests build their own gateway with ``make_gateway``.
NEVER_TRIP = {"window_size": 10_000, "min_calls": 10_000}
FAST_RETRY = {"max_attempts": 2, "base_delay_s": 0.01, "max_delay_s": 0.05}


def gateway_config(
    mock_a_url: str,
    mock_b_url: str,
    *,
    timeouts: dict[str, float] | None = None,
    retry: dict[str, float] | None = None,
    breaker: dict[str, float] | None = None,
    request_timeout_s: float = 10,
    auth_enabled: bool = True,
) -> GatewayConfig:
    t = {"connect_s": 1, "first_byte_s": 1, "idle_s": 0.5, "total_s": 2} | (timeouts or {})
    return GatewayConfig.model_validate(
        {
            "logging": {"level": "WARNING"},
            "redis": {"url": REDIS_URL, "key_prefix": REDIS_PREFIX},
            "auth": {"enabled": auth_enabled, "default_max_tokens": 64},
            "reliability": {
                "request_timeout_s": request_timeout_s,
                "retry": FAST_RETRY | (retry or {}),
                "circuit_breaker": breaker or NEVER_TRIP,
            },
            "providers": {
                "mock-a": {"type": "mock", "base_url": f"{mock_a_url}/v1", "timeouts": t},
                "mock-b": {"type": "mock", "base_url": f"{mock_b_url}/v1", "timeouts": t},
            },
            "routes": [
                {
                    "model": "mock",
                    "targets": [
                        {"provider": "mock-a", "model": "mock-1"},
                        {"provider": "mock-b", "model": "mock-1"},
                    ],
                },
                {"model": "only-a", "targets": [{"provider": "mock-a", "model": "mock-1"}]},
            ],
        }
    )


@pytest.fixture(scope="session", autouse=True)
def _redis_namespace() -> Iterator[None]:
    yield
    client = SyncRedis.from_url(REDIS_URL)
    try:
        keys = list(client.scan_iter(f"{REDIS_PREFIX}:*", count=1000))
        if keys:
            client.delete(*keys)
    except Exception:  # Redis may be down if only unit tests ran
        pass


def create_key(*, rpm: int = 1_000_000, tpm: int = 1_000_000_000, name: str = "test") -> str:
    async def run() -> str:
        redis = Redis.from_url(REDIS_URL)
        try:
            api_key, _ = await KeyStore(redis, REDIS_PREFIX).create(name, rpm=rpm, tpm=tpm)
            return api_key
        finally:
            await redis.aclose()

    # A worker thread has no running event loop, so this works from sync and async tests alike.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run()).result()


@pytest.fixture(scope="session")
def api_key() -> str:
    """A key whose limits are never hit, for tests that are not about rate limiting."""
    return create_key(name="session")


@pytest.fixture(scope="session")
def gateway_url(mock_a: MockServer, mock_b: MockServer) -> Iterator[str]:
    app = create_gateway_app(gateway_config(mock_a.url, mock_b.url))
    with run_server(app) as url:
        yield url


@pytest.fixture
def make_gateway(mock_a: MockServer, mock_b: MockServer) -> Iterator[Any]:
    """Start a dedicated gateway (fresh breakers) with custom reliability settings."""
    with ExitStack() as stack:

        def start(**kwargs: Any) -> str:
            app = create_gateway_app(gateway_config(mock_a.url, mock_b.url, **kwargs))
            return stack.enter_context(run_server(app))

        yield start


@pytest.fixture
def mocks(mock_a: MockServer, mock_b: MockServer) -> Iterator[tuple[MockServer, MockServer]]:
    mock_a.reset()
    mock_b.reset()
    yield mock_a, mock_b
    mock_a.reset()
    mock_b.reset()


@pytest.fixture
async def client(gateway_url: str, api_key: str) -> Any:
    headers = {"authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(base_url=gateway_url, timeout=10, headers=headers) as c:
        yield c
