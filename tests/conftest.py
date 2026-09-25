from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import uvicorn
from starlette.types import ASGIApp

from mock_provider.app import MockSettings
from mock_provider.app import create_app as create_mock_app
from switchyard.config import GatewayConfig
from switchyard.main import create_app as create_gateway_app


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


def gateway_config(mock_a_url: str, mock_b_url: str, **timeouts: float) -> GatewayConfig:
    t = {"connect_s": 1, "first_byte_s": 1, "idle_s": 0.5, "total_s": 2} | timeouts
    return GatewayConfig.model_validate(
        {
            "logging": {"level": "WARNING"},
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


@pytest.fixture(scope="session")
def gateway_url(mock_a: MockServer, mock_b: MockServer) -> Iterator[str]:
    app = create_gateway_app(gateway_config(mock_a.url, mock_b.url))
    with run_server(app) as url:
        yield url


@pytest.fixture
def mocks(mock_a: MockServer, mock_b: MockServer) -> Iterator[tuple[MockServer, MockServer]]:
    mock_a.reset()
    mock_b.reset()
    yield mock_a, mock_b
    mock_a.reset()
    mock_b.reset()


@pytest.fixture
async def client(gateway_url: str) -> Any:
    async with httpx.AsyncClient(base_url=gateway_url, timeout=10) as c:
        yield c
