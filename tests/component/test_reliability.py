"""Retries, failover and circuit breaking through real sockets."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tests.conftest import MockServer

pytestmark = pytest.mark.usefixtures("mocks")

BODY = {"model": "mock", "messages": [{"role": "user", "content": "hi"}]}
STREAM_BODY = BODY | {"stream": True}
Post = Callable[..., httpx.Response]


@pytest.fixture
def headers(api_key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {api_key}"}


@pytest.fixture
def post(headers: dict[str, str]) -> Post:
    def send(url: str, body: dict[str, Any] = BODY) -> httpx.Response:
        return httpx.post(f"{url}/v1/chat/completions", json=body, headers=headers, timeout=10)

    return send


def test_failover_to_secondary_after_retries(
    post: Post, make_gateway: Callable[..., str], mocks: tuple[MockServer, MockServer]
) -> None:
    mock_a, mock_b = mocks
    mock_a.configure(error_rate=1.0, error_status=503)
    url = make_gateway()
    resp = post(url)
    assert resp.status_code == 200
    assert resp.headers["x-switchyard-provider"] == "mock-b"
    assert resp.headers["x-switchyard-attempts"] == "3"
    assert resp.headers["x-switchyard-failovers"] == "1"
    assert mock_a.stats()["requests"] == 2  # max_attempts=2
    assert mock_b.stats()["requests"] == 1


def test_stream_fails_over_when_primary_hangs(
    post: Post, make_gateway: Callable[..., str], mocks: tuple[MockServer, MockServer]
) -> None:
    mock_a, _ = mocks
    mock_a.configure(timeout_rate=1.0)
    url = make_gateway(timeouts={"first_byte_s": 0.3}, retry={"max_attempts": 1})
    started = time.perf_counter()
    resp = post(url, STREAM_BODY)
    elapsed = time.perf_counter() - started
    assert resp.status_code == 200
    assert resp.headers["x-switchyard-provider"] == "mock-b"
    assert resp.text.rstrip().endswith("data: [DONE]")
    assert 0.3 <= elapsed < 1.0  # one first-byte timeout, then the secondary


def test_connection_refused_fails_over(post: Post, make_gateway: Callable[..., str]) -> None:
    """A provider that is down entirely (nothing listening) is a fast failover."""
    from mock_provider.app import create_app as create_mock_app
    from switchyard.main import create_app as create_gateway_app
    from tests.conftest import gateway_config, run_server

    with run_server(create_mock_app()) as healthy:
        config = gateway_config("http://127.0.0.1:1", healthy)
        with run_server(create_gateway_app(config)) as url:
            resp = post(url)
    assert resp.status_code == 200
    assert resp.headers["x-switchyard-provider"] == "mock-b"


def test_breaker_opens_sheds_load_then_recovers(
    post: Post, make_gateway: Callable[..., str], mocks: tuple[MockServer, MockServer]
) -> None:
    mock_a, mock_b = mocks
    mock_a.configure(error_rate=1.0)
    url = make_gateway(
        retry={"max_attempts": 1},
        breaker={"window_size": 4, "min_calls": 4, "failure_rate_threshold": 0.5, "open_s": 0.5},
    )

    for _ in range(4):
        assert post(url).headers["x-switchyard-provider"] == "mock-b"
    assert mock_a.stats()["requests"] == 4
    status = httpx.get(f"{url}/status/providers").json()
    assert status["mock-a"]["circuit"] == "open"

    # While open, the primary is not called at all: no latency paid on a known-bad provider.
    for _ in range(5):
        resp = post(url)
        assert resp.headers["x-switchyard-provider"] == "mock-b"
    assert mock_a.stats()["requests"] == 4

    # Primary recovers; after open_s one trial request closes the breaker again.
    mock_a.configure(error_rate=0.0)
    time.sleep(0.6)
    assert post(url).headers["x-switchyard-provider"] == "mock-a"
    assert httpx.get(f"{url}/status/providers").json()["mock-a"]["circuit"] == "closed"
    assert mock_b.stats()["requests"] == 9


def test_failed_half_open_probe_reopens(
    post: Post, make_gateway: Callable[..., str], mocks: tuple[MockServer, MockServer]
) -> None:
    mock_a, _ = mocks
    mock_a.configure(error_rate=1.0)
    url = make_gateway(
        retry={"max_attempts": 1},
        breaker={"window_size": 2, "min_calls": 2, "failure_rate_threshold": 1.0, "open_s": 0.3},
    )
    post(url)
    post(url)
    time.sleep(0.35)
    post(url)  # probe fails
    assert mock_a.stats()["requests"] == 3
    post(url)
    assert mock_a.stats()["requests"] == 3  # open again, not probing on every request


def test_all_circuits_open_returns_503(
    post: Post, make_gateway: Callable[..., str], mocks: tuple[MockServer, MockServer]
) -> None:
    for mock in mocks:
        mock.configure(error_rate=1.0)
    url = make_gateway(
        retry={"max_attempts": 1},
        breaker={"window_size": 1, "min_calls": 1, "failure_rate_threshold": 1.0, "open_s": 30},
    )
    assert post(url).status_code == 502  # both fail, both breakers trip
    resp = post(url)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "all_circuits_open"
    assert int(resp.headers["retry-after"]) >= 1


def test_mid_stream_abort_is_not_retried_and_does_not_hang(
    post: Post, make_gateway: Callable[..., str], mocks: tuple[MockServer, MockServer]
) -> None:
    mock_a, mock_b = mocks
    mock_a.configure(stream_abort_rate=1.0, fail_after_tokens=3)
    url = make_gateway()
    started = time.perf_counter()
    resp = post(url, STREAM_BODY)
    assert time.perf_counter() - started < 1.0
    events = [line[6:] for line in resp.text.split("\n") if line.startswith("data: ")]
    assert json.loads(events[-1])["error"]["code"] == "connection_dropped"
    # Output from one model must not be spliced with another's (ADR-003).
    assert mock_b.stats()["requests"] == 0
    assert mock_a.stats()["requests"] == 1


def test_client_disconnect_closes_upstream_stream(
    headers: dict[str, str], make_gateway: Callable[..., str], mocks: tuple[MockServer, MockServer]
) -> None:
    mock_a, _ = mocks
    mock_a.configure(inter_token_ms=50, output_tokens=200)
    url = make_gateway()
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json=STREAM_BODY, headers=headers, timeout=10
    ) as r:
        for line in r.iter_lines():
            if '"content"' in line:
                break  # hang up after the first token
    deadline = time.monotonic() + 2
    while mock_a.stats()["in_flight"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert mock_a.stats()["in_flight"] == 0  # the gateway hung up on the upstream too
