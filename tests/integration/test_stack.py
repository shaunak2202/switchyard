"""Smoke tests against the running Docker Compose stack (``make up`` first).

These exercise the real container images and network, which the in-process component tests
cannot: Dockerfile, compose wiring, config interpolation and uvloop/httptools in production mode.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.integration

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8000")
MOCK_PRIMARY = os.environ.get("MOCK_PRIMARY_ADMIN_URL", "http://localhost:9001")
MESSAGES = [{"role": "user", "content": "integration"}]


@pytest.fixture(autouse=True)
def stack() -> Iterator[None]:
    try:
        httpx.get(f"{GATEWAY}/readyz", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("compose stack is not running (make up)")
    httpx.post(f"{MOCK_PRIMARY}/admin/reset")
    yield
    httpx.post(f"{MOCK_PRIMARY}/admin/reset")


def test_completion_through_containers() -> None:
    resp = httpx.post(
        f"{GATEWAY}/v1/chat/completions", json={"model": "mock", "messages": MESSAGES}, timeout=10
    )
    assert resp.status_code == 200
    assert resp.headers["x-switchyard-provider"] == "mock-primary"


def test_stream_through_containers() -> None:
    with httpx.stream(
        "POST",
        f"{GATEWAY}/v1/chat/completions",
        json={"model": "mock", "messages": MESSAGES, "stream": True},
        timeout=10,
    ) as resp:
        events = [line[6:] for line in resp.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert all("choices" in json.loads(e) for e in events[:-1])


def test_kill_mock_mid_stream_does_not_hang_client() -> None:
    httpx.patch(
        f"{MOCK_PRIMARY}/admin/config", json={"stream_abort_rate": 1.0, "fail_after_tokens": 3}
    ).raise_for_status()
    resp = httpx.post(
        f"{GATEWAY}/v1/chat/completions",
        json={"model": "mock-primary/mock-1", "messages": MESSAGES, "stream": True},
        timeout=10,
    )
    events = [line[6:] for line in resp.text.split("\n") if line.startswith("data: ")]
    assert "[DONE]" not in events
    assert "error" in json.loads(events[-1])
