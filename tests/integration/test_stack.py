"""Smoke tests against the running Docker Compose stack (``make up`` first).

These exercise the real container images and network, which the in-process component tests
cannot: Dockerfile, compose wiring, config interpolation and uvloop/httptools in production mode.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.integration

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8000")
MOCK_PRIMARY = os.environ.get("MOCK_PRIMARY_ADMIN_URL", "http://localhost:9001")
MESSAGES = [{"role": "user", "content": "integration"}]


@pytest.fixture(scope="module")
def api_key() -> str:
    """Issue a key the way an operator would: the CLI inside the gateway container."""
    try:
        httpx.get(f"{GATEWAY}/readyz", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("compose stack is not running (make up)")
    out = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "gateway",
            "python",
            "-m",
            "switchyard.cli",
            "keys",
            "create",
            "--name",
            "integration",
            "--rpm",
            "100000",
            "--tpm",
            "100000000",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    key: str = json.loads(out)["api_key"]
    return key


@pytest.fixture(autouse=True)
def stack(api_key: str) -> Iterator[None]:
    httpx.post(f"{MOCK_PRIMARY}/admin/reset")
    yield
    httpx.post(f"{MOCK_PRIMARY}/admin/reset")


@pytest.fixture
def auth(api_key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {api_key}"}


def test_completion_through_containers(auth: dict[str, str]) -> None:
    resp = httpx.post(
        f"{GATEWAY}/v1/chat/completions",
        json={"model": "mock", "messages": MESSAGES},
        timeout=10,
        headers=auth,
    )
    assert resp.status_code == 200
    assert resp.headers["x-switchyard-provider"] == "mock-primary"


def test_stream_through_containers(auth: dict[str, str]) -> None:
    with httpx.stream(
        "POST",
        f"{GATEWAY}/v1/chat/completions",
        json={"model": "mock", "messages": MESSAGES, "stream": True},
        timeout=10,
        headers=auth,
    ) as resp:
        events = [line[6:] for line in resp.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert all("choices" in json.loads(e) for e in events[:-1])


def test_kill_mock_mid_stream_does_not_hang_client(auth: dict[str, str]) -> None:
    httpx.patch(
        f"{MOCK_PRIMARY}/admin/config", json={"stream_abort_rate": 1.0, "fail_after_tokens": 3}
    ).raise_for_status()
    resp = httpx.post(
        f"{GATEWAY}/v1/chat/completions",
        json={"model": "mock-primary/mock-1", "messages": MESSAGES, "stream": True},
        timeout=10,
        headers=auth,
    )
    events = [line[6:] for line in resp.text.split("\n") if line.startswith("data: ")]
    assert "[DONE]" not in events
    assert "error" in json.loads(events[-1])


def _compose(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        ["docker", "compose", *args],
        check=True,
        capture_output=True,
        timeout=120,
        env={**os.environ, **(env or {})},
    )


def test_rate_limit_through_containers() -> None:
    out = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "gateway",
            "python",
            "-m",
            "switchyard.cli",
            "keys",
            "create",
            "--name",
            "tiny",
            "--rpm",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    headers = {"authorization": f"Bearer {json.loads(out)['api_key']}"}
    codes = [
        httpx.post(
            f"{GATEWAY}/v1/chat/completions",
            json={"model": "mock", "messages": MESSAGES},
            headers=headers,
            timeout=10,
        ).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]


def test_kill_primary_container_mid_stream_then_fail_over(auth: dict[str, str]) -> None:
    """SIGKILL the primary mock's container while it is streaming to us."""
    httpx.patch(
        f"{MOCK_PRIMARY}/admin/config", json={"inter_token_ms": 100, "output_tokens": 100}
    ).raise_for_status()
    try:
        events: list[str] = []
        started = time.monotonic()
        with httpx.stream(
            "POST",
            f"{GATEWAY}/v1/chat/completions",
            json={"model": "mock", "messages": MESSAGES, "stream": True},
            timeout=30,
            headers=auth,
        ) as resp:
            assert resp.headers["x-switchyard-provider"] == "mock-primary"
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                events.append(line[6:])
                if len(events) == 3:
                    _compose("kill", "mock-primary")
        elapsed = time.monotonic() - started

        # The client gets an error event and the stream ends; it does not hang.
        assert "[DONE]" not in events
        assert json.loads(events[-1])["error"]["type"] == "upstream_error"
        assert elapsed < 10

        # New requests fail over to the secondary while the primary is down.
        resp2 = httpx.post(
            f"{GATEWAY}/v1/chat/completions",
            json={"model": "mock", "messages": MESSAGES},
            timeout=30,
            headers=auth,
        )
        assert resp2.status_code == 200
        assert resp2.headers["x-switchyard-provider"] == "mock-secondary"
        assert int(resp2.headers["x-switchyard-failovers"]) == 1
    finally:
        _compose("up", "-d", "--wait", "mock-primary")


# The models take ~100 ms per lookup in a CPU-only container, past the shipped 50 ms budget
# (ADR-021), so every lookup would be skipped as a miss. This test checks matching, not latency.
SEMANTIC_TEST_ENV = {
    "SWITCHYARD_SEMANTIC_ENABLED": "true",
    "SWITCHYARD_SEMANTIC_LOOKUP_BUDGET_MS": "2000",
}


@pytest.fixture
def semantic_gateway() -> Iterator[None]:
    """Recreate the gateway with the semantic tier on, then restore it as it was."""
    _compose("up", "-d", "--wait", "gateway", env=SEMANTIC_TEST_ENV)
    try:
        if not httpx.get(f"{GATEWAY}/status/cache", timeout=5).json()["semantic"]:
            pytest.skip("gateway image has no semantic-cache models (make up SEMANTIC=1)")
        yield
    finally:
        _compose("up", "-d", "--wait", "gateway")


@pytest.mark.usefixtures("semantic_gateway")
def test_semantic_cache_with_real_models(auth: dict[str, str]) -> None:
    """The baked-in embedding + verifier models, at the shipped thresholds.

    Each run gets a unique system prompt (part of the cache namespace, not embedded), so the
    user text stays natural: an appended random tag measurably lowers the verifier's score.
    """
    system = f"Test run {time.monotonic_ns()}."

    def ask(text: str) -> httpx.Response:
        return httpx.post(
            f"{GATEWAY}/v1/chat/completions",
            json={"model": "mock", "temperature": 0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": text}]},
            headers=auth,
            timeout=30,
        )  # fmt: skip

    assert ask("What is the capital of France?").headers["x-switchyard-cache"] == "miss"
    time.sleep(0.5)
    hit = ask("what is the capital of france")
    assert hit.headers["x-switchyard-cache"] == "hit-semantic"
    assert float(hit.headers["x-switchyard-cache-verifier-score"]) >= 0.992
    assert ask("What is the capital of Germany?").headers["x-switchyard-cache"] == "miss"
