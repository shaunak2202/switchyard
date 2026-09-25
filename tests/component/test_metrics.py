"""The metrics the dashboard and the load tests depend on actually move."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from switchyard.cache.embeddings import ExactTextVerifier, HashingEmbedder
from tests.conftest import MockServer, create_key

pytestmark = pytest.mark.usefixtures("mocks")

BODY: dict[str, Any] = {"model": "mock", "messages": [{"role": "user", "content": "metrics"}]}


def scrape(url: str) -> dict[tuple[str, frozenset[tuple[str, str]]], float]:
    text = httpx.get(f"{url}/metrics").text
    samples = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            samples[(s.name, frozenset(s.labels.items()))] = s.value
    return samples


def value(samples: dict[Any, float], name: str, **labels: str) -> float:
    """Sum of all samples of ``name`` whose labels include ``labels``."""
    want = set(labels.items())
    return sum(v for (n, lbls), v in samples.items() if n == name and want <= set(lbls))


@pytest.fixture
def post(api_key: str) -> Callable[..., httpx.Response]:
    def send(url: str, body: dict[str, Any] = BODY, key: str | None = None) -> httpx.Response:
        headers = {"authorization": f"Bearer {key or api_key}"}
        return httpx.post(f"{url}/v1/chat/completions", json=body, headers=headers, timeout=10)

    return send


def test_request_counters_and_latency(
    gateway_url: str, post: Callable[..., httpx.Response]
) -> None:
    before = scrape(gateway_url)
    assert post(gateway_url).status_code == 200
    assert post(gateway_url, {**BODY, "model": "gpt-nope"}).status_code == 404
    after = scrape(gateway_url)

    def delta(name: str, **labels: str) -> float:
        return value(after, name, **labels) - value(before, name, **labels)

    assert delta("switchyard_requests_total", route="mock", stream="false", status="200") == 1
    assert delta("switchyard_requests_total", route="unknown", status="404") == 1
    assert delta("switchyard_request_duration_seconds_count", route="mock") == 1
    assert delta("switchyard_gateway_overhead_seconds_count", stream="false") == 1
    assert delta("switchyard_upstream_attempts_total", provider="mock-a", outcome="success") == 1
    assert delta("switchyard_tokens_total", provider="mock-a", type="completion") == 8


def test_streaming_ttft_and_overhead(gateway_url: str, post: Callable[..., httpx.Response]) -> None:
    before = scrape(gateway_url)
    resp = post(gateway_url, {**BODY, "stream": True})
    assert resp.text.endswith("data: [DONE]\n\n")
    after = scrape(gateway_url)
    name = "switchyard_time_to_first_token_seconds_count"
    assert value(after, name, route="mock") - value(before, name, route="mock") == 1
    name = "switchyard_gateway_overhead_seconds_count"
    assert value(after, name, stream="true") - value(before, name, stream="true") == 1


def test_pinned_models_do_not_create_unbounded_labels(
    gateway_url: str, post: Callable[..., httpx.Response]
) -> None:
    post(gateway_url, {**BODY, "model": "mock-a/some-arbitrary-model-string"})
    samples = scrape(gateway_url)
    routes = {dict(lbls).get("route") for (n, lbls) in samples if n == "switchyard_requests_total"}
    assert "mock-a/*" in routes
    assert not any(r and "arbitrary" in r for r in routes)


def test_rate_limit_and_auth_counters(
    gateway_url: str, post: Callable[..., httpx.Response]
) -> None:
    key = create_key(rpm=1, tpm=100_000)
    before = scrape(gateway_url)
    post(gateway_url, key=key)
    assert post(gateway_url, key=key).status_code == 429
    assert post(gateway_url, key="sk-sy-bogus").status_code == 401
    after = scrape(gateway_url)
    name = "switchyard_rate_limited_total"
    assert value(after, name, limit="requests") - value(before, name, limit="requests") == 1
    name = "switchyard_auth_failures_total"
    assert value(after, name, reason="unknown_key") - value(before, name, reason="unknown_key") == 1


def test_failover_errors_and_circuit_state(
    make_gateway: Callable[..., str],
    post: Callable[..., httpx.Response],
    mocks: tuple[MockServer, MockServer],
) -> None:
    mocks[0].configure(error_rate=1.0)
    url = make_gateway(
        retry={"max_attempts": 1},
        breaker={"window_size": 2, "min_calls": 2, "failure_rate_threshold": 1.0, "open_s": 30},
    )
    before = scrape(url)
    for _ in range(3):
        assert post(url).headers["x-switchyard-provider"] == "mock-b"
    after = scrape(url)

    def delta(name: str, **labels: str) -> float:
        return value(after, name, **labels) - value(before, name, **labels)

    assert delta("switchyard_failovers_total", from_provider="mock-a", to_provider="mock-b") == 3
    assert delta("switchyard_upstream_errors_total", provider="mock-a", kind="upstream_5xx") == 2
    assert (
        delta("switchyard_upstream_attempts_total", provider="mock-a", outcome="circuit_open") == 1
    )
    assert value(after, "switchyard_circuit_state", provider="mock-a") == 2
    assert delta("switchyard_circuit_transitions_total", provider="mock-a", to_state="open") == 1


def test_mid_stream_failure_counter(
    gateway_url: str, post: Callable[..., httpx.Response], mocks: tuple[MockServer, MockServer]
) -> None:
    mocks[0].configure(stream_abort_rate=1.0, fail_after_tokens=2)
    before = scrape(gateway_url)
    post(gateway_url, {**BODY, "model": "only-a", "stream": True})
    after = scrape(gateway_url)
    name = "switchyard_mid_stream_failures_total"
    assert value(after, name, provider="mock-a") - value(before, name, provider="mock-a") == 1


def test_cache_counters(
    make_gateway: Callable[..., str], post: Callable[..., httpx.Response]
) -> None:
    url = make_gateway(
        embedder=HashingEmbedder(),
        verifier=ExactTextVerifier(),
        cache={
            "exact": {"enabled": True},
            "semantic": {"enabled": True, "candidate_threshold": 0.5, "verifier_threshold": 0.5},
        },
    )
    tag = time.monotonic_ns()
    q = {"model": "mock", "temperature": 0}
    before = scrape(url)
    post(url, q | {"messages": [{"role": "user", "content": f"Convert 5 km to miles {tag}"}]})
    time.sleep(0.15)
    post(url, q | {"messages": [{"role": "user", "content": f"Convert 5 km to miles {tag}"}]})
    post(url, q | {"messages": [{"role": "user", "content": f"convert 5 KM to miles {tag}!"}]})
    post(url, q | {"messages": [{"role": "user", "content": f"Convert 5 miles to km {tag}"}]})
    after = scrape(url)

    def delta(name: str, **labels: str) -> float:
        return value(after, name, **labels) - value(before, name, **labels)

    assert delta("switchyard_cache_lookups_total", tier="exact", result="hit") == 1
    assert delta("switchyard_cache_lookups_total", tier="exact", result="miss") == 3
    assert delta("switchyard_cache_lookups_total", tier="semantic", result="hit") == 1
    assert delta("switchyard_cache_semantic_rejections_total") == 1  # the reversed conversion
    assert delta("switchyard_requests_total", cache="hit-semantic") == 1
    assert delta("switchyard_cache_model_seconds_count", model="embedding") >= 3
