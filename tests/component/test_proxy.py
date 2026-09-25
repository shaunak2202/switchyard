"""End-to-end through a real gateway and a real mock provider over local sockets."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam

from tests.conftest import MockServer

pytestmark = pytest.mark.usefixtures("mocks")

MESSAGES: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Say hello"}]


def _sse_events(text: str) -> list[str]:
    return [line.removeprefix("data: ") for line in text.split("\n") if line.startswith("data: ")]


async def test_non_streaming_completion(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/chat/completions", json={"model": "mock", "messages": MESSAGES})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] == 8
    assert resp.headers["x-switchyard-provider"] == "mock-a"
    assert resp.headers["x-switchyard-model"] == "mock-1"
    assert float(resp.headers["x-switchyard-upstream-ms"]) > 0
    assert len(resp.headers["x-request-id"]) == 32


async def test_caller_request_id_is_propagated(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "mock", "messages": MESSAGES},
        headers={"x-request-id": "trace-abc-123"},
    )
    assert resp.headers["x-request-id"] == "trace-abc-123"


async def test_streaming_relays_chunks_and_done(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions", json={"model": "mock", "messages": MESSAGES, "stream": True}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(resp.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert len(content.split()) == 8
    # The gateway asked upstream for usage but the caller didn't, so it is stripped.
    assert all(c["choices"] for c in chunks)


async def test_streaming_usage_when_requested(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "mock",
            "messages": MESSAGES,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    chunks = [json.loads(e) for e in _sse_events(resp.text)[:-1]]
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 8


async def test_streaming_actually_streams(client: httpx.AsyncClient, mocks: Any) -> None:
    """The first chunk must reach the client long before the last one is generated."""
    mock_a: MockServer = mocks[0]
    mock_a.configure(ttft_ms=10, inter_token_ms=100, output_tokens=5)
    started = time.perf_counter()
    first_at = None
    async with client.stream(
        "POST", "/v1/chat/completions", json={"model": "mock", "messages": MESSAGES, "stream": True}
    ) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and first_at is None:
                first_at = time.perf_counter() - started
    total = time.perf_counter() - started
    assert first_at is not None
    assert first_at < 0.2
    assert total >= 0.4


def test_openai_sdk_compatibility(gateway_url: str) -> None:
    client = openai.OpenAI(base_url=f"{gateway_url}/v1", api_key="unused", max_retries=0)
    completion = client.chat.completions.create(model="mock", messages=MESSAGES)
    assert completion.choices[0].message.content

    stream = client.chat.completions.create(model="mock", messages=MESSAGES, stream=True)
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert text == completion.choices[0].message.content  # mock output is deterministic

    assert "mock" in [m.id for m in client.models.list()]


async def test_unknown_model_is_openai_style_404(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/chat/completions", json={"model": "gpt-9", "messages": MESSAGES})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"model": "mock", "messages": []}, "messages"),
        ({"model": "mock", "messages": MESSAGES, "temperature": 5}, "temperature"),
        ({"messages": MESSAGES}, "model"),
    ],
)
async def test_invalid_request_is_openai_style_400(
    client: httpx.AsyncClient, payload: dict[str, Any], param: str
) -> None:
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param


async def test_upstream_500_becomes_502(client: httpx.AsyncClient, mocks: Any) -> None:
    mocks[0].configure(error_rate=1.0, error_status=500)
    resp = await client.post("/v1/chat/completions", json={"model": "only-a", "messages": MESSAGES})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_5xx"


async def test_upstream_400_is_passed_through(client: httpx.AsyncClient, mocks: Any) -> None:
    mocks[0].configure(error_rate=1.0, error_status=400)
    resp = await client.post("/v1/chat/completions", json={"model": "only-a", "messages": MESSAGES})
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "injected failure"


async def test_hanging_upstream_times_out_with_504(client: httpx.AsyncClient, mocks: Any) -> None:
    mocks[0].configure(timeout_rate=1.0)
    started = time.perf_counter()
    resp = await client.post(
        "/v1/chat/completions", json={"model": "only-a", "messages": MESSAGES, "stream": True}
    )
    assert resp.status_code == 504
    # first_byte_s=1, one retry: two bounded attempts, not a hang.
    assert time.perf_counter() - started < 2.5
    assert mocks[0].stats()["requests"] == 2


@pytest.mark.parametrize("fault", ["stream_abort_rate", "stream_stall_rate"])
async def test_mid_stream_failure_ends_stream_with_error_event(
    client: httpx.AsyncClient, mocks: Any, fault: str
) -> None:
    mocks[0].configure(**{fault: 1.0, "fail_after_tokens": 3})
    started = time.perf_counter()
    resp = await client.post(
        "/v1/chat/completions", json={"model": "only-a", "messages": MESSAGES, "stream": True}
    )
    elapsed = time.perf_counter() - started
    assert resp.status_code == 200  # committed before the failure
    events = _sse_events(resp.text)
    assert "[DONE]" not in events
    assert json.loads(events[-1])["error"]["type"] == "upstream_error"
    assert elapsed < 1.5  # a stall is cut off by idle_s=0.5, not left hanging


def test_openai_sdk_raises_on_mid_stream_error(gateway_url: str, mocks: Any) -> None:
    mocks[0].configure(stream_abort_rate=1.0, fail_after_tokens=2)
    client = openai.OpenAI(base_url=f"{gateway_url}/v1", api_key="unused", max_retries=0)
    stream = client.chat.completions.create(model="only-a", messages=MESSAGES, stream=True)
    with pytest.raises(openai.APIError):
        for _ in stream:
            pass


async def test_health_endpoints(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    status = (await client.get("/status/providers")).json()
    assert status["mock-a"]["healthy"] is True
    assert status["mock-b"]["circuit"] == "closed"
