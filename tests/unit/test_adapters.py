"""Adapter behaviour against scripted upstream responses (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from switchyard.config import ProviderConfig, Timeouts
from switchyard.errors import FailureKind, ProviderError
from switchyard.providers.groq import GroqProvider
from switchyard.providers.ollama import OllamaProvider
from switchyard.providers.openai_compat import OpenAICompatibleProvider
from switchyard.schemas import ChatCompletionRequest

Handler = Callable[[httpx.Request], httpx.Response]

REQUEST = ChatCompletionRequest.model_validate(
    {"model": "alias", "messages": [{"role": "user", "content": "hi"}]}
)
STREAM_REQUEST = REQUEST.model_copy(update={"stream": True})


def _provider(
    handler: Handler,
    cls: type[OpenAICompatibleProvider] = OpenAICompatibleProvider,
    **config: object,
) -> OpenAICompatibleProvider:
    cfg = ProviderConfig.model_validate(
        {"type": "openai", "base_url": "http://upstream/v1", "timeouts": Timeouts()} | config
    )
    return cls("up", cfg, transport=httpx.MockTransport(handler))


def _sse(*payloads: object) -> bytes:
    out = b""
    for payload in payloads:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        out += f"data: {data}\n\n".encode()
    return out


def _chunk(
    content: str | None = None, finish: str | None = None, **extra: object
) -> dict[str, object]:
    delta = {"content": content} if content is not None else {}
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }


async def _drain(provider: OpenAICompatibleProvider) -> list[str]:
    out: list[str] = []
    async for chunk in provider.stream(STREAM_REQUEST, "m"):
        out.extend(c["delta"].get("content", "") for c in chunk.choices)
    return out


async def test_sends_target_model_and_bearer_key() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"id": "x", "created": 1, "model": "m", "choices": []},
        )

    provider = _provider(handler, api_key="sk-test")
    await provider.complete(REQUEST, "upstream-model")
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {
        "model": "upstream-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


async def test_429_carries_retry_after() -> None:
    provider = _provider(
        lambda _: httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})
    )
    with pytest.raises(ProviderError) as exc:
        await provider.complete(REQUEST, "m")
    assert exc.value.kind is FailureKind.RATE_LIMITED
    assert exc.value.retry_after_s == 7.0
    assert exc.value.message == "slow down"


async def test_non_json_error_body() -> None:
    provider = _provider(lambda _: httpx.Response(502, text="<html>bad gateway</html>"))
    with pytest.raises(ProviderError) as exc:
        await provider.complete(REQUEST, "m")
    assert exc.value.kind is FailureKind.UPSTREAM_5XX
    assert "bad gateway" in exc.value.message


async def test_malformed_completion_is_protocol_error() -> None:
    provider = _provider(lambda _: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderError) as exc:
        await provider.complete(REQUEST, "m")
    assert exc.value.kind is FailureKind.PROTOCOL


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (httpx.ConnectError("refused"), FailureKind.CONNECT),
        (httpx.ReadTimeout("slow"), FailureKind.TIMEOUT),
        (httpx.RemoteProtocolError("peer closed"), FailureKind.CONNECTION_DROPPED),
    ],
)
async def test_transport_errors_are_classified(exc: Exception, kind: FailureKind) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(ProviderError) as err:
        await _provider(handler).complete(REQUEST, "m")
    assert err.value.kind is kind
    with pytest.raises(ProviderError) as err:
        await _drain(_provider(handler))
    assert err.value.kind is kind


async def test_stream_happy_path() -> None:
    body = _sse(_chunk("a"), _chunk("b"), _chunk(finish="stop"), "[DONE]")
    provider = _provider(lambda _: httpx.Response(200, content=body))
    assert await _drain(provider) == ["a", "b", ""]


async def test_stream_error_status_before_first_byte() -> None:
    provider = _provider(lambda _: httpx.Response(503, json={"error": {"message": "overloaded"}}))
    with pytest.raises(ProviderError) as exc:
        await _drain(provider)
    assert exc.value.kind is FailureKind.UPSTREAM_5XX
    assert exc.value.message == "overloaded"


async def test_stream_mid_stream_error_event() -> None:
    body = _sse(_chunk("a"), {"error": {"message": "model crashed"}})
    provider = _provider(lambda _: httpx.Response(200, content=body))
    with pytest.raises(ProviderError, match="model crashed") as exc:
        await _drain(provider)
    assert exc.value.kind is FailureKind.UPSTREAM_5XX


async def test_stream_truncated_without_finish_is_dropped_connection() -> None:
    provider = _provider(lambda _: httpx.Response(200, content=_sse(_chunk("a"))))
    with pytest.raises(ProviderError) as exc:
        await _drain(provider)
    assert exc.value.kind is FailureKind.CONNECTION_DROPPED


async def test_stream_finished_without_done_sentinel_is_accepted() -> None:
    provider = _provider(lambda _: httpx.Response(200, content=_sse(_chunk("a", finish="stop"))))
    assert await _drain(provider) == ["a"]


@pytest.mark.parametrize("bad", ["not json", json.dumps({"id": "only"})])
async def test_stream_garbage_is_protocol_error(bad: str) -> None:
    provider = _provider(lambda _: httpx.Response(200, content=_sse(bad)))
    with pytest.raises(ProviderError) as exc:
        await _drain(provider)
    assert exc.value.kind is FailureKind.PROTOCOL


async def test_groq_usage_extension_is_normalised() -> None:
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    body = _sse(_chunk("a"), _chunk(finish="stop", x_groq={"usage": usage}), "[DONE]")
    provider = _provider(lambda _: httpx.Response(200, content=body), GroqProvider)
    chunks = [c async for c in provider.stream(STREAM_REQUEST, "m")]
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 5


async def test_ollama_health_uses_native_api() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"models": []})

    assert await _provider(handler, OllamaProvider).health()
    assert paths == ["/api/tags"]


async def test_health_false_on_connection_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert not await _provider(handler).health()
    assert not await _provider(handler, OllamaProvider).health()
