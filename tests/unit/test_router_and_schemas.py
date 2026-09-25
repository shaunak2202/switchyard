from __future__ import annotations

import json
import logging

import pytest

from switchyard.config import parse_config
from switchyard.context import accept_or_create_request_id, request_id_var
from switchyard.errors import ModelNotFoundError
from switchyard.logs import JsonFormatter
from switchyard.providers import build_providers
from switchyard.router import Router
from switchyard.schemas import ChatCompletionChunk, ChatCompletionRequest

CONFIG = """
providers:
  a: {type: mock, base_url: "http://a/v1"}
  b: {type: mock, base_url: "http://b/v1"}
  disabled: {type: mock, base_url: "http://off/v1", enabled: false}
routes:
  - model: both
    targets: [{provider: a, model: m1}, {provider: disabled, model: x}, {provider: b, model: m2}]
  - model: dead
    targets: [{provider: disabled, model: x}]
"""


@pytest.fixture
def router() -> Router:
    config = parse_config(CONFIG, {})
    return Router(config, build_providers(config))


def test_alias_resolves_in_priority_order_skipping_disabled(router: Router) -> None:
    targets = router.resolve("both")
    assert [(t.provider.name, t.model) for t in targets] == [("a", "m1"), ("b", "m2")]


def test_route_with_no_enabled_provider_is_not_exposed(router: Router) -> None:
    assert router.models() == ["both"]
    with pytest.raises(ModelNotFoundError):
        router.resolve("dead")


def test_provider_pinning_keeps_slashes_in_upstream_model(router: Router) -> None:
    [target] = router.resolve("a/meta-llama/llama-4")
    assert (target.provider.name, target.model) == ("a", "meta-llama/llama-4")


@pytest.mark.parametrize("model", ["nope", "zzz/m", "a/", "disabled/x"])
def test_unknown_models(router: Router, model: str) -> None:
    with pytest.raises(ModelNotFoundError):
        router.resolve(model)


def test_upstream_body_swaps_model_and_preserves_passthrough_fields() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "fast",
            "messages": [{"role": "user", "content": "hi", "x_custom": 1}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "temperature": 0,
        }
    )
    body = req.upstream_body("llama", stream_usage=True)
    assert body["model"] == "llama"
    assert body["tools"] == [{"type": "function", "function": {"name": "f"}}]
    assert body["messages"][0]["x_custom"] == 1
    assert "stream_options" not in body  # only injected for streams
    assert "top_p" not in body  # unset fields are not invented


def test_stream_usage_is_requested_upstream() -> None:
    req = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    assert not req.wants_stream_usage
    assert req.upstream_body("m", stream_usage=True)["stream_options"] == {"include_usage": True}
    assert "stream_options" not in req.upstream_body("m", stream_usage=False)


def test_usage_only_chunk_detection() -> None:
    base = {"id": "c", "created": 1, "model": "m"}
    assert ChatCompletionChunk.model_validate({**base, "choices": [], "usage": {}}).is_usage_only
    content = ChatCompletionChunk.model_validate(
        {**base, "choices": [{"index": 0, "delta": {"content": "x"}}]}
    )
    assert not content.is_usage_only
    assert content.has_content()


@pytest.mark.parametrize(
    ("incoming", "kept"),
    [("abc-123_X.y", True), ("", False), (None, False), ("a\nb", False), ("x" * 129, False)],
)
def test_request_id_acceptance(incoming: str | None, kept: bool) -> None:
    result = accept_or_create_request_id(incoming)
    assert (result == incoming) is kept
    assert len(result) <= 128


def test_json_log_includes_request_id_and_extras() -> None:
    record = logging.LogRecord("switchyard.test", logging.INFO, __file__, 1, "hello", None, None)
    record.provider = "groq"
    token = request_id_var.set("req-1")
    try:
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["msg"] == "hello"
    assert line["request_id"] == "req-1"
    assert line["provider"] == "groq"
    assert line["level"] == "INFO"
