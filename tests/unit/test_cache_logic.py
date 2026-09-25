from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from starlette.datastructures import Headers

from switchyard.cache.embeddings import HashingEmbedder
from switchyard.cache.normalize import (
    CacheDirective,
    cacheable,
    exact_key,
    semantic_query,
)
from switchyard.cache.streaming import StreamAssembler, replay_as_chunks
from switchyard.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest

BASE: dict[str, Any] = {
    "model": "fast",
    "temperature": 0,
    "messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
}


def req(**overrides: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(BASE | overrides)


def with_user(text: Any) -> ChatCompletionRequest:
    return req(messages=[BASE["messages"][0], {"role": "user", "content": text}])


# -- exact keys --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(req(stream=True, stream_options={"include_usage": True}), id="stream flag"),
        pytest.param(req(user="alice"), id="user field"),
        pytest.param(with_user("  What is the capital of France?\n"), id="outer whitespace"),
        pytest.param(
            with_user([{"type": "text", "text": "What is the capital of France?"}]),
            id="text part list",
        ),
        pytest.param(
            ChatCompletionRequest.model_validate(dict(reversed(list(BASE.items())))),
            id="key order",
        ),
    ],
)
def test_exact_key_ignores_differences_that_cannot_change_output(
    variant: ChatCompletionRequest,
) -> None:
    assert exact_key(variant) == exact_key(req())


def test_exact_key_normalises_stop_order_and_max_tokens_alias() -> None:
    assert exact_key(req(stop=["b", "a"])) == exact_key(req(stop=["a", "b"]))
    assert exact_key(req(stop="a")) == exact_key(req(stop=["a"]))
    assert exact_key(req(max_tokens=50)) == exact_key(req(max_completion_tokens=50))


@pytest.mark.parametrize(
    "variant",
    [
        req(model="local"),
        req(temperature=0.5),
        req(seed=1),
        req(max_tokens=10),
        req(tools=[{"type": "function", "function": {"name": "f"}}]),
        with_user("What is the capital of Spain?"),
        with_user("What is the  capital of France?"),  # internal whitespace is kept
        req(messages=[{"role": "user", "content": "What is the capital of France?"}]),
    ],
)
def test_exact_key_distinguishes_meaningful_changes(variant: ChatCompletionRequest) -> None:
    assert exact_key(variant) != exact_key(req())


# -- policy ------------------------------------------------------------------------------------


def test_only_deterministic_requests_are_cached_unless_opted_in() -> None:
    default = CacheDirective()
    assert cacheable(req(temperature=0), default)
    assert not cacheable(req(temperature=0.7), default)
    unset = ChatCompletionRequest.model_validate(
        {k: v for k, v in BASE.items() if k != "temperature"}
    )
    assert not cacheable(unset, default)  # provider default is a sample, not deterministic
    assert cacheable(req(temperature=0.7), CacheDirective(opt_in=True))


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, CacheDirective()),
        ({"cache-control": "no-cache"}, CacheDirective(read=False)),
        ({"cache-control": "no-store"}, CacheDirective(write=False)),
        ({"cache-control": "No-Cache, no-store"}, CacheDirective(read=False, write=False)),
        ({"x-switchyard-cache": "allow"}, CacheDirective(opt_in=True)),
        ({"x-switchyard-cache": "allow, no-semantic"}, CacheDirective(opt_in=True, semantic=False)),
    ],
)
def test_directive_from_headers(headers: dict[str, str], expected: CacheDirective) -> None:
    assert CacheDirective.from_headers(Headers(headers)) == expected


# -- semantic candidates -----------------------------------------------------------------------


def test_single_turn_prompt_is_a_semantic_candidate() -> None:
    query = semantic_query(req(), max_chars=2000)
    assert query is not None
    assert query.text == "What is the capital of France?"


def test_namespace_separates_models_params_and_system_prompts() -> None:
    def ns(r: ChatCompletionRequest) -> str | None:
        q = semantic_query(r, 2000)
        return q.namespace if q else None

    base = ns(req())
    assert ns(with_user("Something else entirely")) == base  # same namespace, different text
    assert ns(req(model="local")) != base
    assert ns(req(max_tokens=5)) != base
    other_system = [{"role": "system", "content": "You are verbose."}, BASE["messages"][1]]
    assert ns(req(messages=other_system)) != base


@pytest.mark.parametrize(
    "request_",
    [
        req(
            messages=[
                *BASE["messages"],
                {"role": "assistant", "content": "Paris"},
                {"role": "user", "content": "And Spain?"},
            ]
        ),
        req(tools=[{"type": "function", "function": {"name": "f"}}]),
        req(n=2),
        req(response_format={"type": "json_object"}),
        with_user("x" * 2001),
        with_user([{"type": "image_url", "image_url": {"url": "http://x"}}]),
    ],
    ids=["multi-turn", "tools", "n>1", "response_format", "too long", "multimodal"],
)
def test_unsafe_requests_are_not_semantic_candidates(request_: ChatCompletionRequest) -> None:
    assert semantic_query(request_, max_chars=2000) is None


async def test_hashing_embedder_behaves_like_an_embedding() -> None:
    emb = HashingEmbedder()
    a = await emb.embed("What is the capital of France?")
    b = await emb.embed("what is the capital of france")
    c = await emb.embed("How do I sort a list in Python?")
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert float(a @ b) > 0.99
    assert float(a @ c) < 0.5


# -- stream <-> completion ---------------------------------------------------------------------


def _chunk(delta: dict[str, Any], finish: str | None = None, **kw: Any) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c1",
        created=5,
        model="m",
        choices=[{"index": kw.pop("index", 0), "delta": delta, "finish_reason": finish}],
        **kw,
    )


def test_assembler_rebuilds_completion() -> None:
    asm = StreamAssembler()
    asm.add(_chunk({"role": "assistant"}))
    asm.add(_chunk({"content": "Hello"}))
    asm.add(_chunk({"content": " world"}))
    asm.add(_chunk({}, finish="stop"))
    completion = asm.build()
    assert completion is not None
    assert completion.choices[0]["message"] == {"role": "assistant", "content": "Hello world"}
    assert completion.choices[0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "bad",
    [
        _chunk({"tool_calls": [{"index": 0, "function": {"name": "f"}}]}, finish="tool_calls"),
        _chunk({"content": "x"}, index=1, finish="stop"),
    ],
    ids=["tool calls", "second choice"],
)
def test_assembler_refuses_streams_it_cannot_represent(bad: ChatCompletionChunk) -> None:
    asm = StreamAssembler()
    asm.add(_chunk({"content": "a"}))
    asm.add(bad)
    assert asm.build() is None


def test_assembler_requires_a_finish_reason() -> None:
    asm = StreamAssembler()
    asm.add(_chunk({"content": "partial"}))
    assert asm.build() is None


def test_replay_round_trips_through_assembler() -> None:
    original = ChatCompletion(
        id="c",
        created=1,
        model="m",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The quick  brown fox\njumps."},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    )
    chunks = list(replay_as_chunks(original, include_usage=True))
    assert chunks[-1].is_usage_only
    asm = StreamAssembler()
    for chunk in chunks:
        asm.add(chunk)
    rebuilt = asm.build()
    assert rebuilt is not None
    assert rebuilt.choices[0]["message"] == original.choices[0]["message"]
    assert len(chunks) > 3  # content is split, not sent as one blob
