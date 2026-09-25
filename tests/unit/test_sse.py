from __future__ import annotations

from collections.abc import AsyncIterator

from switchyard.api.sse import DONE_EVENT, encode_event
from switchyard.providers.openai_compat import iter_sse_data
from switchyard.schemas import ChatCompletionChunk


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def _collect(*lines: str) -> list[str]:
    return [data async for data in iter_sse_data(_lines(*lines))]


async def test_parses_events_separated_by_blank_lines() -> None:
    assert await _collect("data: a", "", "data: b", "") == ["a", "b"]


async def test_joins_multiline_data_and_skips_comments_and_other_fields() -> None:
    lines = (": keep-alive", "event: message", "id: 1", "data: x", "data: y", "", "")
    assert await _collect(*lines) == ["x\ny"]


async def test_flushes_trailing_event_without_blank_line() -> None:
    assert await _collect("data: [DONE]") == ["[DONE]"]


async def test_data_without_space_after_colon() -> None:
    assert await _collect("data:{}", "") == ["{}"]


def test_encode_round_trips_exactly_what_upstream_sent() -> None:
    raw = (
        '{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",'
        '"choices":[],"x_vendor":{"a":1}}'
    )
    chunk = ChatCompletionChunk.model_validate_json(raw)
    assert encode_event(chunk) == b"data: " + raw.encode() + b"\n\n"
    assert DONE_EVENT == b"data: [DONE]\n\n"
