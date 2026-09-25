"""Bridging cached completions and streams.

A cache entry is always stored in non-streaming shape (``ChatCompletion``), so one entry serves
both kinds of caller:

* a *streaming* miss is assembled into a completion as it is relayed, then stored;
* a *streaming* hit is replayed as a synthetic chunk sequence.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from switchyard.schemas import ChatCompletion, ChatCompletionChunk, Usage

_PIECES = re.compile(r"\S+\s*|\s+")


class StreamAssembler:
    """Accumulates a single-choice text stream into a ``ChatCompletion``.

    Streams it cannot represent faithfully (several choices, tool-call deltas) mark themselves
    ``cacheable = False`` instead of producing a lossy entry.
    """

    def __init__(self) -> None:
        self.cacheable = True
        self._id: str | None = None
        self._created = 0
        self._model = ""
        self._role = "assistant"
        self._parts: list[str] = []
        self._finish: str | None = None
        self._usage: Usage | None = None

    def add(self, chunk: ChatCompletionChunk) -> None:
        if self._id is None:
            self._id, self._created, self._model = chunk.id, chunk.created, chunk.model
        if chunk.usage is not None:
            self._usage = chunk.usage
        for choice in chunk.choices:
            delta = choice.get("delta") or {}
            if choice.get("index", 0) != 0 or delta.get("tool_calls") or delta.get("function_call"):
                self.cacheable = False
            if delta.get("role"):
                self._role = delta["role"]
            if delta.get("content"):
                self._parts.append(delta["content"])
            if choice.get("finish_reason"):
                self._finish = choice["finish_reason"]

    def build(self) -> ChatCompletion | None:
        if not self.cacheable or self._id is None or self._finish is None:
            return None
        return ChatCompletion(
            id=self._id,
            created=self._created,
            model=self._model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": self._role, "content": "".join(self._parts)},
                    "finish_reason": self._finish,
                }
            ],
            usage=self._usage,
        )


def replay_as_chunks(
    completion: ChatCompletion, *, include_usage: bool, words_per_chunk: int = 4
) -> Iterator[ChatCompletionChunk]:
    """Turn a cached completion back into a plausible OpenAI chunk sequence."""

    def chunk(choices: list[dict[str, object]], usage: Usage | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=completion.id,
            created=completion.created,
            model=completion.model,
            choices=choices,
            usage=usage,
        )

    for choice in completion.choices:
        index = choice.get("index", 0)
        message = choice.get("message") or {}
        yield chunk([{"index": index, "delta": {"role": message.get("role", "assistant")}}])
        pieces = _PIECES.findall(str(message.get("content") or ""))
        for start in range(0, len(pieces), words_per_chunk):
            text = "".join(pieces[start : start + words_per_chunk])
            yield chunk([{"index": index, "delta": {"content": text}, "finish_reason": None}])
        yield chunk([{"index": index, "delta": {}, "finish_reason": choice.get("finish_reason")}])
    if include_usage:
        yield chunk([], usage=completion.usage or Usage())
