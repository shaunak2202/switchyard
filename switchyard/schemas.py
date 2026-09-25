"""OpenAI chat-completions wire types.

Models are deliberately permissive (``extra="allow"``): the gateway validates what it needs to
route, cache and meter, and passes everything else (tools, response_format, logprobs, ...)
through untouched. Serialising with ``exclude_unset=True`` reproduces exactly what was received.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatMessage(_Open):
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class StreamOptions(_Open):
    include_usage: bool = False


class ChatCompletionRequest(_Open):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    n: int | None = Field(default=None, ge=1, le=8)
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    user: str | None = None

    @property
    def wants_stream_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)

    def upstream_body(self, model: str, *, stream_usage: bool) -> dict[str, Any]:
        """The JSON body to send upstream: the caller's request with the target model swapped in."""
        body = self.model_dump(mode="json", exclude_unset=True)
        body["model"] = model
        body["stream"] = self.stream
        if self.stream and stream_usage:
            # Always ask for usage so token-based rate limits can be reconciled; the extra
            # usage-only chunk is stripped again if the caller did not ask for it.
            body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
        return body


class Usage(_Open):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(_Open):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: Usage | None = None


class ChatCompletionChunk(_Open):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage | None = None

    @property
    def is_usage_only(self) -> bool:
        return not self.choices and self.usage is not None

    def has_content(self) -> bool:
        return any((choice.get("delta") or {}).get("content") for choice in self.choices)
