"""Groq: OpenAI-compatible, authenticated with a bearer key."""

from __future__ import annotations

from switchyard.providers.openai_compat import OpenAICompatibleProvider
from switchyard.schemas import ChatCompletionChunk, Usage


class GroqProvider(OpenAICompatibleProvider):
    def normalize_chunk(self, chunk: ChatCompletionChunk) -> ChatCompletionChunk:
        # Groq reports streaming usage in a vendor extension on the final chunk
        # (``x_groq.usage``) rather than, or as well as, the standard ``usage`` field.
        if chunk.usage is None:
            extra = (chunk.model_extra or {}).get("x_groq")
            if isinstance(extra, dict) and isinstance(extra.get("usage"), dict):
                chunk.usage = Usage.model_validate(extra["usage"])
        return chunk
