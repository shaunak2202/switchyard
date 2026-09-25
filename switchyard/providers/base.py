"""The provider adapter interface.

Everything above this layer (routing, retries, circuit breaking, caching, metering) talks to
providers only through ``Provider``. Adapters own three things: the wire format, authentication,
and mapping provider failures onto ``ProviderError``/``FailureKind``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from switchyard.config import ProviderConfig
from switchyard.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest


class Provider(ABC):
    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        """Run a non-streaming completion. Raises ``ProviderError``."""

    @abstractmethod
    def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream completion chunks. Raises ``ProviderError`` before or during iteration.

        Implementations must release the upstream connection when the iterator is closed early
        (``aclose()``), which is how a client disconnect propagates upstream.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Cheap liveness probe that costs no tokens."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release pooled connections."""
