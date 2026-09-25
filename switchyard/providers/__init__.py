"""Provider adapters and the registry that builds them from config."""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx

from switchyard.config import GatewayConfig, ProviderConfig
from switchyard.providers.base import Provider
from switchyard.providers.groq import GroqProvider
from switchyard.providers.mock import MockProvider
from switchyard.providers.ollama import OllamaProvider
from switchyard.providers.openai_compat import OpenAICompatibleProvider

__all__ = ["Provider", "ProviderRegistry", "build_providers"]

logger = logging.getLogger(__name__)

_ADAPTERS: dict[str, type[OpenAICompatibleProvider]] = {
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
    "openai": OpenAICompatibleProvider,
}

TransportFactory = Callable[[str, ProviderConfig], httpx.AsyncBaseTransport | None]

ProviderRegistry = dict[str, Provider]


def build_providers(
    config: GatewayConfig, transport_factory: TransportFactory | None = None
) -> ProviderRegistry:
    """Instantiate every enabled provider.

    A provider whose type requires a key but has none (e.g. ``GROQ_API_KEY`` unset) is skipped
    with a warning instead of failing startup, so the stack runs out of the box on mock/Ollama.
    """
    providers: ProviderRegistry = {}
    for name, pcfg in config.providers.items():
        if not pcfg.enabled:
            logger.info("provider disabled by config", extra={"provider": name})
            continue
        if pcfg.type == "groq" and pcfg.api_key is None:
            logger.warning("provider skipped: no API key configured", extra={"provider": name})
            continue
        transport = transport_factory(name, pcfg) if transport_factory else None
        providers[name] = _ADAPTERS[pcfg.type](name, pcfg, transport=transport)
    return providers
