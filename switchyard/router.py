"""Model routing: turn the caller's ``model`` into an ordered list of (provider, model) targets."""

from __future__ import annotations

from dataclasses import dataclass

from switchyard.config import GatewayConfig
from switchyard.errors import ModelNotFoundError
from switchyard.providers import Provider, ProviderRegistry


@dataclass(frozen=True, slots=True)
class Target:
    provider: Provider
    model: str


class Router:
    """Resolves public model names.

    Two forms are accepted:

    * a route alias from config (``"fast"``), which maps to a priority-ordered target list, and
    * ``"<provider>/<upstream-model>"`` to pin one provider, e.g. ``"groq/llama-3.1-8b-instant"``.
      Upstream model names may themselves contain ``/``; only the first segment is checked.
    """

    def __init__(self, config: GatewayConfig, providers: ProviderRegistry) -> None:
        self._providers = providers
        self._routes: dict[str, list[Target]] = {}
        for route in config.routes:
            targets = [
                Target(providers[t.provider], t.model)
                for t in route.targets
                if t.provider in providers
            ]
            if targets:
                self._routes[route.model] = targets

    def resolve(self, model: str) -> list[Target]:
        if model in self._routes:
            return self._routes[model]
        provider_name, sep, upstream_model = model.partition("/")
        if sep and upstream_model and provider_name in self._providers:
            return [Target(self._providers[provider_name], upstream_model)]
        raise ModelNotFoundError(model)

    def models(self) -> list[str]:
        return sorted(self._routes)
