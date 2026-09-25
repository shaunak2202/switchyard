"""Ollama via its OpenAI-compatible ``/v1`` API. Health uses the native API, which is cheaper."""

from __future__ import annotations

import httpx

from switchyard.providers.openai_compat import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    async def health(self) -> bool:
        native_root = self.config.base_url.removesuffix("/v1")
        try:
            response = await self._client.get(f"{native_root}/api/tags", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
