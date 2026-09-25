"""The mock provider (see ``mock_provider/``) speaks the OpenAI format; health is ``/health``."""

from __future__ import annotations

import httpx

from switchyard.providers.openai_compat import OpenAICompatibleProvider


class MockProvider(OpenAICompatibleProvider):
    async def health(self) -> bool:
        root = self.config.base_url.removesuffix("/v1")
        try:
            response = await self._client.get(f"{root}/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200
