"""Authentication plus requests-per-minute and tokens-per-minute limits, in one Redis round trip.

Both checks run inside one Lua script (``admit.lua``), so the hot path costs a single
``EVALSHA`` and the check-and-decrement is atomic across every gateway process and replica.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from importlib import resources

from redis.asyncio import Redis

from switchyard.auth.keys import KeyStore, hash_api_key, key_id_for


def _script(name: str) -> str:
    return resources.files("switchyard.ratelimit").joinpath(name).read_text()


class AdmitStatus(IntEnum):
    ADMITTED = 1
    LIMITED = 0
    UNKNOWN_KEY = -1
    DISABLED = -2
    TOO_LARGE = -3


@dataclass(frozen=True, slots=True)
class Decision:
    status: AdmitStatus
    key_hash: str
    retry_after_ms: int
    limit_requests: int
    remaining_requests: int
    limit_tokens: int
    remaining_tokens: int
    token_cost: int

    @property
    def admitted(self) -> bool:
        return self.status is AdmitStatus.ADMITTED

    @property
    def key_id(self) -> str:
        return key_id_for(self.key_hash)

    @property
    def retry_after_s(self) -> int:
        """Whole seconds for the ``Retry-After`` header (rounded up, never 0)."""
        return max(1, math.ceil(self.retry_after_ms / 1000))

    def headers(self) -> dict[str, str]:
        """OpenAI-style rate-limit headers."""
        if self.status not in (AdmitStatus.ADMITTED, AdmitStatus.LIMITED):
            return {}
        headers = {
            "x-ratelimit-limit-requests": str(self.limit_requests),
            "x-ratelimit-remaining-requests": str(max(self.remaining_requests, 0)),
            "x-ratelimit-limit-tokens": str(self.limit_tokens),
            "x-ratelimit-remaining-tokens": str(max(self.remaining_tokens, 0)),
        }
        if self.status is AdmitStatus.LIMITED:
            headers["retry-after"] = str(self.retry_after_s)
            headers["retry-after-ms"] = str(self.retry_after_ms)
        return headers


class RateLimiter:
    def __init__(self, redis: Redis, keys: KeyStore) -> None:
        self.redis = redis
        self.keys = keys
        self._admit = redis.register_script(_script("admit.lua"))
        self._reconcile = redis.register_script(_script("reconcile.lua"))

    async def admit(self, api_key: str, token_cost: int) -> Decision:
        key_hash = hash_api_key(api_key)
        cost = max(int(token_cost), 1)
        raw = await self._admit(
            keys=[
                self.keys.meta_key(key_hash),
                self.keys.bucket_key(key_hash, "rpm"),
                self.keys.bucket_key(key_hash, "tpm"),
            ],
            args=[cost],
        )
        status, retry_ms, rpm, rpm_left, tpm, tpm_left = (int(v) for v in raw)
        return Decision(
            status=AdmitStatus(status),
            key_hash=key_hash,
            retry_after_ms=retry_ms,
            limit_requests=rpm,
            remaining_requests=rpm_left,
            limit_tokens=tpm,
            remaining_tokens=tpm_left,
            token_cost=cost,
        )

    async def reconcile(self, decision: Decision, actual_tokens: int) -> int:
        """Refund (or charge) the difference between the estimate and real usage."""
        delta = decision.token_cost - actual_tokens
        if delta == 0:
            return decision.remaining_tokens
        result = await self._reconcile(
            keys=[self.keys.bucket_key(decision.key_hash, "tpm")],
            args=[delta, decision.limit_tokens],
        )
        return int(result)
