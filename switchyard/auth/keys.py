"""API key issuance and lookup.

Keys look like ``sk-sy-<43 url-safe chars>`` (256 bits of randomness). Only the SHA-256 of a key
is stored. A slow password hash (bcrypt, argon2) is unnecessary here: those exist to protect
low-entropy human passwords from brute force, and brute-forcing a 256-bit random key is already
infeasible. A fast hash also keeps lookup O(1) by hash, with no per-request KDF cost on the hot
path (ADR-011).

Redis layout (``{hash}`` is a Redis Cluster hash tag, so a key's metadata and both of its
rate-limit buckets always live in the same slot and one Lua script can touch all three):

* ``<prefix>:{<hash>}:meta``: hash with name, key_id, rpm, tpm, disabled, created_at
* ``<prefix>:keys``: hash key_id → key hash, for listing and revoking by id
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass

from redis.asyncio import Redis

KEY_PREFIX = "sk-sy-"


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def key_id_for(key_hash: str) -> str:
    """A short, non-secret identifier that is safe to log and to show in dashboards."""
    return f"key_{key_hash[:12]}"


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    key_id: str
    name: str
    rpm: int
    tpm: int
    disabled: bool
    created_at: int


class KeyStore:
    def __init__(self, redis: Redis, prefix: str = "sy") -> None:
        self.redis = redis
        self.prefix = prefix

    def meta_key(self, key_hash: str) -> str:
        return f"{self.prefix}:{{{key_hash}}}:meta"

    def bucket_key(self, key_hash: str, kind: str) -> str:
        return f"{self.prefix}:{{{key_hash}}}:{kind}"

    @property
    def index_key(self) -> str:
        return f"{self.prefix}:keys"

    async def create(self, name: str, *, rpm: int, tpm: int) -> tuple[str, ApiKeyRecord]:
        """Create a key. The plaintext is returned exactly once and never stored."""
        if rpm < 1 or tpm < 1:
            raise ValueError("rpm and tpm must be >= 1")
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)
        record = ApiKeyRecord(
            key_id=key_id_for(key_hash),
            name=name,
            rpm=rpm,
            tpm=tpm,
            disabled=False,
            created_at=int(time.time()),
        )
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                self.meta_key(key_hash),
                mapping={
                    "key_id": record.key_id,
                    "name": name,
                    "rpm": rpm,
                    "tpm": tpm,
                    "disabled": 0,
                    "created_at": record.created_at,
                },
            )
            pipe.hset(self.index_key, record.key_id, key_hash)
            await pipe.execute()
        return api_key, record

    async def get_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        raw = await self.redis.hgetall(self.meta_key(key_hash))
        if not raw:
            return None
        data = {_text(k): _text(v) for k, v in raw.items()}
        return ApiKeyRecord(
            key_id=data["key_id"],
            name=data["name"],
            rpm=int(data["rpm"]),
            tpm=int(data["tpm"]),
            disabled=data["disabled"] == "1",
            created_at=int(data["created_at"]),
        )

    async def list(self) -> list[ApiKeyRecord]:
        hashes = await self.redis.hvals(self.index_key)
        records = [await self.get_by_hash(_text(h)) for h in hashes]
        return sorted((r for r in records if r), key=lambda r: r.created_at)

    async def set_disabled(self, key_id: str, disabled: bool = True) -> bool:
        key_hash = await self.redis.hget(self.index_key, key_id)
        if key_hash is None:
            return False
        await self.redis.hset(self.meta_key(_text(key_hash)), "disabled", int(disabled))
        return True

    async def update_limits(self, key_id: str, *, rpm: int, tpm: int) -> bool:
        key_hash = await self.redis.hget(self.index_key, key_id)
        if key_hash is None:
            return False
        await self.redis.hset(self.meta_key(_text(key_hash)), mapping={"rpm": rpm, "tpm": tpm})
        return True
