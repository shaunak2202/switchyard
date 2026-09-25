"""Request admission: API key authentication and rate limiting, plus post-response settlement."""

from __future__ import annotations

import logging

import anyio
from fastapi import Request
from redis.exceptions import RedisError

from switchyard.auth.keys import KeyStore, hash_api_key
from switchyard.errors import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
)
from switchyard.ratelimit.limiter import AdmitStatus, Decision, RateLimiter
from switchyard.ratelimit.tokens import estimate_request_tokens
from switchyard.schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


class Guard:
    def __init__(
        self,
        limiter: RateLimiter,
        keys: KeyStore,
        *,
        enabled: bool,
        default_max_tokens: int,
    ) -> None:
        self.limiter = limiter
        self.keys = keys
        self.enabled = enabled
        self.default_max_tokens = default_max_tokens

    async def admit(self, request: Request, body: ChatCompletionRequest) -> Decision | None:
        """Authenticate and charge rate limits. ``None`` when auth is disabled."""
        if not self.enabled:
            return None
        api_key = bearer_token(request)
        if api_key is None:
            raise AuthenticationError("Missing API key. Send 'Authorization: Bearer <key>'.")
        cost = estimate_request_tokens(body, self.default_max_tokens)
        try:
            decision = await self.limiter.admit(api_key, cost)
        except RedisError as exc:
            # Fail closed: without Redis we cannot tell a valid key from an invalid one.
            logger.error("rate limiter unavailable", extra={"error": repr(exc)})
            raise ServiceUnavailableError(
                "authentication backend unavailable", code="auth_unavailable"
            ) from exc

        request.state.key_id = decision.key_id
        match decision.status:
            case AdmitStatus.ADMITTED:
                return decision
            case AdmitStatus.LIMITED:
                raise RateLimitError(
                    f"Rate limit exceeded for {decision.key_id}. "
                    f"Retry after {decision.retry_after_s}s.",
                    code="rate_limit_exceeded",
                    headers=decision.headers(),
                )
            case AdmitStatus.TOO_LARGE:
                raise InvalidRequestError(
                    f"This request needs ~{cost} tokens, more than the key's limit of "
                    f"{decision.limit_tokens} tokens per minute. Lower max_tokens.",
                    code="tokens_exceed_limit",
                    param="max_tokens",
                )
            case AdmitStatus.DISABLED:
                raise AuthenticationError("This API key has been disabled.")
            case _:
                raise AuthenticationError()

    async def authenticate(self, request: Request) -> None:
        """Key check without charging a rate limit (for cheap metadata endpoints)."""
        if not self.enabled:
            return
        api_key = bearer_token(request)
        if api_key is None:
            raise AuthenticationError("Missing API key. Send 'Authorization: Bearer <key>'.")
        try:
            record = await self.keys.get_by_hash(hash_api_key(api_key))
        except RedisError as exc:
            raise ServiceUnavailableError(
                "authentication backend unavailable", code="auth_unavailable"
            ) from exc
        if record is None or record.disabled:
            raise AuthenticationError()

    async def settle(self, decision: Decision | None, actual_tokens: int) -> None:
        """Reconcile the pre-charged token estimate with real usage. Never raises.

        Shielded from cancellation: this runs from ``finally`` blocks when a client disconnects
        mid-stream, and a lost settlement would leave the key over- or under-charged.
        """
        if decision is None:
            return
        with anyio.CancelScope(shield=True):
            try:
                await self.limiter.reconcile(decision, actual_tokens)
            except RedisError as exc:
                logger.warning(
                    "token reconciliation failed",
                    extra={"key_id": decision.key_id, "error": repr(exc)},
                )
