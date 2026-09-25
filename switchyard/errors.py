"""Error taxonomy.

Two families:

* ``ProviderError`` is raised by provider adapters. It carries a ``FailureKind`` that the
  reliability layer uses to decide whether to retry the same provider and/or fail over to the
  next one. Adapters are the only place that knows provider-specific error shapes.
* ``GatewayError`` is what the HTTP layer turns into an OpenAI-shaped error response.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class FailureKind(StrEnum):
    TIMEOUT = "timeout"
    CONNECT = "connect"
    CONNECTION_DROPPED = "connection_dropped"
    UPSTREAM_5XX = "upstream_5xx"
    RATE_LIMITED = "rate_limited"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    PROTOCOL = "protocol"
    CIRCUIT_OPEN = "circuit_open"

    @property
    def retryable(self) -> bool:
        """Worth retrying against the *same* provider (transient)."""
        return self in _RETRYABLE

    @property
    def failover(self) -> bool:
        """Worth trying the *next* provider. Only a malformed client request is final."""
        return self is not FailureKind.BAD_REQUEST

    @property
    def counts_against_provider(self) -> bool:
        """Evidence that the provider is unhealthy, for the circuit breaker. A bad request or a
        model the provider does not serve says nothing about its health."""
        return self not in (
            FailureKind.BAD_REQUEST,
            FailureKind.NOT_FOUND,
            FailureKind.CIRCUIT_OPEN,
        )


_RETRYABLE = frozenset(
    {
        FailureKind.TIMEOUT,
        FailureKind.CONNECT,
        FailureKind.CONNECTION_DROPPED,
        FailureKind.UPSTREAM_5XX,
        FailureKind.RATE_LIMITED,
    }
)


def classify_status(status: int) -> FailureKind:
    if status == 429:
        return FailureKind.RATE_LIMITED
    if status in (401, 403):
        return FailureKind.AUTH
    if status == 404:
        return FailureKind.NOT_FOUND
    if status == 408:
        return FailureKind.TIMEOUT
    if status >= 500:
        return FailureKind.UPSTREAM_5XX
    return FailureKind.BAD_REQUEST


class ProviderError(Exception):
    def __init__(
        self,
        provider: str,
        kind: FailureKind,
        message: str,
        *,
        status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(f"{provider}: {kind.value}: {message}")
        self.provider = provider
        self.kind = kind
        self.message = message
        self.status = status
        self.retry_after_s = retry_after_s


class GatewayError(Exception):
    status_code: int = 500
    error_type: str = "api_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param
        self.headers = headers or {}
        if status_code is not None:
            self.status_code = status_code

    def to_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class InvalidRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"


class ModelNotFoundError(GatewayError):
    status_code = 404
    error_type = "invalid_request_error"

    def __init__(self, model: str) -> None:
        super().__init__(
            f"The model {model!r} does not exist or is not routable by this gateway.",
            code="model_not_found",
            param="model",
        )


class UpstreamError(GatewayError):
    """Every eligible provider failed. 502 unless we know better (timeout → 504)."""

    status_code = 502
    error_type = "upstream_error"


class ServiceUnavailableError(GatewayError):
    status_code = 503
    error_type = "service_unavailable"


def gateway_error_from_provider(err: ProviderError) -> GatewayError:
    """Translate the *last* provider failure into what the caller sees."""
    if err.kind is FailureKind.BAD_REQUEST:
        # The request itself is bad; surface the upstream message with the upstream status.
        return InvalidRequestError(err.message, code="upstream_bad_request", status_code=err.status)
    if err.kind is FailureKind.TIMEOUT:
        return UpstreamError(
            f"upstream {err.provider} timed out", code="upstream_timeout", status_code=504
        )
    if err.kind is FailureKind.CIRCUIT_OPEN:
        return ServiceUnavailableError(
            "no healthy provider is available for this model", code="all_circuits_open"
        )
    return UpstreamError(f"upstream {err.provider} failed: {err.kind.value}", code=err.kind.value)
