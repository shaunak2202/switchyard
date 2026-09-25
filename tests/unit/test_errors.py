from __future__ import annotations

import pytest

from switchyard.errors import (
    FailureKind,
    InvalidRequestError,
    ProviderError,
    ServiceUnavailableError,
    UpstreamError,
    classify_status,
    gateway_error_from_provider,
)


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, FailureKind.BAD_REQUEST),
        (422, FailureKind.BAD_REQUEST),
        (401, FailureKind.AUTH),
        (403, FailureKind.AUTH),
        (404, FailureKind.NOT_FOUND),
        (408, FailureKind.TIMEOUT),
        (429, FailureKind.RATE_LIMITED),
        (500, FailureKind.UPSTREAM_5XX),
        (503, FailureKind.UPSTREAM_5XX),
    ],
)
def test_classify_status(status: int, kind: FailureKind) -> None:
    assert classify_status(status) is kind


def test_retry_and_failover_policy() -> None:
    transient = {
        FailureKind.TIMEOUT,
        FailureKind.CONNECT,
        FailureKind.CONNECTION_DROPPED,
        FailureKind.UPSTREAM_5XX,
        FailureKind.RATE_LIMITED,
    }
    for kind in FailureKind:
        assert kind.retryable is (kind in transient), kind
    # A bad key or unknown model on one provider says nothing about the next one...
    assert FailureKind.AUTH.failover and FailureKind.NOT_FOUND.failover
    # ...but a malformed request will be malformed everywhere.
    assert not FailureKind.BAD_REQUEST.failover


def test_bad_request_keeps_upstream_status_and_message() -> None:
    err = gateway_error_from_provider(
        ProviderError("p", FailureKind.BAD_REQUEST, "max_tokens too large", status=422)
    )
    assert isinstance(err, InvalidRequestError)
    assert err.status_code == 422
    assert err.to_body()["error"]["message"] == "max_tokens too large"


def test_timeout_maps_to_504_and_open_circuit_to_503() -> None:
    timeout = gateway_error_from_provider(ProviderError("p", FailureKind.TIMEOUT, "slow"))
    assert isinstance(timeout, UpstreamError)
    assert timeout.status_code == 504
    open_ = gateway_error_from_provider(ProviderError("p", FailureKind.CIRCUIT_OPEN, "open"))
    assert isinstance(open_, ServiceUnavailableError)
    assert open_.status_code == 503


def test_other_failures_are_502_without_leaking_upstream_detail() -> None:
    err = gateway_error_from_provider(
        ProviderError("groq", FailureKind.AUTH, "Invalid API Key gsk_abc", status=401)
    )
    assert err.status_code == 502
    assert "gsk_abc" not in err.message
