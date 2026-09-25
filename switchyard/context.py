"""Per-request context shared between middleware, handlers and log records."""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")


def new_request_id() -> str:
    return uuid.uuid4().hex


def accept_or_create_request_id(incoming: str | None) -> str:
    """Propagate a caller-supplied request id if it is sane, otherwise mint a new one.

    Accepting arbitrary header values would let callers inject newlines or huge strings into
    our logs, so anything outside a conservative charset/length is replaced.
    """
    if incoming and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return new_request_id()
