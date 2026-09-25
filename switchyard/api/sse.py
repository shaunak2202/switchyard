"""Server-sent event encoding for OpenAI-style streams."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

DONE_EVENT = b"data: [DONE]\n\n"


def encode_event(payload: BaseModel | dict[str, Any]) -> bytes:
    if isinstance(payload, BaseModel):
        body = payload.model_dump_json(exclude_unset=True)
    else:
        body = json.dumps(payload, separators=(",", ":"))
    return b"data: " + body.encode() + b"\n\n"
