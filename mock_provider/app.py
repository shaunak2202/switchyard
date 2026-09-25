"""Mock LLM provider.

Simulates the behaviours a gateway has to survive: time-to-first-token, per-token pacing,
HTTP errors, hangs (timeouts), connections dropped mid-stream and streams that stall. Every
knob is set from ``MOCK_*`` env vars at startup and can be changed at runtime through
``PATCH /admin/config``, which is how chaos load tests flip a healthy provider into a failing
one mid-run.

Output text is a deterministic function of the prompt, so caching behaviour is observable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_VOCAB = [
    "the",
    "gateway",
    "routes",
    "each",
    "request",
    "to",
    "a",
    "healthy",
    "provider",
    "while",
    "caching",
    "repeated",
    "prompts",
    "and",
    "shedding",
    "load",
    "fairly",
    "across",
    "keys",
    "so",
    "latency",
    "stays",
    "predictable",
    "under",
    "pressure",
    "even",
    "when",
    "an",
    "upstream",
    "model",
    "slows",
    "down",
    "or",
    "fails",
]


class MockSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOCK_", validate_assignment=True)

    ttft_ms: float = Field(default=50.0, ge=0, description="delay before the first token")
    ttft_jitter_ms: float = Field(default=0.0, ge=0, description="uniform +/- jitter on ttft")
    inter_token_ms: float = Field(default=5.0, ge=0, description="delay between tokens")
    output_tokens: int = Field(default=32, ge=1, le=4096)
    error_rate: float = Field(default=0.0, ge=0, le=1, description="fraction answered with error")
    error_status: int = Field(default=500, ge=400, le=599)
    timeout_rate: float = Field(default=0.0, ge=0, le=1, description="fraction that hang")
    hang_s: float = Field(default=600.0, ge=0)
    stream_abort_rate: float = Field(default=0.0, ge=0, le=1, description="drop conn mid-stream")
    stream_stall_rate: float = Field(default=0.0, ge=0, le=1, description="stop sending mid-stream")
    fail_after_tokens: int = Field(default=5, ge=0)
    seed: int | None = None


class ConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttft_ms: float | None = None
    ttft_jitter_ms: float | None = None
    inter_token_ms: float | None = None
    output_tokens: int | None = None
    error_rate: float | None = None
    error_status: int | None = None
    timeout_rate: float | None = None
    hang_s: float | None = None
    stream_abort_rate: float | None = None
    stream_stall_rate: float | None = None
    fail_after_tokens: int | None = None


@dataclass
class Stats:
    requests: int = 0
    streams: int = 0
    in_flight: int = 0
    injected_errors: int = 0
    injected_timeouts: int = 0
    injected_aborts: int = 0
    injected_stalls: int = 0


class MockStreamAborted(Exception):
    """Raised inside a streaming body to make the server drop the connection."""


def _prompt_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return "\n".join(parts)


def _tokens_for(prompt: str, count: int) -> list[str]:
    digest = hashlib.sha256(prompt.encode()).digest()
    rng = random.Random(digest)
    return [("" if i == 0 else " ") + rng.choice(_VOCAB) for i in range(count)]


def create_app(settings: MockSettings | None = None) -> FastAPI:
    defaults = settings or MockSettings()
    app = FastAPI(title="Switchyard mock provider")
    app.state.settings = defaults.model_copy()
    app.state.stats = Stats()
    app.state.rng = random.Random(defaults.seed)

    def cfg() -> MockSettings:
        current: MockSettings = app.state.settings
        return current

    def roll(rate: float) -> bool:
        return rate > 0 and bool(app.state.rng.random() < rate)

    def ttft_s() -> float:
        s = cfg()
        jitter = (
            app.state.rng.uniform(-s.ttft_jitter_ms, s.ttft_jitter_ms) if s.ttft_jitter_ms else 0
        )
        return max(s.ttft_ms + jitter, 0.0) / 1000

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": "mock-1", "object": "model", "created": 0, "owned_by": "mock"}],
        }

    @app.get("/admin/config")
    async def get_config() -> dict[str, Any]:
        return cfg().model_dump()

    @app.patch("/admin/config")
    async def patch_config(patch: ConfigPatch) -> dict[str, Any]:
        updated = cfg().model_dump() | patch.model_dump(exclude_none=True)
        app.state.settings = MockSettings.model_validate(updated)
        return cfg().model_dump()

    @app.post("/admin/reset")
    async def reset() -> dict[str, Any]:
        app.state.settings = defaults.model_copy()
        app.state.stats = Stats()
        return cfg().model_dump()

    @app.get("/admin/stats")
    async def stats() -> dict[str, int]:
        return asdict(app.state.stats)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> Response:
        body: dict[str, Any] = await request.json()
        s = cfg()
        stats: Stats = app.state.stats
        stats.requests += 1

        if roll(s.timeout_rate):
            stats.injected_timeouts += 1
            await asyncio.sleep(s.hang_s)
        if roll(s.error_rate):
            stats.injected_errors += 1
            return JSONResponse(
                {"error": {"message": "injected failure", "type": "mock_error"}},
                status_code=s.error_status,
            )

        prompt = _prompt_text(body)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        count = min(s.output_tokens, int(max_tokens)) if max_tokens else s.output_tokens
        tokens = _tokens_for(prompt, count)
        usage = {
            "prompt_tokens": max(len(prompt) // 4, 1),
            "completion_tokens": count,
            "total_tokens": max(len(prompt) // 4, 1) + count,
        }
        completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        model = str(body.get("model", "mock-1"))

        if not body.get("stream"):
            stats.in_flight += 1
            try:
                await asyncio.sleep(ttft_s() + count * s.inter_token_ms / 1000)
            finally:
                stats.in_flight -= 1
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "".join(tokens)},
                            "finish_reason": "stop" if count == s.output_tokens else "length",
                        }
                    ],
                    "usage": usage,
                }
            )

        stats.streams += 1
        abort = roll(s.stream_abort_rate)
        stall = not abort and roll(s.stream_stall_rate)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def event(choices: list[dict[str, Any]], **extra: Any) -> bytes:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": choices,
                **extra,
            }
            return b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n"

        async def generate() -> AsyncIterator[bytes]:
            stats.in_flight += 1
            try:
                await asyncio.sleep(ttft_s())
                yield event([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
                for i, token in enumerate(tokens):
                    if i == s.fail_after_tokens and abort:
                        stats.injected_aborts += 1
                        raise MockStreamAborted("injected mid-stream abort")
                    if i == s.fail_after_tokens and stall:
                        stats.injected_stalls += 1
                        await asyncio.sleep(s.hang_s)
                    if i:
                        await asyncio.sleep(s.inter_token_ms / 1000)
                    yield event([{"index": 0, "delta": {"content": token}, "finish_reason": None}])
                finish = "stop" if count == s.output_tokens else "length"
                yield event([{"index": 0, "delta": {}, "finish_reason": finish}])
                if include_usage:
                    yield event([], usage=usage)
                yield b"data: [DONE]\n\n"
            finally:
                stats.in_flight -= 1

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app
