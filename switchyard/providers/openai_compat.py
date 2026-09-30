"""Adapter for any upstream that speaks the OpenAI chat-completions wire format.

Groq, Ollama (``/v1``) and the mock provider all do, so they share this implementation and
subclass it only where they genuinely differ (auth, health probes, usage reporting).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx
from pydantic import ValidationError

from switchyard.config import ProviderConfig
from switchyard.errors import FailureKind, ProviderError, classify_status
from switchyard.providers.base import Provider
from switchyard.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest

DONE_SENTINEL = "[DONE]"


async def iter_sse_data(lines: AsyncIterator[str]) -> AsyncGenerator[str, None]:
    """Yield the ``data`` payload of each server-sent event.

    Implements the parts of the SSE spec that matter here: events are separated by a blank line,
    multi-line ``data:`` fields are joined with ``\\n``, and ``:`` comment lines (keep-alives)
    are ignored.
    """
    data_lines: list[str] = []
    async for line in lines:
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data_lines.append(value.removeprefix(" "))
    if data_lines:
        yield "\n".join(data_lines)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500] or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return f"HTTP {response.status_code}"


class _Shard:
    """One ``httpx.AsyncClient`` plus a semaphore sized to its connection pool.

    Why shards (ADR-019): httpcore's pool re-scans every connection and every queued request on
    each request start and finish, so its CPU cost grows with pool size times queue length.
    Profiling under load put ~45% of gateway CPU there and throughput collapsed past
    saturation. Several small pools keep each scan short, and the semaphore keeps httpcore's own
    queue empty: excess requests wait on an O(1) semaphore instead of a list that is scanned.
    """

    __slots__ = ("client", "in_flight", "slots")

    def __init__(self, client: httpx.AsyncClient, size: int) -> None:
        self.client = client
        self.slots = asyncio.Semaphore(size)
        self.in_flight = 0


class OpenAICompatibleProvider(Provider):
    #: Whether the upstream honours ``stream_options.include_usage``.
    supports_stream_usage: bool = True

    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, config)
        t = config.timeouts
        headers = {"content-type": "application/json"}
        if config.api_key is not None:
            headers["authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        per_shard = max(config.max_connections // config.pool_shards, 1)
        self._shards = [
            _Shard(
                httpx.AsyncClient(
                    base_url=config.base_url,
                    headers=headers,
                    # httpx's read timeout is only a backstop; precise first-byte/idle/total
                    # deadlines are enforced with asyncio timeouts so they compose with retries.
                    timeout=httpx.Timeout(
                        connect=t.connect_s,
                        read=max(t.first_byte_s, t.idle_s, t.total_s),
                        write=t.connect_s,
                        # The semaphore guarantees a free connection, so the pool never waits.
                        pool=t.connect_s,
                    ),
                    limits=httpx.Limits(
                        max_connections=per_shard,
                        max_keepalive_connections=per_shard,
                        keepalive_expiry=config.keepalive_expiry_s,
                    ),
                    transport=transport,
                ),
                per_shard,
            )
            for _ in range(config.pool_shards)
        ]
        self._client = self._shards[0].client  # for cheap calls like health probes

    def _pick_shard(self) -> _Shard:
        # Least loaded; N is small (single digits), so a linear scan is cheaper than a heap.
        return min(self._shards, key=lambda shard: shard.in_flight)

    # -- hooks -----------------------------------------------------------------------------

    def normalize_chunk(self, chunk: ChatCompletionChunk) -> ChatCompletionChunk:
        return chunk

    # -- error mapping ---------------------------------------------------------------------

    def _from_http_error(self, exc: httpx.HTTPError) -> ProviderError:
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(self.name, FailureKind.TIMEOUT, f"{type(exc).__name__}")
        if isinstance(exc, httpx.ConnectError):
            return ProviderError(self.name, FailureKind.CONNECT, str(exc) or "connect failed")
        return ProviderError(
            self.name, FailureKind.CONNECTION_DROPPED, f"{type(exc).__name__}: {exc}"
        )

    def _from_status(self, response: httpx.Response) -> ProviderError:
        return ProviderError(
            self.name,
            classify_status(response.status_code),
            _error_message(response),
            status=response.status_code,
            retry_after_s=_retry_after(response),
        )

    def _timeout(self, what: str, seconds: float) -> ProviderError:
        return ProviderError(self.name, FailureKind.TIMEOUT, f"no {what} within {seconds:g}s")

    # -- Provider --------------------------------------------------------------------------

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        body = request.upstream_body(model, stream_usage=False)
        body["stream"] = False
        total = self.config.timeouts.total_s
        shard = self._pick_shard()
        shard.in_flight += 1
        try:
            async with asyncio.timeout(total), shard.slots:
                response = await shard.client.post("/chat/completions", json=body)
        except TimeoutError:
            raise self._timeout("response", total) from None
        except httpx.HTTPError as exc:
            raise self._from_http_error(exc) from exc
        finally:
            shard.in_flight -= 1
        if response.status_code >= 400:
            raise self._from_status(response)
        try:
            return ChatCompletion.model_validate_json(response.content)
        except ValidationError as exc:
            raise ProviderError(self.name, FailureKind.PROTOCOL, "malformed completion") from exc

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        timeouts = self.config.timeouts
        body = request.upstream_body(model, stream_usage=self.supports_stream_usage)
        shard = self._pick_shard()
        http_request = shard.client.build_request("POST", "/chat/completions", json=body)
        loop = asyncio.get_running_loop()
        first_byte_deadline = loop.time() + timeouts.first_byte_s

        shard.in_flight += 1
        acquired = False
        try:
            async with asyncio.timeout_at(first_byte_deadline):
                await shard.slots.acquire()
                acquired = True
                response = await shard.client.send(http_request, stream=True)
        except BaseException as exc:
            if acquired:
                shard.slots.release()
            shard.in_flight -= 1
            if isinstance(exc, TimeoutError):
                raise self._timeout("response headers", timeouts.first_byte_s) from None
            if isinstance(exc, httpx.HTTPError):
                raise self._from_http_error(exc) from exc
            raise

        events = iter_sse_data(response.aiter_lines())
        try:
            if response.status_code >= 400:
                await response.aread()
                raise self._from_status(response)

            received_any = False
            finished = False
            while True:
                # Deadlines are applied around each read, never across a ``yield``: time the
                # consumer spends writing to a slow client must not count against upstream.
                deadline = loop.time() + timeouts.idle_s if received_any else first_byte_deadline
                try:
                    async with asyncio.timeout_at(deadline):
                        data = await anext(events)
                except StopAsyncIteration:
                    if finished:
                        return
                    raise ProviderError(
                        self.name,
                        FailureKind.CONNECTION_DROPPED,
                        "stream ended before completion",
                    ) from None
                except TimeoutError:
                    what = "stream data" if received_any else "first token"
                    limit = timeouts.idle_s if received_any else timeouts.first_byte_s
                    raise self._timeout(what, limit) from None
                except httpx.HTTPError as exc:
                    raise self._from_http_error(exc) from exc

                if data == DONE_SENTINEL:
                    return
                chunk = self._parse_chunk(data)
                received_any = True
                finished = finished or any(c.get("finish_reason") for c in chunk.choices)
                yield self.normalize_chunk(chunk)
        finally:
            try:
                await events.aclose()
                await response.aclose()
            finally:
                shard.slots.release()
                shard.in_flight -= 1

    def _parse_chunk(self, data: str) -> ChatCompletionChunk:
        try:
            payload: Any = json.loads(data)
        except ValueError as exc:
            raise ProviderError(self.name, FailureKind.PROTOCOL, "invalid JSON in stream") from exc
        if isinstance(payload, dict) and "error" in payload:
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ProviderError(self.name, FailureKind.UPSTREAM_5XX, f"mid-stream error: {message}")
        try:
            return ChatCompletionChunk.model_validate(payload)
        except ValidationError as exc:
            raise ProviderError(self.name, FailureKind.PROTOCOL, "malformed chunk") from exc

    async def health(self) -> bool:
        try:
            response = await self._client.get("/models", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        for shard in self._shards:
            await shard.client.aclose()
