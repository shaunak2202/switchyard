# Architecture decision records

Short, dated records of decisions that are not obvious from the code. Newest last.

---

## ADR-001: One OpenAI-compatible adapter base, thin per-provider subclasses

**Date:** 2026-09-25 · **Status:** accepted

**Context.** Groq, Ollama (`/v1`) and our mock provider all speak the OpenAI chat-completions
wire format. They differ in authentication, health endpoints and small extensions (Groq reports
streaming usage in `x_groq.usage`).

**Decision.** `Provider` is an abstract interface (`complete`, `stream`, `health`, `aclose`).
`OpenAICompatibleProvider` implements it once; `GroqProvider`, `OllamaProvider` and
`MockProvider` override only what differs. Adapters are the only code that knows provider error
shapes: they map every failure onto `ProviderError(kind: FailureKind)`.

**Consequences.** Adding an OpenAI-compatible provider is config-only (`type: openai`). A
non-OpenAI provider (for example Anthropic's native API) would be a new `Provider` subclass
that translates formats; nothing above the adapter layer changes.

---

## ADR-002: Separate "retryable" from "failover-able" failures

**Date:** 2026-09-25 · **Status:** accepted

**Context.** Treating "should I retry?" as a single yes/no is wrong for a multi-provider
gateway. A 401 from Groq (bad key) will never succeed on retry, but Ollama may serve the request
fine. A 400 (malformed request) will fail on every provider.

**Decision.** `FailureKind` has two properties. `retryable` (same provider): timeouts, connect
errors, dropped connections, 5xx, 429. `failover` (next provider): everything except
`BAD_REQUEST`.

**Consequences.** Upstream 400/422 errors are returned to the caller with the upstream status
and message, because the caller can fix them. Upstream 401/403 become 502: the caller cannot fix
our provider credentials, and the upstream message could leak key fragments.

---

## ADR-003: Pull the first chunk before committing a streaming response

**Date:** 2026-09-25 · **Status:** accepted

**Context.** With SSE, the HTTP status line is sent before the body. If we return a
`StreamingResponse` right away, an upstream failure before the first token can only be reported
as an in-band error event with a misleading `200`.

**Decision.** `ChatService.stream` awaits the first upstream chunk before the handler returns.
Failures up to that point produce a real HTTP error (and, from Phase 2, trigger retry and
failover). After the first byte, a failure produces an OpenAI-style `data: {"error": ...}` event
and the stream ends without `[DONE]`; the OpenAI SDKs raise `APIError` on that event.

**Consequences.** Failover is possible only before the first byte. Resuming a half-finished
generation on another provider would splice two different models' outputs into one answer,
which is worse than a clean error.

---

## ADR-004: Deadlines per read, not across `yield`

**Date:** 2026-09-25 · **Status:** accepted

**Context.** A single `asyncio.timeout` around a streaming generator also counts time spent
suspended at `yield`, that is, time the *client* takes to read. A slow client would then look
like a slow provider.

**Decision.** Streams get three bounds: `first_byte_s` (request start → first chunk), `idle_s`
(gap between chunks) and, for non-streaming calls, `total_s`. Each is applied around a single
`anext()` on the upstream iterator, never across a `yield`. httpx's own read timeout is set
higher and is only a backstop.

**Consequences.** A healthy stream that keeps producing tokens is never cut off by a total
deadline, while a stalled one is detected within `idle_s`.

---

## ADR-005: Readiness excludes upstream providers

**Date:** 2026-09-25 · **Status:** accepted

**Decision.** `/readyz` checks only the gateway's own dependencies (config now, Redis from
Phase 3). Provider health is exposed separately at `/status/providers`.

**Consequences.** When a provider has an outage, requests fail over; replicas do not all turn
unready at once, which would turn a partial outage into a total one.

---

## ADR-006: Raw ASGI middleware for request context

**Date:** 2026-09-25 · **Status:** accepted

**Decision.** Request IDs and access logging use a raw ASGI middleware rather than Starlette's
`BaseHTTPMiddleware`, which wraps streaming bodies and has a history of contextvar and
cancellation problems. The access log is written after the last body byte, so a streamed
request's logged duration is its real duration.

---

## ADR-007: Component tests use real sockets, not ASGI transports

**Date:** 2026-09-25 · **Status:** accepted

**Decision.** Component tests start the gateway and two mock providers under uvicorn on
ephemeral ports in background threads. httpx's `ASGITransport` buffers response bodies, so it
cannot show that streaming really streams, and it cannot simulate a dropped TCP connection.
Container-level tests (`-m integration`) additionally run against `docker compose`.
