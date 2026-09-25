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

---

## ADR-008: Full-jitter backoff, `Retry-After` as a floor, one request deadline

**Date:** 2026-09-25 · **Status:** accepted

**Context.** When a provider blips, every in-flight request fails at nearly the same instant.
Deterministic exponential backoff makes them all retry at the same instant too.

**Decision.** Delay before retry *n* is `uniform(0, min(max_delay, base · 2^(n-1)))` (full
jitter). A provider's `Retry-After` is a floor on that delay. If `Retry-After` is longer than
`max_delay_s`, we skip the remaining retries on that provider and fail over, instead of holding
the client. All retries and failovers for one request share one `request_timeout_s` deadline.

**Consequences.** Retrying non-streaming LLM calls is not free: a request that timed out on
our side may still have been generated, and billed, upstream. `max_attempts` defaults to 2 per
provider for that reason.

---

## ADR-009: Failure-rate circuit breaker over a count-based sliding window

**Date:** 2026-09-25 · **Status:** accepted

**Context.** A consecutive-failures breaker never trips on a provider that fails every other
request: each success resets the count.

**Decision.** Each provider has a breaker over its last `window_size` calls. It opens when at
least `min_calls` outcomes are recorded and the failure rate is at least
`failure_rate_threshold`. After `open_s` it goes half-open and admits `half_open_max_calls`
trial calls. Only failures that say something about provider health count: bad requests and
unknown models are excluded, and a client disconnect is neither a success nor a failure. A
stream's outcome is recorded when the stream *ends*, so a mid-stream drop counts against the
provider.

**Consequences.** When a breaker is open, failover to the next provider costs nothing: no
timeout is paid on a provider we already know is bad. Every `allow()` is paired with exactly one
of `record_success`, `record_failure` or `release`, including on cancellation. Otherwise a
half-open breaker would leak its trial permit and stay half-open forever. There is a test for
this.

---

## ADR-010: Breaker state is per process

**Date:** 2026-09-25 · **Status:** accepted

**Decision.** Breakers live in process memory, not in Redis.

**Consequences.** With N workers or replicas, each detects a bad provider on its own, which
costs up to N × `min_calls` failed requests instead of `min_calls`. In exchange, the hot path
makes no extra network round trip per request, and there is no shared-state failure mode (Redis
down should not mean "every breaker is open"). Envoy's outlier detection makes the same
per-host trade-off.

---

## ADR-011: API keys are stored as SHA-256 hashes, not bcrypt/argon2

**Date:** 2026-09-25 · **Status:** accepted

**Context.** Keys must never be stored in plaintext. Password-hashing functions are the reflex
choice.

**Decision.** Keys are 256-bit random tokens (`sk-sy-` + 43 url-safe chars). Only
`sha256(key)` is stored, and it doubles as the Redis lookup key.

**Consequences.** Slow KDFs exist to make brute-forcing *low-entropy* secrets expensive. A
256-bit random key cannot be brute-forced whatever the hash, so a KDF would only add ~50–100 ms
of CPU to every request and rule out O(1) lookup by hash. GitHub and Stripe tokens follow the
same pattern. This reasoning only holds because the gateway generates the keys; if users chose
their own keys, a KDF would be required.

---

## ADR-012: Token bucket in one Lua script: auth + RPM + TPM, Redis clock, all-or-nothing

**Date:** 2026-09-25 · **Status:** accepted

**Context.** Rate limiting has to be exact under concurrency across processes and replicas. A
read-modify-write from the application (GET, compute, SET) admits more than the limit when
requests race. `tests/component/test_limiter.py` contains a control test showing this.

**Decision.**
- **Token bucket, not a fixed or sliding window.** It allows a burst of up to one minute's
  allowance and then a smooth steady rate. There is no doubled burst at window edges (the
  fixed-window flaw), and it stores O(1) state per key instead of a log of timestamps (the
  sliding-log cost). Two buckets per key: requests/min and tokens/min.
- **One script does key lookup, disabled check and both buckets.** Admission is one `EVALSHA`
  round trip. Redis executes scripts serially, so check-and-decrement is atomic.
- **All-or-nothing.** A request is admitted only if both buckets can pay. A rejected request
  charges nothing, so a client hammering at the limit doesn't dig itself deeper.
- **Redis `TIME`, not the gateway's clock**, so replicas with clock skew cannot over-refill.
- **Cluster-ready keys.** `sy:{<hash>}:meta|rpm|tpm` share a hash tag and so always land in one
  cluster slot, which a multi-key script requires.
- **Estimate, then reconcile.** Token usage is known only after the response. We pre-charge
  `prompt_estimate + max_tokens` (or `default_max_tokens`) and refund or charge the difference
  once the provider reports usage. Streams report it in the final chunk; if absent, we estimate
  from the streamed characters. Abandoned streams are charged for what was produced, and
  failed calls are refunded in full. A bucket can go into debt, which refill repays.

**Consequences.** Admission costs one Redis round trip (~0.1–0.3 ms locally). The concurrency
tests fire 400 simultaneous admissions across 8 independent connection pools, and 150 parallel
HTTP requests through the gateway, and assert the limit holds *exactly*. Raising a key's limit
changes its refill rate immediately but does not grant a fresh burst.

---

## ADR-013: Fail closed when Redis is unavailable

**Date:** 2026-09-25 · **Status:** accepted

**Decision.** If Redis cannot be reached during admission, the request gets `503
auth_unavailable`, and `/readyz` reports not-ready so load balancers drain the replica.

**Consequences.** Failing open would be friendlier to availability for rate limiting alone.
But authentication shares that round trip, and "Redis is down, so anyone can use our provider
keys" is not acceptable. A deployment that wants fail-open rate limiting would need a separate
auth path, for example a local key cache.

---

## ADR-014: Blocking Redis connection pool

**Date:** 2026-09-25 · **Status:** accepted

**Context.** Found while writing the concurrency tests: redis-py's default asyncio
`ConnectionPool` raises `MaxConnectionsError` the moment every connection is in use, rather
than waiting.

**Decision.** The gateway uses `BlockingConnectionPool`. When the pool is exhausted, callers
wait up to `socket_timeout_s` for a connection.

**Consequences.** A traffic spike above the pool size adds queueing latency instead of an
immediate 503. Pool size (`redis.max_connections`) becomes a tuning knob that the load tests
exercise.

---

## ADR-015: Cache design: exact first, deterministic requests only, one entry shape

**Date:** 2026-09-25 · **Status:** accepted

**Decision.**
- **Eligibility.** Only `temperature: 0` requests are cached, unless the caller sends
  `X-Switchyard-Cache: allow`. With temperature > 0 (or unset, the provider default) the
  caller asked for a *sample*, and replaying one sample to everyone changes the API's meaning.
- **Exact key.** SHA-256 of the canonical JSON of every output-affecting field. Normalisation
  only removes differences that cannot change the answer: key order, `\r\n`, surrounding
  whitespace, a single text part vs a string, stop-sequence order, and `max_completion_tokens`
  vs `max_tokens`.
- **Caller control.** `Cache-Control: no-cache` skips the lookup, `no-store` skips the
  write, and `X-Switchyard-Cache: no-semantic` restricts matching to exact.
- **One entry shape.** Entries are stored as non-streaming completions. Streaming misses are
  assembled as they are relayed, and streaming hits are replayed as synthetic chunks, so one
  entry serves both kinds of caller. Streams with tool calls or several choices are not cached.
- **Only finished answers are stored.** Errors, aborted streams and content-filtered answers
  never are. Writes happen in a tracked background task after the response is sent.
- **Hits** count against requests/min but refund the token estimate: they consume no
  provider tokens.
- **Failure is a miss.** A Redis error during lookup or store is logged and treated as a miss.
  The cache must never be why a request fails.

---

## ADR-016: Semantic matching is two-stage, with thresholds chosen from labelled data

**Date:** 2026-09-25 · **Status:** accepted

**Context.** A semantic hit serves an answer generated for a *different* prompt, so a false hit
is a wrong answer. The acceptance rule was fixed **before** running the evaluation: *the lowest
threshold whose false-hit rate is ≤ 1% on every evaluation source.* The sources are 155
hand-labelled LLM-style prompt pairs (`data/semantic_pairs.jsonl`, heavy on hard negatives),
2,000 pairs from PAWS (adversarial word-order swaps) and 2,000 from QQP (real user questions).

**Findings** (`results/semantic-threshold/20260925T184805Z/`):
- **Embedding-only matching (all-MiniLM-L6-v2 cosine) cannot meet the rule at any useful
  threshold.** The only compliant value is 1.0, which means no hits at all. At 0.95, the
  threshold that satisfies QQP alone, it still serves a wrong answer for 9.5% of the
  hand-labelled non-paraphrases and 77.5% of PAWS non-paraphrases. The worst category is
  direction reversals ("5 km to miles" vs "5 miles to km", similarity 0.993): 7 of 10 get
  through.
- **Two-stage matching meets the rule.** The embedding proposes the nearest cached prompt as a
  candidate (similarity ≥ 0.80), and a cross-encoder (`cross-encoder/quora-distilroberta-base`)
  must score the pair ≥ 0.992. This keeps the false-hit rate ≤ 1% on all three sources, with
  **0 false hits on the hand-labelled hard negatives in every category**. The cost is recall:
  the hit rate on paraphrases is low (8% on the prompt set, 29% on QQP).

**Decision.** Ship the two-stage configuration and enable it by default. A cache that is
rarely wrong and sometimes helps is worth running; one that is often wrong is not. Embedding-only
mode still exists (`verifier_model: null`) but logs a warning at startup.

**Consequences.**
- The verifier adds ~6 ms per semantic *candidate* (p50, measured), and nothing when no
  candidate exists. The embedding adds ~3 ms to every cache-eligible request that misses the
  exact cache.
- The verifier was trained on Quora duplicate questions, so its QQP numbers may be optimistic.
  The prompt and PAWS sets are not affected, and those are what bind the threshold.
- Per-pair rates are not live hit rates. In production the chance of a false hit grows with
  the number of near-neighbours in the cache, so the rule should be re-checked on real traffic.
- Changing either model invalidates the thresholds. Re-run
  `scripts/eval_semantic_threshold.py` first.
