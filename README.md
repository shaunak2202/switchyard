# Switchyard

Switchyard is an OpenAI-compatible LLM inference gateway. Point any OpenAI SDK at it by changing
the base URL. Behind that one endpoint it routes requests across providers (Groq, Ollama, and a
fault-injecting mock), streams tokens back over SSE, and fails cleanly when an upstream
misbehaves. It is built as a production-style service: structured logs, request tracing, typed
config, and tests that exercise real sockets and real failure modes.

> **Status:** Phases 1–2 (core proxy, reliability) of 6 are done. Rate limiting, caching,
> observability and load-test results are in progress. Performance numbers are `TBD` until they
> are measured and saved in `results/`.

## Architecture

```mermaid
flowchart LR
    C[Client / OpenAI SDK] -->|POST /v1/chat/completions| MW
    subgraph GW[Switchyard gateway]
        MW[Request ID + JSON access log] --> H[Chat handler]
        H --> S[ChatService<br/>retry · backoff · failover]
        S --> R[Router<br/>model alias → ordered targets]
        S --> CB[Circuit breaker<br/>per provider]
        R --> A1[Groq adapter]
        R --> A2[Ollama adapter]
        R --> A3[Mock adapter]
    end
    A1 --> G[(Groq API)]
    A2 --> O[(Ollama)]
    A3 --> M[(mock-provider)]
    GW -.-> RD[(Redis)]
```

## Quick start

Requires Docker with Compose v2 and Python 3.11 (for the tests).

```bash
git clone <this repo> && cd switchyard
make up            # builds and starts gateway, 2 mock providers and Redis; waits for health
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "mock", "messages": [{"role": "user", "content": "hello"}]}'
```

The same request from the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="mock", messages=[{"role": "user", "content": "hello"}], stream=True
):
    print(chunk.choices[0].delta.content or "", end="") if chunk.choices else None
```

To use real providers, put `GROQ_API_KEY` in `.env` (copied from `.env.example` by `make up`)
and/or run Ollama on the host, then use `"model": "fast"`.

| Command | What it does |
|---|---|
| `make up` / `make down` | Start / stop the stack |
| `make test` | Unit and component tests with coverage |
| `make test-integration` | Tests against the running compose stack |
| `make check` | Lint (ruff), type-check (mypy --strict) and tests: what CI runs |

## Features and design decisions

Longer write-ups are in [`docs/decisions.md`](docs/decisions.md).

- **OpenAI-compatible surface.** `POST /v1/chat/completions` (streaming and non-streaming) and
  `GET /v1/models`. Unknown request fields (tools, `response_format`, ...) pass through
  unchanged; errors use the OpenAI error shape, so SDKs raise their usual exceptions.
- **Routing.** A public model name (`fast`, `local`, `mock`) maps to an ordered list of
  `(provider, upstream model)` targets in `config/gateway.yaml`. Callers can pin a provider with
  `"groq/llama-3.1-8b-instant"`.
- **Provider adapters.** One OpenAI-compatible base class with thin Groq, Ollama and mock
  subclasses. Adapters map every upstream failure onto a `FailureKind` that separates
  *retryable* (same provider) from *failover-able* (next provider). [ADR-001, ADR-002]
- **Streaming that fails honestly.** The first upstream chunk is received before the `200` is
  committed, so early failures return real HTTP errors. A failure after that ends the stream
  with an OpenAI-style error event instead of hanging the client. [ADR-003]
- **Timeouts that match streaming.** Separate first-byte, inter-chunk idle, and total
  deadlines, applied per read so a slow *client* is never mistaken for a slow provider.
  [ADR-004]
- **Retries with full-jitter backoff.** Only retryable failures (timeouts, connect errors,
  5xx, 429) are retried, at most `max_attempts` per provider. The provider's `Retry-After` is a
  floor on the delay, and one request-wide deadline bounds the total time. [ADR-008]
- **Circuit breaker per provider.** Closed → open → half-open, driven by the failure *rate*
  over a sliding window, so intermittent failures still trip it. While it is open, the
  provider is skipped at no cost. [ADR-009, ADR-010]
- **Failover.** Targets are tried in priority order. Responses report
  `x-switchyard-attempts` and `x-switchyard-failovers`, and `/status/providers` shows each
  breaker's state.
- **Tracing and logs.** Every response carries `x-request-id` (the caller's, if it is sane).
  Logs are one JSON object per line with the request ID attached, including logs written from
  inside streaming generators.
- **Health.** `/healthz` (liveness), `/readyz` (own dependencies only; see ADR-005), and
  `/status/providers` (on-demand upstream probes).
- **Mock provider.** Simulates time to first token, token pacing, HTTP errors, hangs, dropped
  connections and stalled streams. Every setting can be changed at runtime through
  `PATCH /admin/config`, which is how tests and chaos load tests inject faults.

## Benchmarks

`TBD`: Phase 6. Tables will be generated from `results/`.

## Failure modes

| Failure | Behaviour |
|---|---|
| Upstream 5xx / 429 / timeout / connect error before first byte | Retry with backoff, then fail over to the next target; `502`/`504` only if every target fails |
| Upstream hangs | Cut off at `first_byte_s` (stream) or `total_s`, then retry or fail over |
| Provider keeps failing | Breaker opens; the provider is skipped without paying a timeout; trial call after `open_s` |
| Every provider's breaker is open | `503 all_circuits_open` with `Retry-After` |
| Upstream 400 / 422 | Not retried or failed over; passed through with upstream status and message |
| Upstream 401 / 403 | Fail over (our credentials, not the caller's problem); `502` if nothing else works |
| Connection dropped or stalled mid-stream | Error SSE event, stream ends within `idle_s`, no `[DONE]`; counts against the breaker; never spliced onto another provider |
| Client disconnects mid-stream | Upstream request is closed; breaker is not charged |
| Retries exceed the request budget | `504 request_timeout` at `request_timeout_s` |
| Unknown model | `404 model_not_found` |

## Limitations

`TBD`: written once the load tests exist.

## License

MIT
