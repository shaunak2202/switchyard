"""Prometheus metrics.

Naming follows Prometheus conventions (``_total`` counters, ``_seconds`` histograms, base
units). Label values are always from a bounded set: route aliases from config, provider names
from config, enums. User-controlled strings, such as a pinned ``provider/model`` or an API key,
never become label values, because unbounded label cardinality is how a metrics backend falls
over.

The gateway runs one process per container (ADR-017), so the default in-process registry is
exact. Multi-worker deployments would need prometheus_client's multiprocess mode.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Latency buckets tuned for an LLM gateway: sub-millisecond cache hits through to
# multi-second generations.
LATENCY_BUCKETS = (
    0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 30.0, 60.0,
)  # fmt: skip
# Gateway overhead is expected to be small: finer resolution at the low end.
OVERHEAD_BUCKETS = (
    0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075,
    0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0,
)  # fmt: skip
MODEL_BUCKETS = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1, 0.25)

REQUESTS = Counter(
    "switchyard_requests_total",
    "Chat completion requests by final outcome.",
    ["route", "stream", "status", "cache"],
)
REQUEST_DURATION = Histogram(
    "switchyard_request_duration_seconds",
    "Time from request received to last byte sent (whole stream for streaming requests).",
    ["route", "stream", "cache"],
    buckets=LATENCY_BUCKETS,
)
TTFT = Histogram(
    "switchyard_time_to_first_token_seconds",
    "Streaming: time from request received to the first content token sent to the client.",
    ["route", "cache"],
    buckets=LATENCY_BUCKETS,
)
OVERHEAD = Histogram(
    "switchyard_gateway_overhead_seconds",
    "Time spent in the gateway itself: total (or TTFT for streams) minus upstream time. "
    "Upstream-served requests only.",
    ["stream"],
    buckets=OVERHEAD_BUCKETS,
)
IN_FLIGHT = Gauge("switchyard_requests_in_flight", "Chat completion requests being served.")

UPSTREAM_ATTEMPTS = Counter(
    "switchyard_upstream_attempts_total",
    "Calls to providers, including retries. outcome is 'success' or a failure kind.",
    ["provider", "outcome"],
)
UPSTREAM_ERRORS = Counter(
    "switchyard_upstream_errors_total",
    "Failed provider calls by failure kind (subset of attempts).",
    ["provider", "kind"],
)
UPSTREAM_DURATION = Histogram(
    "switchyard_upstream_duration_seconds",
    "Provider latency: whole response (non-streaming) or time to first chunk (streaming).",
    ["provider", "stream"],
    buckets=LATENCY_BUCKETS,
)
RETRIES = Counter("switchyard_retries_total", "Retries against the same provider.", ["provider"])
FAILOVERS = Counter(
    "switchyard_failovers_total",
    "Moves from one provider to the next within a request.",
    ["from_provider", "to_provider"],
)
MID_STREAM_FAILURES = Counter(
    "switchyard_mid_stream_failures_total",
    "Streams that failed after the first byte had been sent.",
    ["provider", "kind"],
)
CIRCUIT_STATE = Gauge(
    "switchyard_circuit_state",
    "Circuit breaker state per provider: 0 closed, 1 half-open, 2 open.",
    ["provider"],
)
CIRCUIT_TRANSITIONS = Counter(
    "switchyard_circuit_transitions_total",
    "Circuit breaker state changes.",
    ["provider", "to_state"],
)

CACHE_LOOKUPS = Counter(
    "switchyard_cache_lookups_total",
    "Cache lookups by tier and result. A request that misses exact and then tries semantic "
    "is counted once per tier.",
    ["tier", "result"],
)
CACHE_VERIFIER_REJECTIONS = Counter(
    "switchyard_cache_semantic_rejections_total",
    "Semantic candidates found by the embedding but vetoed by the verifier.",
)
EMBEDDING_DURATION = Histogram(
    "switchyard_cache_model_seconds",
    "Latency of the semantic-cache models.",
    ["model"],
    buckets=MODEL_BUCKETS,
)

RATE_LIMITED = Counter(
    "switchyard_rate_limited_total",
    "Requests rejected with 429, by which bucket was exhausted.",
    ["limit"],
)
AUTH_FAILURES = Counter("switchyard_auth_failures_total", "Requests rejected with 401.", ["reason"])
TOKENS = Counter(
    "switchyard_tokens_total",
    "Tokens reported by providers.",
    ["provider", "type"],
)

LOOP_LAG = Gauge(
    "switchyard_event_loop_lag_seconds",
    "Smoothed event-loop lag: how late a timer fires. The overload signal (ADR-020).",
)
SHED = Counter(
    "switchyard_load_shed_total",
    "Requests refused with 503 because the event loop was overloaded.",
)
SEMANTIC_SKIPPED = Counter(
    "switchyard_cache_semantic_skipped_total",
    "Semantic lookups or stores skipped to protect latency (ADR-021).",
    ["stage", "reason"],
)
BATCH_SIZE = Histogram(
    "switchyard_cache_model_batch_size",
    "Prompts per semantic-cache model forward pass.",
    ["model"],
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
