"""Derive a run's reported numbers from its raw files.

    python loadtest/analyze.py results/loadtest/<run>     # rewrites <run>/metrics.json

Everything here works on files saved in the run directory (k6 summary, per-request CSV, raw
Prometheus series, gateway logs), so any number in the README can be recomputed and audited
without re-running the test. Prometheus data is stored as *raw cumulative* counter and bucket
series (not PromQL aggregates); histogram quantiles use the same linear interpolation as
Prometheus's ``histogram_quantile``.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Raw series fetched for every run. Values are cumulative, so any window can be computed later.
PROM_SERIES = {
    "requests": "switchyard_requests_total",
    "overhead_bucket": "switchyard_gateway_overhead_seconds_bucket",
    "ttft_bucket": "switchyard_time_to_first_token_seconds_bucket",
    "duration_bucket": "sum by (le, stream, cache) (switchyard_request_duration_seconds_bucket)",
    "cache_lookups": "switchyard_cache_lookups_total",
    "semantic_rejections": "switchyard_cache_semantic_rejections_total",
    "failovers": "switchyard_failovers_total",
    "circuit_state": "switchyard_circuit_state",
    "upstream_errors": "switchyard_upstream_errors_total",
    "shed": "switchyard_load_shed_total",
    "loop_lag": "switchyard_event_loop_lag_seconds",
    "in_flight": "switchyard_requests_in_flight",
    "rate_limited": "switchyard_rate_limited_total",
    "semantic_skipped": "switchyard_cache_semantic_skipped_total",
    "batch_size_bucket": "switchyard_cache_model_batch_size_bucket",
    "model_seconds_bucket": "switchyard_cache_model_seconds_bucket",
}


def fetch_prometheus(base_url: str, start: float, end: float) -> dict[str, Any]:
    out: dict[str, Any] = {"start": start - 10, "end": end + 10, "step": 5, "series": {}}
    for name, query in PROM_SERIES.items():
        resp = httpx.get(
            f"{base_url}/api/v1/query_range",
            params={"query": query, "start": start - 10, "end": end + 10, "step": 5},
            timeout=30,
        )
        resp.raise_for_status()
        out["series"][name] = [
            {"labels": r["metric"], "values": [[float(t), float(v)] for t, v in r["values"]]}
            for r in resp.json()["data"]["result"]
        ]
    return out


# -- primitives --------------------------------------------------------------------------------


def value_at(values: list[list[float]], t: float) -> float:
    """Latest sample at or before ``t`` (0 before the series starts)."""
    best = 0.0
    for ts, v in values:
        if ts <= t:
            best = v
        else:
            break
    return best


def increase(values: list[list[float]], t0: float, t1: float) -> float:
    """Counter increase over [t0, t1], reset-aware like PromQL's ``increase()``.

    Each run restarts the gateway, so a counter can drop to 0 inside a window (and until the new
    process reports a series, Prometheus may still return the old process's last value). A drop
    between consecutive samples is a reset: the new value counts from zero.
    """
    total = 0.0
    prev = value_at(values, t0)
    for ts, v in values:
        if ts <= t0:
            continue
        if ts > t1:
            break
        total += v - prev if v >= prev else v
        prev = v
    return total


def delta(series: list[dict[str, Any]], t0: float, t1: float, **match: str) -> float:
    return sum(
        increase(s["values"], t0, t1)
        for s in series
        if all(s["labels"].get(k) == v for k, v in match.items())
    )


def bucket_deltas(
    series: list[dict[str, Any]], t0: float, t1: float, **match: str
) -> dict[float, float]:
    buckets: dict[float, float] = defaultdict(float)
    for s in series:
        if all(s["labels"].get(k) == v for k, v in match.items()):
            le = math.inf if s["labels"]["le"] == "+Inf" else float(s["labels"]["le"])
            buckets[le] += increase(s["values"], t0, t1)
    return dict(buckets)


def histogram_quantile(q: float, buckets: dict[float, float]) -> float | None:
    """Prometheus semantics: linear interpolation within the bucket holding the q-th rank."""
    if not buckets:
        return None
    ordered = sorted(buckets.items())
    total = ordered[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in ordered:
        if count >= rank:
            if math.isinf(le):
                return prev_le
            if count == prev_count:
                return le
            return prev_le + (le - prev_le) * (rank - prev_count) / (count - prev_count)
        prev_le, prev_count = le, count
    return ordered[-2][0] if len(ordered) > 1 else None


def _in_overflow(q: float, buckets: dict[float, float]) -> bool:
    """True if the q-th rank lies in the +Inf bucket: the quantile is only a lower bound."""
    finite = [c for le, c in buckets.items() if not math.isinf(le)]
    total = buckets.get(math.inf, 0.0)
    return bool(total) and bool(finite) and max(finite) < q * total


def quantiles_ms(buckets: dict[float, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    capped = []
    for q in (0.5, 0.95, 0.99):
        v = histogram_quantile(q, buckets)
        key = f"p{int(q * 100)}"
        out[key] = None if v is None else round(v * 1000, 3)
        if _in_overflow(q, buckets):
            capped.append(key)
    out["at_least"] = capped  # these quantiles exceed the largest bucket: lower bounds only
    out["count"] = round(buckets.get(math.inf, 0.0))  # the +Inf bucket holds every observation
    return out


def _mean_from_buckets(buckets: dict[float, float]) -> float | None:
    """Approximate mean from a histogram whose buckets are exact integer batch sizes."""
    ordered = sorted(buckets.items())
    total = ordered[-1][1] if ordered else 0
    if not total:
        return None
    acc, prev = 0.0, 0.0
    for le, cum in ordered:
        if not math.isinf(le):
            acc += le * (cum - prev)
            prev = cum
    return round(acc / total, 2)


def trend(summary: dict[str, Any], key: str) -> dict[str, float] | None:
    metric = summary["metrics"].get(key)
    if not metric:
        return None
    v = metric["values"]
    return {
        "p50": round(v["med"], 3),
        "p95": round(v["p(95)"], 3),
        "p99": round(v["p(99)"], 3),
        "max": round(v["max"], 3),
        "count": v.get("count", 0),
    }


def count(summary: dict[str, Any], key: str) -> int:
    metric = summary["metrics"].get(key)
    return int(metric["values"].get("count", 0)) if metric else 0


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(math.ceil(q * len(ordered)) - 1, len(ordered) - 1)]


def read_requests(run_dir: Path) -> list[dict[str, Any]]:
    """One row per request from the ``request_log`` samples, with start/end in epoch ms."""
    rows = []
    with gzip.open(run_dir / "samples.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            if row["metric_name"] != "request_log":
                continue
            tags = dict(p.split("=", 1) for p in row["extra_tags"].split("&") if "=" in p)
            end = float(row["timestamp"])
            duration = float(row["metric_value"])
            rows.append(
                {
                    "start": end - duration,
                    "end": end,
                    "duration": duration,
                    "ok": tags.get("ok") == "true",
                    "code": tags.get("code"),
                    "provider": tags.get("provider"),
                    "stream": tags.get("stream") == "true",
                    "phase": tags.get("phase") or row.get("scenario"),
                    "scenario": row.get("scenario"),
                }
            )
    rows.sort(key=lambda r: r["start"])
    return rows


# -- scenarios ---------------------------------------------------------------------------------


def analyze_ramp(run: dict[str, Any], summary: dict[str, Any], prom: dict[str, Any]) -> dict:
    env = run["env"]
    steps = [int(s) for s in env["STEPS"].split(",")]
    step_s, gap_s = int(env["STEP_SECONDS"]), int(env["GAP_SECONDS"])
    stream = env.get("STREAM") == "1"
    series = prom["series"]
    t_start = run["window"]["start"]
    rows = []
    for i, rate in enumerate(steps):
        tag = str(rate).zfill(5)
        total = count(summary, f"http_reqs{{step:{tag}}}")
        ok = count(summary, f"status_codes{{step:{tag},code:200}}")
        ok_latency = trend(summary, f"http_req_duration{{step:{tag},expected_response:true}}")
        # Prometheus window for the step. Steps are separated by idle gaps, so the window is
        # widened by half a gap on each side: scrapes every 5 s would otherwise clip the edges.
        step_start = t_start + 1 + i * (step_s + gap_s)  # +~1 s for the k6 container to start
        w0 = step_start - gap_s / 2
        w1 = step_start + step_s + gap_s / 2
        row = {
            "offered_rps": rate,
            "sent": total,
            "achieved_rps": round(total / step_s, 1),
            "ok_rps": round(ok / step_s, 1),
            "error_pct": round(100 * (1 - ok / total), 2) if total else None,
            "status": {
                code: count(summary, f"status_codes{{step:{tag},code:{code}}}")
                for code in ("200", "429", "502", "503", "504", "0")
            },
            "dropped_iterations": count(summary, f"dropped_iterations{{step:{tag}}}"),
            "latency_ms": ok_latency,
            "ttfb_ms": trend(summary, f"http_req_waiting{{step:{tag}}}") if stream else None,
            "server_overhead_ms": quantiles_ms(
                bucket_deltas(
                    series["overhead_bucket"], w0, w1, stream="true" if stream else "false"
                )
            ),
            "server_ttft_ms": quantiles_ms(bucket_deltas(series["ttft_bucket"], w0, w1))
            if stream
            else None,
            "shed": round(delta(series["shed"], w0, w1)),
        }
        rows.append(row)

    # Pre-registered rule for "sustainable": <= 1% errors, nothing dropped by k6, and p99 at
    # most 2x the p99 of the lightest step.
    base_p99 = rows[0]["latency_ms"]["p99"] if rows[0]["latency_ms"] else None
    sustainable = [
        r
        for r in rows
        if r["error_pct"] is not None
        and r["error_pct"] <= 1.0
        and r["dropped_iterations"] == 0
        and base_p99
        and r["latency_ms"]
        and r["latency_ms"]["p99"] <= 2 * base_p99
    ]
    best = max(sustainable, key=lambda r: r["offered_rps"]) if sustainable else None
    knee = next((r for r in rows if best and r["offered_rps"] > best["offered_rps"]), None)
    peak_ok = max(rows, key=lambda r: r["ok_rps"])
    return {
        "rule": "sustainable = error <= 1%, no dropped iterations, p99 <= 2x lightest-step p99",
        "steps": rows,
        "headline": {
            "max_sustainable_rps": best["offered_rps"] if best else None,
            "at_max": best,
            "first_unsustainable_rps": knee["offered_rps"] if knee else None,
            "peak_successful_rps": peak_ok["ok_rps"],
        },
    }


def analyze_steady(run: dict[str, Any], summary: dict[str, Any], prom: dict[str, Any]) -> dict:
    outcomes = {
        c: count(summary, f"cache_outcomes{{cache:{c}}}")
        for c in ("hit-exact", "hit-semantic", "miss", "bypass")
    }
    total = sum(outcomes.values()) or 1
    w0, w1 = run["window"]["start"], run["window"]["end"]
    series = prom["series"]
    duration_s = float(run["env"]["DURATION"].rstrip("s"))
    return {
        "headline": {
            "offered_rps": int(run["env"]["RATE"]),
            "requests": count(summary, "http_reqs"),
            "achieved_rps": round(count(summary, "http_reqs") / duration_s, 1),
            "dropped_iterations": count(summary, "dropped_iterations"),
            "failed": count(summary, "request_failures"),
            "hit_rate": round((outcomes["hit-exact"] + outcomes["hit-semantic"]) / total, 4),
            "exact_hit_rate": round(outcomes["hit-exact"] / total, 4),
            "semantic_hit_rate": round(outcomes["hit-semantic"] / total, 4),
        },
        "outcomes": outcomes,
        "latency_ms_by_cache": {
            c: trend(summary, f"latency_by_cache{{cache:{c}}}") for c in outcomes
        },
        "stream_ttfb_ms_by_cache": {
            c: trend(summary, f"stream_ttfb{{cache:{c}}}") for c in outcomes
        },
        "server": {
            "cache_lookups": {
                f"{tier}:{result}": round(
                    delta(series["cache_lookups"], w0, w1, tier=tier, result=result)
                )
                for tier in ("exact", "semantic")
                for result in ("hit", "miss")
            },
            "semantic_verifier_rejections": round(delta(series["semantic_rejections"], w0, w1)),
            "semantic_skipped": {
                f"{stage}:{reason}": round(
                    delta(series["semantic_skipped"], w0, w1, stage=stage, reason=reason)
                )
                for stage in ("lookup", "store")
                for reason in ("budget", "queue_full")
            },
            "model_latency_ms": {
                m: quantiles_ms(bucket_deltas(series["model_seconds_bucket"], w0, w1, model=m))
                for m in ("embedding", "verifier")
            },
            "batch_size_mean": {
                m: _mean_from_buckets(bucket_deltas(series["batch_size_bucket"], w0, w1, model=m))
                for m in ("embed", "verify")
            },
            "overhead_ms": {
                s: quantiles_ms(bucket_deltas(series["overhead_bucket"], w0, w1, stream=s))
                for s in ("false", "true")
            },
            "ttft_ms_by_cache": {
                c: quantiles_ms(bucket_deltas(series["ttft_bucket"], w0, w1, cache=c))
                for c in ("miss", "hit-exact", "hit-semantic")
            },
            "shed": round(delta(series["shed"], w0, w1)),
        },
    }


def _control_times(run_dir: Path) -> dict[str, float]:
    times = {}
    for m in re.finditer(r"CONTROL (\S+) at (\d+)", (run_dir / "k6.log").read_text()):
        times[m.group(1).split(":")[0]] = float(m.group(2))
    return times


def _breaker_events(run_dir: Path) -> list[dict[str, Any]]:
    events = []
    with gzip.open(run_dir / "gateway.log.gz", "rt") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("msg") == "circuit breaker state change":
                ts = datetime.fromisoformat(rec["ts"]).timestamp() * 1000
                events.append({"t_ms": ts, "provider": rec["provider"], "to": rec["to_state"]})
    return events


def analyze_chaos(run_dir: Path, run: dict[str, Any], summary: dict[str, Any]) -> dict:
    requests = [r for r in read_requests(run_dir) if r["phase"] == "traffic"]
    control = _control_times(run_dir)
    fault, recover = control["fault"], control["recover"]
    t0 = requests[0]["start"]

    # Per-second timeline by request *start* time: "what happened to traffic sent at t".
    seconds: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in requests:
        seconds[int((r["start"] - t0) // 1000)].append(r)
    timeline = []
    for s in sorted(seconds):
        rs = seconds[s]
        ok_lat = [r["duration"] for r in rs if r["ok"]]
        by_provider: dict[str, int] = defaultdict(int)
        for r in rs:
            by_provider[r["provider"]] += 1
        timeline.append(
            {
                "t": s,
                "sent": len(rs),
                "failed": sum(not r["ok"] for r in rs),
                "p50_ms": pct(ok_lat, 0.5),
                "p95_ms": pct(ok_lat, 0.95),
                "max_ms": max(ok_lat) if ok_lat else None,
                "by_provider": dict(by_provider),
            }
        )

    def window(a: float, b: float) -> dict[str, Any]:
        rs = [r for r in requests if a <= r["start"] < b]
        lat = [r["duration"] for r in rs if r["ok"]]
        return {
            "sent": len(rs),
            "failed": sum(not r["ok"] for r in rs),
            "error_pct": round(100 * sum(not r["ok"] for r in rs) / len(rs), 3) if rs else None,
            "p50_ms": pct(lat, 0.5),
            "p95_ms": pct(lat, 0.95),
            "p99_ms": pct(lat, 0.99),
            "max_ms": max(lat) if lat else None,
        }

    warm_up = min(10_000, (fault - t0) / 2)  # skip connection warm-up, but keep a baseline
    baseline = window(t0 + warm_up, fault)
    fault_s = int((fault - t0) // 1000)
    recover_s = int((recover - t0) // 1000)
    limit = max(1.5 * (baseline["p95_ms"] or 0), (baseline["p95_ms"] or 0) + 50)
    # Recovery: first second after the fault from which every second until the provider
    # comes back has zero failures and p95 within limit.
    recovered_at = None
    for i, row in enumerate(timeline):
        if row["t"] < fault_s or row["t"] >= recover_s:
            continue
        rest = [x for x in timeline[i:] if x["t"] < recover_s]
        if all(x["failed"] == 0 and (x["p95_ms"] or 0) <= limit for x in rest):
            recovered_at = row["t"]
            break
    breaker = [
        {**e, "t_after_fault_s": round((e["t_ms"] - fault) / 1000, 2)}
        for e in _breaker_events(run_dir)
        if e["t_ms"] >= fault - 1000
    ]
    primary_back = next(
        (
            x["t"] - recover_s
            for x in timeline
            if x["t"] >= recover_s and x["by_provider"].get("mock-primary", 0) > 0.5 * x["sent"]
        ),
        None,
    )
    fault_window = window(fault, recover)
    return {
        "fault": run["env"].get("FAULT"),
        "rule": (
            "recovery = seconds from fault injection until every following second (while the "
            "fault lasts) has 0 failed requests and p95 <= max(1.5x, +50 ms) of pre-fault p95"
        ),
        "headline": {
            "fault": run["env"].get("FAULT"),
            "recovery_s": None if recovered_at is None else recovered_at - fault_s,
            "failed_during_fault": fault_window["failed"],
            "sent_during_fault": fault_window["sent"],
            "error_pct_during_fault": fault_window["error_pct"],
            "p99_ms_during_fault": fault_window["p99_ms"],
            "breaker_open_after_s": next(
                (e["t_after_fault_s"] for e in breaker if e["to"] == "OPEN"), None
            ),
            "primary_serving_again_after_recover_s": primary_back,
        },
        "windows": {
            "before": baseline,
            "during": fault_window,
            "after": window(recover, requests[-1]["start"] + 1),
        },
        "breaker_events": breaker,
        "timeline": timeline,
    }


def analyze_burst(run_dir: Path, run: dict[str, Any], summary: dict[str, Any]) -> dict:
    requests = read_requests(run_dir)
    rpm = run["key_limits"]["rpm"]
    capacity, refill_per_s = rpm, rpm / 60
    t0 = requests[0]["start"]
    admitted = sorted(r["start"] for r in requests if r["code"] == "200")
    # Client-side send time is an upper bound on when the server admitted a request, so the
    # bound check below is conservative against the limiter, not in its favour.
    worst_excess = -math.inf
    for i, t in enumerate(admitted, start=1):
        allowed = capacity + refill_per_s * (t - t0) / 1000
        worst_excess = max(worst_excess, i - allowed)
    burst = [r for r in requests if r["phase"] == "burst"]
    sustain = [r for r in requests if r["phase"] == "sustain"]
    span_s = (requests[-1]["start"] - t0) / 1000

    def lat(rs: list[dict[str, Any]], code: str) -> dict[str, float | None]:
        values = [r["duration"] for r in rs if r["code"] == code]
        return {"p50": pct(values, 0.5), "p99": pct(values, 0.99), "count": len(values)}

    # Steady-state admission: the second half of the sustain phase. The first half still spends
    # tokens that refilled while the bucket sat idle between the phases.
    if sustain:
        s0, s1 = min(r["start"] for r in sustain), max(r["start"] for r in sustain)
        half = s0 + (s1 - s0) / 2
        sustain_ok = [r for r in sustain if r["code"] == "200" and r["start"] >= half]
        sustain_span = (s1 - half) / 1000
    else:
        sustain_ok, sustain_span = [], 0
    return {
        "headline": {
            "key_rpm": rpm,
            "burst_sent": len(burst),
            "burst_admitted": sum(r["code"] == "200" for r in burst),
            "bucket_capacity": capacity,
            "sustain_admitted_per_s": round(len(sustain_ok) / sustain_span, 2)
            if sustain_span
            else None,
            "refill_per_s": refill_per_s,
            "total_admitted": len(admitted),
            "theoretical_max_admitted": math.floor(capacity + refill_per_s * span_s),
            "max_excess_over_bound": round(worst_excess, 2),
            "other_status": sorted({r["code"] for r in requests} - {"200", "429"}),
        },
        "latency_ms": {
            "admitted": lat(requests, "200"),
            "rejected_429": lat(requests, "429"),
        },
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    prom = json.loads((run_dir / "prometheus.json").read_text())
    scenario = run["scenario"]
    if scenario == "ramp":
        metrics = analyze_ramp(run, summary, prom)
    elif scenario == "steady":
        metrics = analyze_steady(run, summary, prom)
    elif scenario == "chaos":
        metrics = analyze_chaos(run_dir, run, summary)
    elif scenario == "burst":
        metrics = analyze_burst(run_dir, run, summary)
    else:
        raise ValueError(scenario)
    metrics = {"scenario": scenario, "variant": run["variant"], **metrics}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(analyze(Path(arg))["headline"], indent=2))
