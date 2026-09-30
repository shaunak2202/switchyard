"""Regenerate the measured sections of README.md from results/.

Every number in the README's generated sections is read from a results file. Sections are
delimited by marker comments, and anything between the markers is overwritten:

    <!-- generated:semantic-cache -->  ...  <!-- /generated:semantic-cache -->

Run ``python scripts/render_readme.py`` (or ``make readme``). ``--check`` exits non-zero if the
README is out of date, so CI can enforce it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"


def latest(kind: str) -> Path | None:
    runs = sorted(p for p in (RESULTS / kind).glob("*") if p.is_dir())
    return runs[-1] if runs else None


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def semantic_cache() -> str:
    run = latest("semantic-threshold")
    if run is None:
        return "`TBD`: run `python scripts/eval_semantic_threshold.py`."
    metrics: dict[str, Any] = json.loads((run / "metrics.json").read_text())
    env: dict[str, Any] = json.loads((run / "environment.json").read_text())
    rel = run.relative_to(ROOT)

    rows = []

    def add(label: str, threshold: Any, profile: dict[str, Any]) -> None:
        at = profile["at_chosen"]
        if threshold is None or not at:
            rows.append(f"| {label} | none meets the rule | n/a | n/a | n/a |")
            return
        cells = " | ".join(
            f"{pct(at[s]['hit_rate'])} / **{pct(at[s]['false_hit_rate'])}**"
            for s in ("prompts", "paws", "qqp")
        )
        rows.append(f"| {label} | {threshold} | {cells} |")

    emb = metrics["embedding"]
    for name, profile in emb["profiles"].items():
        add(f"Embedding only, `{name}` profile", profile["chosen_threshold"], profile)
    for verifier, data in metrics["two_stage"].items():
        for name, profile in data["profiles"].items():
            t = profile["chosen_threshold"]
            model = verifier.split("/")[-1]
            label = f"Embedding ≥ {data['candidate_threshold']} + `{model}`, `{name}` profile"
            add(label, t, profile)

    datasets = env["datasets"]
    sizes = {
        "prompts": sum(
            1 for line in (run / "scores.jsonl").read_text().splitlines() if '"prompts"' in line
        ),
        "paws": datasets["paws"]["sampled"],
        "qqp": datasets["qqp"]["sampled"],
    }
    latency = metrics["embedding_latency_single_prompt"]
    verifier_latency = {
        name.split("/")[-1]: data["single_pair_latency_ms"]
        for name, data in metrics["two_stage"].items()
    }
    hw = env["hardware"]
    lines = [
        f"Rule: **{metrics['rule']}** (fixed before running the evaluation).",
        "",
        "| Configuration | Threshold | prompts hit / **false hit** | PAWS hit / **false hit** "
        "| QQP hit / **false hit** |",
        "|---|---|---|---|---|",
        *rows,
        "",
        f"Pairs: prompts = {sizes['prompts']} (hand-labelled, `data/semantic_pairs.jsonl`), "
        f"PAWS = {sizes['paws']}, QQP = {sizes['qqp']} (stratified samples, seed {env['seed']}). "
        "*Hit* = share of true paraphrase pairs matched; *false hit* = share of "
        "non-paraphrase pairs matched, i.e. a wrong answer served.",
        "",
        f"Latency on {hw.get('cpu', hw['machine'])}: embedding p50 {latency['p50_ms']} ms / "
        f"p95 {latency['p95_ms']} ms per prompt; "
        + ", ".join(
            f"verifier `{name}` p50 {v['p50']} ms / p95 {v['p95']} ms per pair"
            for name, v in verifier_latency.items()
        )
        + ".",
        "",
        f"Source: [`{rel}`]({rel}/metrics.json)",
    ]
    return "\n".join(lines)


# -- load tests ------------------------------------------------------------------------------


def load_runs() -> dict[tuple[str, str], Path]:
    """Latest run directory per (scenario, variant)."""
    latest_runs: dict[tuple[str, str], Path] = {}
    for run in sorted((RESULTS / "loadtest").glob("*__*__*")):
        if (run / "metrics.json").exists():
            _, scenario, variant = run.name.split("__")
            latest_runs[(scenario, variant)] = run
    return latest_runs


def _metrics(run: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run / "metrics.json").read_text())
    return data


def _link(run: Path) -> str:
    rel = run.relative_to(ROOT)
    return f"[`{rel.name}`]({rel})"


def _ms(value: Any) -> str:
    return "n/a" if value is None else f"{value:,.1f}"


TBD = "`TBD`: run `make loadtest`."


def _q(hist: dict[str, Any] | None, key: str) -> str:
    """A histogram quantile; '≥' when it lies beyond the largest bucket (a lower bound)."""
    if not hist or hist.get(key) is None:
        return "n/a"
    prefix = "≥ " if key in hist.get("at_least", []) else ""
    return f"{prefix}{hist[key]:,.1f}"


def loadtest_env() -> str:
    runs = load_runs()
    if not runs:
        return TBD
    env = json.loads((next(iter(runs.values())) / "environment.json").read_text())
    host, docker = env["host"], env["docker"]
    vm = env.get("vm") or {}
    return (
        f"**Hardware:** {host['cpu']} ({host['cores']} cores, {host['memory_gb']} GB) running "
        f"Docker {docker['server_version']} in a Colima VM with {docker['ncpu']} vCPUs and "
        f"{docker['mem_total_gb']} GB ({vm.get('arch', '')} {vm.get('runtime', '')}). "
        f"{env['notes']} k6: `{env['k6_image']}`. Mock providers: ~50 ms to first token, 32 "
        "tokens at 5 ms each (~210 ms per non-streaming call). Every run directory also holds "
        "the gateway config, mock settings, k6 scripts, raw k6 summary and raw Prometheus data."
    )


def loadtest_ramp() -> str:
    runs = load_runs()
    default = runs.get(("ramp", "default"))
    if default is None:
        return TBD
    mock = runs.get(("ramp", "mock-direct"))
    mock_steps = {r["offered_rps"]: r for r in _metrics(mock)["steps"]} if mock is not None else {}
    m = _metrics(default)
    lines = [
        f"Rule: {m['rule']}.",
        "",
        "| Offered req/s | Successful req/s | Errors | p50 | p95 | p99 | Overhead p50 / p99 "
        "(server) | p50 over mock-direct | Shed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in m["steps"]:
        lat = r["latency_ms"] or {}
        oh = r["server_overhead_ms"]
        base = (mock_steps.get(r["offered_rps"]) or {}).get("latency_ms") or {}
        client_oh = (
            f"{lat['p50'] - base['p50']:+.1f} ms" if lat.get("p50") and base.get("p50") else "n/a"
        )
        lines.append(
            f"| {r['offered_rps']} | {r['ok_rps']} | {r['error_pct']}% | {_ms(lat.get('p50'))} "
            f"| {_ms(lat.get('p95'))} | {_ms(lat.get('p99'))} | {_q(oh, 'p50')} / "
            f"{_q(oh, 'p99')} ms | {client_oh} | {r['shed']} |"
        )
    lines += [
        "",
        "Latencies are client-side (k6) for successful requests, in ms. *Overhead (server)* is "
        "the gateway's own histogram: total time minus the upstream call. *p50 over "
        "mock-direct* compares with k6 hitting the mock directly at the same rate.",
        "",
        "**Variants** (same steps; the headline rule applied to each):",
        "",
        "| Variant | Max sustainable req/s | Peak successful req/s | p99 at max (ms) | Run |",
        "|---|---|---|---|---|",
    ]
    labels = {
        "default": "As shipped (sharded pools, load shedding)",
        "no-shedding": "Load shedding off",
        "single-httpx-pool": "Single httpx pool per provider, no shedding (pre-fix)",
        "streaming": "Streaming requests",
        "mock-direct": "Mock alone (no gateway): baseline",
    }
    for variant, label in labels.items():
        run = runs.get(("ramp", variant))
        if run is None:
            continue
        h = _metrics(run)["headline"]
        at = h["at_max"] or {}
        p99 = (at.get("latency_ms") or {}).get("p99")
        lines.append(
            f"| {label} | {h['max_sustainable_rps'] or 'none'} | {h['peak_successful_rps']} "
            f"| {_ms(p99)} | {_link(run)} |"
        )
    return "\n".join(lines)


def loadtest_streaming() -> str:
    run = load_runs().get(("ramp", "streaming"))
    if run is None:
        return TBD
    m = _metrics(run)
    lines = [
        "| Offered req/s | Successful req/s | Errors | TTFT p50 / p95 / p99 (server) | "
        "TTFB p50 / p99 (client) | Overhead to first byte p50 / p99 |",
        "|---|---|---|---|---|---|",
    ]
    for r in m["steps"]:
        t = r["server_ttft_ms"] or {}
        b = r["ttfb_ms"] or {}
        oh = r["server_overhead_ms"]
        lines.append(
            f"| {r['offered_rps']} | {r['ok_rps']} | {r['error_pct']}% | {_q(t, 'p50')} / "
            f"{_q(t, 'p95')} / {_q(t, 'p99')} ms | {_ms(b.get('p50'))} / "
            f"{_ms(b.get('p99'))} ms | {_q(oh, 'p50')} / {_q(oh, 'p99')} ms |"
        )
    lines += [
        "",
        "TTFT = request received to first *content* token sent (the mock's first content token "
        "comes ~5 ms after its role chunk). TTFB is measured by k6. Source: " + _link(run),
    ]
    return "\n".join(lines)


def loadtest_cache() -> str:
    runs = load_runs()
    rows = []
    labels = {
        "default": "Exact only (as shipped)",
        "semantic": "Exact + semantic, 50 ms budget",
        "semantic-budget-250ms": "Exact + semantic, 250 ms budget",
    }
    for variant, label in labels.items():
        run = runs.get(("steady", variant))
        if run is None:
            continue
        m = _metrics(run)
        h = m["headline"]
        lat = m["latency_ms_by_cache"]
        hit, miss = lat.get("hit-exact") or {}, lat.get("miss") or {}
        rows.append(
            f"| {label} | {h['achieved_rps']} | {h['failed']} | {h['exact_hit_rate'] * 100:.1f}% "
            f"| {h['semantic_hit_rate'] * 100:.2f}% | {_ms(hit.get('p50'))} / "
            f"{_ms(hit.get('p99'))} | {_ms(miss.get('p50'))} / {_ms(miss.get('p99'))} "
            f"| {_link(run)} |"
        )
    if not rows:
        return TBD
    first = runs.get(("steady", "default"))
    env = json.loads((first / "run.json").read_text())["env"] if first else {}
    return "\n".join(
        [
            f"{env.get('RATE', '?')} req/s for {env.get('DURATION', '?')}; 80% of requests ask "
            "one of 1,000 questions with Zipf(1.1) popularity (15% of those re-typed in lower "
            "case without punctuation), 20% are one-off prompts; half stream; all "
            "`temperature: 0`. Latency in ms (client-side, successful requests).",
            "",
            "| Configuration | req/s | Failed | Exact hits | Semantic hits | Hit p50 / p99 "
            "| Miss p50 / p99 | Run |",
            "|---|---|---|---|---|---|---|---|",
            *rows,
        ]
    )


def loadtest_chaos() -> str:
    runs = load_runs()
    rows = []
    labels = {
        "errors": "Primary returns 500 on every call",
        "hang": "Primary accepts and never answers",
        "drop": "Primary drops streams after 3 tokens",
    }
    for variant, label in labels.items():
        run = runs.get(("chaos", variant))
        if run is None:
            continue
        m = _metrics(run)
        h = m["headline"]
        before = m["windows"]["before"]
        rec = h["recovery_s"]
        opened = h["breaker_open_after_s"]
        rows.append(
            f"| {label} | {h['failed_during_fault']} / {h['sent_during_fault']} "
            f"({h['error_pct_during_fault']}%) | {_ms(before['p99_ms'])} → "
            f"{_ms(h['p99_ms_during_fault'])} | "
            f"{'n/a' if opened is None else f'{opened} s'}"
            f" | {'not within the fault window' if rec is None else str(rec) + ' s'} "
            f"| {h['primary_serving_again_after_recover_s']} s | {_link(run)} |"
        )
    if not rows:
        return TBD
    first = next(runs[k] for k in runs if k[0] == "chaos")
    env = json.loads((first / "run.json").read_text())["env"]
    return "\n".join(
        [
            f"{env['RATE']} req/s (half streaming) for {env['DURATION_SECONDS']} s; the primary "
            f"mock is broken at {env['FAULT_AT']} s and repaired at {env['RECOVER_AT']} s; "
            "`mock-secondary` is the failover target.",
            "",
            "| Fault | Failed / sent during fault | p99 before → during (ms) | Breaker opened "
            "after | Recovery time | Primary back after repair | Run |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
            f"Recovery rule: {_metrics(first)['rule']}.",
        ]
    )


def loadtest_burst() -> str:
    run = load_runs().get(("burst", "default"))
    if run is None:
        return TBD
    m = _metrics(run)
    h = m["headline"]
    lat = m["latency_ms"]
    other = ", ".join(h["other_status"]) or "none"
    return "\n".join(
        [
            "| Key limit | Burst sent / admitted | Bucket capacity | Admitted in total / "
            "theoretical max | Worst excess over bound | Steady-state admission | 429 p50 / p99 "
            "| 200 p50 / p99 | Other statuses |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| {h['key_rpm']} req/min | {h['burst_sent']} / {h['burst_admitted']} "
            f"| {h['bucket_capacity']} | {h['total_admitted']} / "
            f"{h['theoretical_max_admitted']} | {h['max_excess_over_bound']} requests "
            f"| {h['sustain_admitted_per_s']} req/s (refill {h['refill_per_s']:g}/s) "
            f"| {_ms(lat['rejected_429']['p50'])} / {_ms(lat['rejected_429']['p99'])} ms "
            f"| {_ms(lat['admitted']['p50'])} / {_ms(lat['admitted']['p99'])} ms | {other} |",
            "",
            "The bound is `capacity + refill * elapsed`, checked against every admitted request "
            "using client-side send times (so it is slightly *stricter* than what the server "
            "saw). Source: " + _link(run),
        ]
    )


def grafana_screenshot() -> str:
    shots = sorted((RESULTS / "grafana").glob("*/dashboard.png"))
    if not shots:
        return "`TBD`: run `python scripts/grafana_screenshot.py <run-dir>` after `make loadtest`."
    shot = shots[-1]
    info = json.loads((shot.parent / "info.json").read_text())
    rel = shot.relative_to(ROOT)
    return (
        f"![Grafana dashboard during the {info['scenario']} run]({rel})\n\n"
        f"*The provisioned Grafana dashboard over the window of run `{info['run']}` "
        f"({info['description']}). Captured by `scripts/grafana_screenshot.py`.*"
    )


SECTIONS: dict[str, Callable[[], str]] = {
    "grafana-screenshot": grafana_screenshot,
    "semantic-cache": semantic_cache,
    "loadtest-env": loadtest_env,
    "loadtest-ramp": loadtest_ramp,
    "loadtest-streaming": loadtest_streaming,
    "loadtest-cache": loadtest_cache,
    "loadtest-chaos": loadtest_chaos,
    "loadtest-burst": loadtest_burst,
}


def render(text: str) -> str:
    for name, build in SECTIONS.items():
        pattern = re.compile(
            rf"(<!-- generated:{name} -->).*?(<!-- /generated:{name} -->)", re.DOTALL
        )
        if not pattern.search(text):
            continue
        text = pattern.sub(
            lambda m, b=build: f"{m.group(1)}\n{b()}\n{m.group(2)}",
            text,  # type: ignore[misc]
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if README is stale")
    args = parser.parse_args()
    current = README.read_text()
    updated = render(current)
    if args.check:
        if updated != current:
            print("README.md is out of date: run `make readme`", file=sys.stderr)
            return 1
        return 0
    README.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
