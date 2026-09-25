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


SECTIONS: dict[str, Callable[[], str]] = {
    "semantic-cache": semantic_cache,
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
