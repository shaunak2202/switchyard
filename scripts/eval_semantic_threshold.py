"""Choose the semantic-cache similarity threshold from labelled data.

A semantic cache hit returns an answer generated for a *different* prompt. A false hit serves a
wrong answer, which is far worse than a miss (a miss only costs one provider call). So the rule
is fixed before looking at the data:

    threshold = the lowest value whose false-hit rate is <= --max-false-hit-rate
                on EVERY source in the profile

Lowest, because a lower threshold means more hits. Every source, because averaging would let
an easy source hide failures on a hard one. Two profiles are reported:

  * strict: prompts + paws + qqp. General LLM traffic, where near-miss prompts are common.
  * faq:    qqp only. FAQ/support-style traffic, where users re-ask the same questions.

Optionally (--verifier) a two-stage design is evaluated as well: the embedding proposes a
candidate (similarity >= --candidate-threshold) and a cross-encoder must confirm it.

Sources:
  * prompts: hand-labelled LLM-style prompt pairs in data/semantic_pairs.jsonl, heavy on hard
    negatives (swapped entities, numbers, direction, negation, language, format).
  * paws: PAWS labeled_final test split. Adversarial pairs with high word overlap.
  * qqp: Quora Question Pairs (GLUE validation). Real user questions, duplicate or not.

The public datasets are sampled with a fixed seed and are not redistributed: only row indices,
labels and similarities are written to results/.

    python scripts/eval_semantic_threshold.py            # writes results/semantic-threshold/<ts>/
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Pair:
    source: str
    a: str
    b: str
    label: int
    ref: str  # row index for public data, category for hand-labelled data


def load_prompts() -> list[Pair]:
    pairs = []
    for line in (ROOT / "data" / "semantic_pairs.jsonl").read_text().splitlines():
        row = json.loads(line)
        pairs.append(Pair("prompts", row["a"], row["b"], int(row["label"]), row["category"]))
    return pairs


def load_public(sample: int, seed: int) -> tuple[list[Pair], dict[str, Any]]:
    from datasets import load_dataset

    specs = {
        "paws": (
            "google-research-datasets/paws",
            "labeled_final",
            "test",
            "sentence1",
            "sentence2",
        ),
        "qqp": ("nyu-mll/glue", "qqp", "validation", "question1", "question2"),
    }
    pairs: list[Pair] = []
    provenance: dict[str, Any] = {}
    for source, (name, config, split, col_a, col_b) in specs.items():
        ds = load_dataset(name, config, split=split)
        rng = random.Random(seed)
        # Stratified sample: equal positives and negatives, so both rates are well estimated.
        positives = [i for i, y in enumerate(ds["label"]) if y == 1]
        negatives = [i for i, y in enumerate(ds["label"]) if y == 0]
        chosen = rng.sample(positives, sample // 2) + rng.sample(negatives, sample // 2)
        for i in sorted(chosen):
            row = ds[i]
            pairs.append(Pair(source, row[col_a], row[col_b], int(row["label"]), str(i)))
        provenance[source] = {
            "dataset": name,
            "config": config,
            "split": split,
            "fingerprint": ds._fingerprint,
            "rows_in_split": len(ds),
            "sampled": len(chosen),
        }
    return pairs, provenance


def rates(sims: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = sims >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    return {
        "threshold": round(threshold, 3),
        "hit_rate": tp / max(tp + fn, 1),  # recall on true paraphrases
        "false_hit_rate": fp / max(fp + tn, 1),  # wrong answers served, per non-paraphrase
        "precision": tp / max(tp + fp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def embed_latency(model: Any, texts: list[str], repeats: int = 300) -> dict[str, float]:
    """Single-prompt latency, the way the gateway calls the model (one prompt per request)."""
    for text in texts[:20]:
        model.encode(text, normalize_embeddings=True)
    samples = []
    for i in range(repeats):
        started = time.perf_counter()
        model.encode(texts[i % len(texts)], normalize_embeddings=True)
        samples.append((time.perf_counter() - started) * 1000)
    arr = np.array(samples)
    return {
        "n": repeats,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(arr.mean()), 3),
    }


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    if sys.platform == "darwin":

        def sysctl(key: str) -> str:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=False
            ).stdout.strip()

        info["cpu"] = sysctl("machdep.cpu.brand_string")
        info["cores"] = sysctl("hw.ncpu")
        info["memory_gb"] = round(int(sysctl("hw.memsize") or 0) / 2**30, 1)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sample", type=int, default=2000, help="pairs per public dataset")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--max-false-hit-rate", type=float, default=0.01)
    parser.add_argument("--verifier", action="append", default=[], help="cross-encoder model(s)")
    parser.add_argument("--candidate-threshold", type=float, default=0.80)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "results" / "semantic-threshold" / stamp
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_prompts()
    public, provenance = load_public(args.sample, args.seed)
    pairs += public
    print(
        f"{len(pairs)} pairs: "
        + ", ".join(f"{s}={sum(p.source == s for p in pairs)}" for s in ("prompts", "paws", "qqp"))
    )

    model = SentenceTransformer(args.model, device="cpu")
    texts = sorted({t for p in pairs for t in (p.a, p.b)})
    started = time.perf_counter()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=64, convert_to_numpy=True)
    batch_seconds = time.perf_counter() - started
    index = {t: i for i, t in enumerate(texts)}
    sims = np.array([float(vectors[index[p.a]] @ vectors[index[p.b]]) for p in pairs])
    labels = np.array([p.label for p in pairs])
    sources = np.array([p.source for p in pairs])

    grid = [round(t, 3) for t in np.arange(0.50, 0.99, 0.01)] + [
        round(t, 3) for t in np.arange(0.99, 1.0001, 0.002)
    ]
    sweep: list[dict[str, Any]] = []
    for source in ("prompts", "paws", "qqp", "all"):
        mask = np.ones_like(labels, dtype=bool) if source == "all" else sources == source
        for t in grid:
            sweep.append({"source": source, **rates(sims[mask], labels[mask], t)})

    profiles = {"strict": ("prompts", "paws", "qqp"), "faq": ("qqp",)}

    def choose(rows: list[dict[str, Any]], sources: tuple[str, ...]) -> float | None:
        for t in grid:
            at_t = [r for r in rows if r["threshold"] == t and r["source"] in sources]
            if all(r["false_hit_rate"] <= args.max_false_hit_rate for r in at_t):
                return t
        return None

    def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for profile, members in profiles.items():
            t = choose(rows, members)
            out[profile] = {
                "sources": list(members),
                "chosen_threshold": t,
                "at_chosen": {
                    r["source"]: r for r in rows if r["threshold"] == t and r["source"] != "all"
                },
            }
        return out

    embedding_report = report(sweep)

    def by_category(accepted: np.ndarray) -> dict[str, dict[str, int]]:
        """Which kinds of hand-labelled hard negative get through a given configuration?"""
        out: dict[str, dict[str, int]] = {}
        for p, hit in zip(pairs, accepted, strict=True):
            if p.source != "prompts" or p.label == 1:
                continue
            bucket = out.setdefault(p.ref, {"negatives": 0, "false_hits": 0})
            bucket["negatives"] += 1
            bucket["false_hits"] += int(hit)
        return out

    breakdowns: dict[str, Any] = {}
    faq_t = embedding_report["faq"]["chosen_threshold"]
    if faq_t is not None:
        breakdowns[f"embedding>={faq_t}"] = by_category(sims >= faq_t)

    verifiers: dict[str, Any] = {}
    for name in args.verifier:
        from sentence_transformers import CrossEncoder

        encoder = CrossEncoder(name, device="cpu")
        scores = np.array(
            encoder.predict([(p.a, p.b) for p in pairs], batch_size=64, show_progress_bar=False)
        )
        candidate = sims >= args.candidate_threshold
        v_sweep = []
        for source in ("prompts", "paws", "qqp"):
            mask = sources == source
            for t in grid:
                gated = np.where(candidate[mask], scores[mask], -np.inf)
                v_sweep.append({"source": source, **rates(gated, labels[mask], t)})
        single = []
        for p in pairs[:200]:
            started = time.perf_counter()
            encoder.predict([(p.a, p.b)], show_progress_bar=False)
            single.append((time.perf_counter() - started) * 1000)
        v_report = report(v_sweep)
        strict_t = v_report["strict"]["chosen_threshold"]
        if strict_t is not None:
            breakdowns[
                f"{name}: embedding>={args.candidate_threshold} and verifier>={strict_t}"
            ] = by_category(candidate & (scores >= strict_t))
        verifiers[name] = {
            "candidate_threshold": args.candidate_threshold,
            "profiles": v_report,
            "single_pair_latency_ms": {
                "p50": round(float(np.percentile(single, 50)), 3),
                "p95": round(float(np.percentile(single, 95)), 3),
            },
        }
        slug = name.replace("/", "_")
        with (out / f"verifier_sweep_{slug}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(v_sweep[0]))
            writer.writeheader()
            writer.writerows(v_sweep)

    latency = embed_latency(model, texts)
    metrics = {
        "rule": f"lowest threshold with false_hit_rate <= {args.max_false_hit_rate} "
        "on every source in the profile",
        "embedding": {"model": args.model, "profiles": embedding_report},
        "two_stage": verifiers,
        "hand_labelled_false_hits_by_category": breakdowns,
        "embedding_latency_single_prompt": latency,
        "embedding_batch": {
            "texts": len(texts),
            "seconds": round(batch_seconds, 3),
            "texts_per_second": round(len(texts) / batch_seconds, 1),
        },
    }
    environment = {
        "generated_at": stamp,
        "command": " ".join(sys.argv),
        "model": args.model,
        "seed": args.seed,
        "sample_per_public_source": args.sample,
        "datasets": provenance,
        "versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "torch_threads": torch.get_num_threads(),
        "hardware": hardware(),
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    with (out / "sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    with (out / "scores.jsonl").open("w") as f:
        for p, s in zip(pairs, sims, strict=True):
            row = {"source": p.source, "ref": p.ref, "label": p.label, "similarity": round(s, 5)}
            if p.source == "prompts":  # our own data: safe to include the text
                row |= {"a": p.a, "b": p.b}
            f.write(json.dumps(row) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
