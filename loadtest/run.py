"""Run a load-test scenario and save everything needed to reproduce and audit it.

    python loadtest/run.py ramp --variant default
    python loadtest/run.py all            # every scenario and variant (~50 min)

Each run gets its own directory, ``results/loadtest/<UTC timestamp>__<scenario>__<variant>/``:

    run.json            what was run: scenario, variant, env, window, git SHA, command line
    environment.json    hardware, Colima VM, Docker, image IDs, k6/Python versions
    gateway.yaml        the exact gateway config the gateway was restarted with
    mock-settings.json  both mock providers' settings as reported by their admin API
    scripts/            copies of the k6 scripts used
    summary.json        k6's complete end-of-test summary (raw)
    k6.log              k6 stdout/stderr
    samples.csv.gz      every request (timeline scenarios only)
    prometheus.json     server-side metrics for the run window
    gateway.log.gz      gateway logs for the run window
    metrics.json        derived numbers (loadtest/analyze.py); what the README reports

Only the mock providers are ever targeted. The runner refuses to start if a scenario's target
is not a mock (hard rule: never load-test a real provider).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "loadtest"
K6_IMAGE = "grafana/k6:1.2.2"
NETWORK = "switchyard_default"
PROMETHEUS = "http://localhost:9090"
MOCK_ADMIN = {"mock-primary": "http://localhost:9001", "mock-secondary": "http://localhost:9002"}

# Mock behaviour for every run unless a scenario overrides it: ~50 ms to first token, 32 tokens
# at 5 ms each, so a non-streaming call takes ~210 ms upstream.
MOCK_DEFAULTS = {
    "ttft_ms": 50.0,
    "ttft_jitter_ms": 0.0,
    "inter_token_ms": 5.0,
    "output_tokens": 32,
    "error_rate": 0.0,
    "timeout_rate": 0.0,
    "stream_abort_rate": 0.0,
    "stream_stall_rate": 0.0,
    "hang_s": 600.0,
}

RAMP_STEPS = "50,100,200,300,400,500,600,800,1000"

SCENARIOS: dict[str, dict[str, Any]] = {
    # (a) max throughput / latency knee, non-streaming cache-bypass traffic
    "ramp": {
        "script": "ramp.js",
        "env": {"STEPS": RAMP_STEPS, "STEP_SECONDS": "30", "GAP_SECONDS": "10"},
        "variants": {
            "default": {},
            "no-shedding": {"gateway": {"overload": {"enabled": False}}},
            # The pre-ADR-019 connection pool: one httpx client per provider, no shedding.
            "single-httpx-pool": {
                "gateway": {
                    "overload": {"enabled": False},
                    "providers": {"mock-primary": {"pool_shards": 1}},
                }
            },
            "streaming": {"env": {"STREAM": "1"}},
            # Baseline: k6 straight at the mock. Gateway overhead = gateway minus this.
            "mock-direct": {"env": {"TARGET": "http://mock-primary:9000"}},
        },
    },
    # (b) steady load with a cache-hit mix
    "steady": {
        "script": "steady.js",
        "env": {"RATE": "200", "DURATION": "180s"},
        "variants": {
            # As shipped: exact cache only.
            "default": {},
            # Semantic tier on with its default 50 ms budget.
            "semantic": {"gateway": {"cache": {"semantic": {"enabled": True}}}},
            # Semantic tier allowed up to 250 ms, as in front of multi-second LLM calls.
            "semantic-budget-250ms": {
                "gateway": {"cache": {"semantic": {"enabled": True, "lookup_budget_ms": 250}}}
            },
        },
        "flush_cache": True,
    },
    # (c) chaos: the primary starts failing at 40 s and recovers at 80 s
    "chaos": {
        "script": "chaos.js",
        "env": {"RATE": "150", "DURATION_SECONDS": "120", "FAULT_AT": "40", "RECOVER_AT": "80"},
        "variants": {
            "errors": {"env": {"FAULT": "errors"}},
            "hang": {"env": {"FAULT": "hang"}},
            "drop": {"env": {"FAULT": "drop"}},
        },
        "csv": True,
    },
    # (d) rate-limit burst against one key
    "burst": {
        "script": "burst.js",
        "env": {"BURST": "2000", "BURST_VUS": "200", "SUSTAIN_RATE": "50", "SUSTAIN_SECONDS": "60"},
        "variants": {"default": {}},
        "key": {"rpm": 300, "tpm": 100_000_000},
        "csv": True,
    },
}


def sh(*args: str, check: bool = True, timeout: float = 600) -> str:
    out = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if check and out.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def environment() -> dict[str, Any]:
    def sysctl(key: str) -> str:
        return sh("sysctl", "-n", key, check=False).strip()

    docker = json.loads(sh("docker", "info", "--format", "{{json .}}"))
    images = {
        name: sh("docker", "image", "inspect", name, "--format", "{{.Id}}").strip()
        for name in ("switchyard-gateway:dev", "switchyard-mock:dev", "redis:8.2-alpine")
    }
    colima = sh("colima", "list", "--json", check=False).strip()
    return {
        "host": {
            "cpu": sysctl("machdep.cpu.brand_string"),
            "cores": sysctl("hw.ncpu"),
            "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30, 1),
            "os": platform.platform(),
        },
        "vm": json.loads(colima.splitlines()[0]) if colima else None,
        "docker": {
            "server_version": docker.get("ServerVersion"),
            "ncpu": docker.get("NCPU"),
            "mem_total_gb": round(docker.get("MemTotal", 0) / 2**30, 1),
            "kernel": docker.get("KernelVersion"),
        },
        "images": images,
        "k6_image": K6_IMAGE,
        "k6_version": sh("docker", "run", "--rm", K6_IMAGE, "version").strip(),
        "k6_cpus": 2,
        "notes": (
            "Gateway, mocks, Redis, Prometheus and k6 share the same VM. The gateway is one "
            "uvicorn process (one core); each mock is one process; k6 is limited to 2 CPUs."
        ),
    }


def git_state() -> dict[str, Any]:
    return {
        "sha": sh("git", "-C", str(ROOT), "rev-parse", "HEAD").strip(),
        "dirty": bool(sh("git", "-C", str(ROOT), "status", "--porcelain").strip()),
    }


def restart_gateway(run_dir: Path) -> None:
    override = run_dir / "compose.override.yml"
    override.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "gateway": {
                        "environment": {"SWITCHYARD_CONFIG": "/run-config/gateway.yaml"},
                        "volumes": [f"{run_dir}:/run-config:ro"],
                    }
                }
            }
        )
    )
    sh(
        "docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "-f", str(override),
        "up", "-d", "--wait", "--force-recreate", "--no-deps", "gateway",
    )  # fmt: skip
    time.sleep(6)  # at least one Prometheus scrape of the fresh process


def configure_mocks(settings: dict[str, Any]) -> dict[str, Any]:
    reported = {}
    for name, url in MOCK_ADMIN.items():
        httpx.post(f"{url}/admin/reset", timeout=10).raise_for_status()
        resp = httpx.patch(f"{url}/admin/config", json=settings, timeout=10)
        resp.raise_for_status()
        reported[name] = resp.json()
    return reported


def flush_cache() -> None:
    sh(
        "docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "exec", "-T", "redis",
        "sh", "-c", "redis-cli --scan --pattern 'sy:cache:*' | xargs -r redis-cli del >/dev/null",
    )  # fmt: skip


def create_key(name: str, rpm: int, tpm: int) -> str:
    out = sh(
        "docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "exec", "-T", "gateway",
        "python", "-m", "switchyard.cli", "keys", "create",
        "--name", name, "--rpm", str(rpm), "--tpm", str(tpm),
    )  # fmt: skip
    key: str = json.loads(out)["api_key"]
    return key


def run_one(
    scenario_name: str,
    variant_name: str,
    *,
    env_overrides: dict[str, str] | None = None,
    results: Path = RESULTS,
) -> Path:
    scenario = SCENARIOS[scenario_name]
    variant = scenario["variants"][variant_name]
    env = {**scenario["env"], **variant.get("env", {}), **(env_overrides or {})}
    target = env.get("TARGET", "http://gateway:8000")
    if target not in ("http://gateway:8000", "http://mock-primary:9000"):
        raise SystemExit(f"refusing to load-test {target}: only the gateway and mocks allowed")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results / f"{stamp}__{scenario_name}__{variant_name}"
    run_dir.mkdir(parents=True)
    print(f"==> {scenario_name}/{variant_name} -> {run_dir}", flush=True)

    base = yaml.safe_load((ROOT / "config" / "gateway.yaml").read_text())
    gateway_config = deep_merge(base, variant.get("gateway", {}))
    (run_dir / "gateway.yaml").write_text(yaml.safe_dump(gateway_config, sort_keys=False))
    restart_gateway(run_dir)

    mock_settings = deep_merge(MOCK_DEFAULTS, scenario.get("mock", {}))
    (run_dir / "mock-settings.json").write_text(
        json.dumps(configure_mocks(mock_settings), indent=2) + "\n"
    )
    if scenario.get("flush_cache"):
        flush_cache()
    key_limits = scenario.get("key", {"rpm": 100_000_000, "tpm": 10**12})
    env["API_KEY"] = create_key(f"loadtest-{scenario_name}", **key_limits)
    env.setdefault("TARGET", target)

    scripts = run_dir / "scripts"
    scripts.mkdir()
    for f in ("lib.js", scenario["script"]):
        shutil.copy(ROOT / "loadtest" / f, scripts / f)

    k6 = [
        "docker", "run", "--rm", "--network", NETWORK, "--cpus", "2",
        "-v", f"{ROOT / 'loadtest'}:/scripts:ro", "-v", f"{run_dir}:/out",
        "-e", "K6_CSV_TIME_FORMAT=unix_milli",
    ]  # fmt: skip
    for k, v in env.items():
        k6 += ["-e", f"{k}={v}"]
    k6 += [K6_IMAGE, "run", "--quiet"]
    if scenario.get("csv"):
        k6 += ["--out", "csv=/out/samples.csv.gz"]
    k6 += [f"/scripts/{scenario['script']}"]

    started = time.time()
    proc = subprocess.run(k6, capture_output=True, text=True, check=False)
    finished = time.time()
    (run_dir / "k6.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode not in (0, 99):  # 99 = thresholds crossed; ours never fail
        raise RuntimeError(f"k6 failed ({proc.returncode}); see {run_dir / 'k6.log'}")

    time.sleep(6)  # final Prometheus scrape
    (run_dir / "prometheus.json").write_text(
        json.dumps(analyze.fetch_prometheus(PROMETHEUS, started, finished), indent=1) + "\n"
    )
    logs = sh(
        "docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "logs", "--no-log-prefix",
        "--since", datetime.fromtimestamp(started - 5, UTC).isoformat(), "gateway",
    )  # fmt: skip
    with gzip.open(run_dir / "gateway.log.gz", "wt") as f:
        f.write(logs)

    run_info = {
        "scenario": scenario_name,
        "variant": variant_name,
        "script": scenario["script"],
        "env": {k: v for k, v in env.items() if k != "API_KEY"},
        "key_limits": key_limits,
        "window": {"start": started, "end": finished},
        "git": git_state(),
        "command": " ".join(sys.argv),
        "k6_command": " ".join(a if "API_KEY=" not in a else "API_KEY=<redacted>" for a in k6),
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    (run_dir / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    (run_dir / "compose.override.yml").unlink()

    metrics = analyze.analyze(run_dir)
    print(json.dumps(metrics.get("headline", {}), indent=2), flush=True)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario", choices=[*SCENARIOS, "all"])
    parser.add_argument("--variant", help="default: every variant of the scenario")
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE",
        help="override a scenario env var (for smoke-testing the harness)",
    )  # fmt: skip
    parser.add_argument(
        "--out", type=Path, default=RESULTS,
        help="results root (use a scratch dir for smoke tests; results/ is for real runs)",
    )  # fmt: skip
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.env)
    try:
        httpx.get("http://localhost:8000/readyz", timeout=3).raise_for_status()
    except httpx.HTTPError:
        print("stack is not running: `make up` first", file=sys.stderr)
        return 1

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        variants = [args.variant] if args.variant else list(SCENARIOS[name]["variants"])
        for variant in variants:
            run_one(name, variant, env_overrides=overrides, results=args.out.resolve())
    # Leave the stack on its normal config.
    sh("docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "up", "-d", "--wait",
       "--force-recreate", "--no-deps", "gateway")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
