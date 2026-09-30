"""Generate the Grafana dashboard JSON (dashboards as code).

Editing a 1,500-line JSON file by hand is how dashboards rot. The dashboard is defined here in
a few lines per panel, and ``make dashboard`` writes
``config/grafana/dashboards/switchyard.json``, which Grafana provisions at startup. CI checks
the committed JSON is up to date.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "grafana" / "dashboards" / "switchyard.json"
DS = {"type": "prometheus", "uid": "prometheus"}
RATE = "$__rate_interval"

_panels: list[dict[str, Any]] = []
_y = 0
_x = 0
_row_height = 0


def _place(width: int, height: int) -> dict[str, int]:
    global _x, _y, _row_height
    if _x + width > 24:
        _x, _y = 0, _y + _row_height
        _row_height = 0
    pos = {"x": _x, "y": _y, "w": width, "h": height}
    _x += width
    _row_height = max(_row_height, height)
    return pos


def row(title: str) -> None:
    global _x, _y, _row_height
    _x, _y, _row_height = 0, _y + _row_height, 0
    _panels.append(
        {"type": "row", "title": title, "collapsed": False, "gridPos": _place(24, 1), "panels": []}
    )
    _x, _y, _row_height = 0, _y + 1, 0


def target(expr: str, legend: str, ref: str) -> dict[str, Any]:
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}


def panel(
    title: str,
    queries: list[tuple[str, str]],
    *,
    unit: str = "short",
    width: int = 8,
    height: int = 8,
    kind: str = "timeseries",
    description: str = "",
    stack: bool = False,
    mappings: list[dict[str, Any]] | None = None,
) -> None:
    field_defaults: dict[str, Any] = {"unit": unit}
    if unit == "percentunit":
        field_defaults |= {"min": 0, "max": 1}  # ratios: a fixed 0-100% axis
    if kind == "timeseries":
        field_defaults["custom"] = {
            "drawStyle": "line",
            "lineWidth": 1,
            "fillOpacity": 15 if stack else 5,
            "showPoints": "never",
            "stacking": {"mode": "normal" if stack else "none"},
        }
    if mappings:
        field_defaults["mappings"] = mappings
    _panels.append(
        {
            "type": kind,
            "title": title,
            "description": description,
            "datasource": DS,
            "gridPos": _place(width, height),
            "targets": [
                target(expr, legend, chr(ord("A") + i)) for i, (expr, legend) in enumerate(queries)
            ],
            "fieldConfig": {"defaults": field_defaults, "overrides": []},
            "options": {
                "legend": {"displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "multi"},
            },
        }
    )


def quantiles(metric: str, selector: str = "", by: str = "") -> list[tuple[str, str]]:
    group = f"le{', ' + by if by else ''}"
    suffix = f" {{{{{by}}}}}" if by else ""
    return [
        (
            f"histogram_quantile({q}, sum by ({group}) "
            f"(rate({metric}_bucket{{{selector}}}[{RATE}])))",
            f"p{int(q * 100)}{suffix}",
        )
        for q in (0.5, 0.95, 0.99)
    ]


def build() -> dict[str, Any]:
    row("Traffic")
    panel(
        "Requests / s by status",
        [(f"sum by (status) (rate(switchyard_requests_total[{RATE}]))", "{{status}}")],
        unit="reqps",
        stack=True,
    )
    panel(
        "Error ratio (5xx)",
        [
            (
                f'(sum(rate(switchyard_requests_total{{status=~"5.."}}[{RATE}])) or vector(0)) '
                f"/ clamp_min(sum(rate(switchyard_requests_total[{RATE}])), 1e-9)",
                "5xx / all",
            )
        ],
        unit="percentunit",
    )
    panel("In-flight requests", [("sum(switchyard_requests_in_flight)", "in flight")])

    row("Latency")
    panel(
        "Request latency (non-streaming)",
        quantiles("switchyard_request_duration_seconds", 'stream="false"'),
        unit="s",
    )
    panel(
        "Time to first token (streaming)",
        quantiles("switchyard_time_to_first_token_seconds"),
        unit="s",
    )
    panel(
        "Gateway overhead",
        quantiles("switchyard_gateway_overhead_seconds", by="stream"),
        unit="s",
        description="Total latency (TTFB for streams) minus upstream time: what the gateway adds.",
    )
    panel(
        "Upstream latency p95 by provider",
        [
            (
                f"histogram_quantile(0.95, sum by (le, provider, stream) "
                f"(rate(switchyard_upstream_duration_seconds_bucket[{RATE}])))",
                "{{provider}} stream={{stream}}",
            )
        ],
        unit="s",
        width=12,
    )
    panel(
        "Tokens / s by provider",
        [
            (
                f"sum by (provider, type) (rate(switchyard_tokens_total[{RATE}]))",
                "{{provider}} {{type}}",
            )
        ],
        width=12,
    )

    row("Cache")
    panel(
        "Hit ratio by tier",
        [
            (
                f'sum by (tier) (rate(switchyard_cache_lookups_total{{result="hit"}}[{RATE}])) '
                f"/ clamp_min(sum by (tier) (rate(switchyard_cache_lookups_total[{RATE}])), 1e-9)",
                "{{tier}}",
            )
        ],
        unit="percentunit",
    )
    panel(
        "Requests by cache outcome",
        [(f"sum by (cache) (rate(switchyard_requests_total[{RATE}]))", "{{cache}}")],
        unit="reqps",
        stack=True,
    )
    panel(
        "Semantic cache: model latency p95 and verifier vetoes",
        [
            (
                f"histogram_quantile(0.95, sum by (le, model) "
                f"(rate(switchyard_cache_model_seconds_bucket[{RATE}])))",
                "{{model}} p95 (s)",
            ),
            (f"rate(switchyard_cache_semantic_rejections_total[{RATE}])", "vetoes / s"),
        ],
    )

    row("Reliability")
    panel(
        "Circuit breaker state",
        [("max by (provider) (switchyard_circuit_state)", "{{provider}}")],
        kind="state-timeline",
        width=12,
        mappings=[
            {
                "type": "value",
                "options": {
                    "0": {"text": "closed", "color": "green"},
                    "1": {"text": "half-open", "color": "yellow"},
                    "2": {"text": "open", "color": "red"},
                },
            }
        ],
    )
    panel(
        "Failovers / s",
        [
            (
                f"sum by (from_provider, to_provider) (rate(switchyard_failovers_total[{RATE}]))",
                "{{from_provider}} → {{to_provider}}",
            )
        ],
        width=12,
    )
    panel(
        "Upstream errors / s by provider and kind",
        [
            (
                f"sum by (provider, kind) (rate(switchyard_upstream_errors_total[{RATE}]))",
                "{{provider}} {{kind}}",
            )
        ],
        width=12,
        stack=True,
    )
    panel(
        "Retries and mid-stream failures / s",
        [
            (f"sum by (provider) (rate(switchyard_retries_total[{RATE}]))", "retries {{provider}}"),
            (
                f"sum by (provider) (rate(switchyard_mid_stream_failures_total[{RATE}]))",
                "mid-stream {{provider}}",
            ),
        ],
        width=12,
    )

    row("Rate limiting and auth")
    panel(
        "429 rejections / s by bucket",
        [(f"sum by (limit) (rate(switchyard_rate_limited_total[{RATE}]))", "{{limit}}")],
        unit="reqps",
        width=12,
    )
    panel(
        "401 rejections / s by reason",
        [(f"sum by (reason) (rate(switchyard_auth_failures_total[{RATE}]))", "{{reason}}")],
        unit="reqps",
        width=12,
    )

    return {
        "uid": "switchyard",
        "title": "Switchyard gateway",
        "tags": ["switchyard", "llm-gateway"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": _panels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build(), indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != rendered:
            print(f"{OUT.relative_to(ROOT)} is out of date: run `make dashboard`", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(rendered)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(_panels)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
