"""Capture the provisioned Grafana dashboard over a load-test run's time window.

    pip install -e '.[tools]' && playwright install chromium
    python scripts/grafana_screenshot.py results/loadtest/<run>

Writes ``results/grafana/<run>/dashboard.png`` and ``info.json``. Grafana serves anonymous
read-only access in the local stack, so no credentials are needed. Prometheus keeps two days of
data, so capture soon after the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    "chaos": "the primary mock is broken mid-run, so traffic fails over and the breaker opens",
    "ramp": "offered load stepped up until the gateway saturates",
    "steady": "steady load with a cache-hit mix",
    "burst": "a rate-limit burst against one key",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--grafana", default="http://localhost:3000")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    run_dir = args.run_dir.resolve()
    run = json.loads((run_dir / "run.json").read_text())
    start_ms = int((run["window"]["start"] - 15) * 1000)
    end_ms = int((run["window"]["end"] + 15) * 1000)
    url = f"{args.grafana}/d/switchyard?orgId=1&from={start_ms}&to={end_ms}&kiosk&theme=light"

    out = ROOT / "results" / "grafana" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Grafana scrolls inside its own container and only renders panels in view, so a
        # full-page capture comes out blank: use a viewport tall enough for every panel.
        page = browser.new_page(viewport={"width": 1800, "height": 2100}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(6000)  # panels render after their queries return
        page.screenshot(path=str(out / "dashboard.png"))
        browser.close()
    (out / "info.json").write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "scenario": run["scenario"],
                "variant": run["variant"],
                "description": DESCRIPTIONS.get(run["scenario"], run["scenario"]),
                "url": url,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {(out / 'dashboard.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
