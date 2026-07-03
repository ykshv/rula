from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="evals/results/soak_report.json")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"missing {path}; soak gate fails closed", file=sys.stderr)
        return 1

    report = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "duration_minutes": report.get("duration_minutes", 0) >= 60,
        "turns": report.get("turns", 0) >= 120,
        "unrecovered_errors": report.get("unrecovered_errors", 1) == 0,
        "stale_generation_rendered": not report.get("stale_generation_rendered", True),
        "latency_regression_ratio": report.get("latency_regression_ratio", 99) <= 1.10,
        "thermal_throttling_sustained": not report.get("thermal_throttling_sustained", True),
    }
    print(json.dumps({"checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
