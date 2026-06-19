#!/usr/bin/env python3
"""Committed M5-style forecasting benchmark wrapper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed", action="store_true", help="Run the committed M5 sample.")
    parser.add_argument(
        "--official-wrmsse",
        action="store_true",
        help="Report WRMSSE-compatible scoring when M5 inputs include weights.",
    )
    parser.add_argument("--no-hyperopt", action="store_true")
    parser.add_argument("--output", default="artifacts/forecasting_m5_committed.json")
    args = parser.parse_args()
    if not args.no_hyperopt:
        parser.error("--no-hyperopt is required for benchmark integrity")
    if not args.official_wrmsse:
        parser.error("--official-wrmsse is required so M5 is not reduced to RMSE")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "forecasting_library_benchmark.py"),
        "--source",
        "m5",
        "--m5-series-limit",
        "96",
        "--m5-history-days",
        "365",
        "--model-roster",
        "cartoboost",
        "--output",
        args.output,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
