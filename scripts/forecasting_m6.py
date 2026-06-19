#!/usr/bin/env python3
"""Committed M6-style forecasting benchmark wrapper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed", action="store_true", help="Run the committed M6 sample.")
    parser.add_argument(
        "--official-style",
        action="store_true",
        help="Use rank-probability/portfolio-style reporting when available.",
    )
    parser.add_argument("--no-hyperopt", action="store_true")
    parser.add_argument("--output", default="artifacts/forecasting_m6_committed.json")
    args = parser.parse_args()
    if not args.no_hyperopt:
        parser.error("--no-hyperopt is required for benchmark integrity")
    if not args.official_style:
        parser.error("--official-style is required so M6 is not reported as RMSE-only")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "forecasting_library_benchmark.py"),
        "--source",
        "m6",
        "--m6-series-limit",
        "96",
        "--model-roster",
        "cartoboost",
        "--no-hyperopt",
        "--output",
        args.output,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
