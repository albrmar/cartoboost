#!/usr/bin/env python3
"""Committed M4-style forecasting benchmark wrapper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed", action="store_true", help="Run the committed M4 sample.")
    parser.add_argument(
        "--no-hyperopt",
        action="store_true",
        help="Required marker: CartoBoost forecasting benchmarks use fixed model menus.",
    )
    parser.add_argument("--output", default="artifacts/forecasting_m4_committed.json")
    args = parser.parse_args()
    if not args.no_hyperopt:
        parser.error("--no-hyperopt is required for benchmark integrity")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "forecasting_library_benchmark.py"),
        "--source",
        "m4",
        "--m4-suite",
        "--m4-series-limit",
        "96",
        "--model-roster",
        "cartoboost",
        "--output",
        args.output,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
