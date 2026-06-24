#!/usr/bin/env python3
"""Non-M forecasting generalization benchmark wrapper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Run the maintained compact non-M scalable external-roster check.",
    )
    parser.add_argument(
        "--no-hyperopt",
        action="store_true",
        help="Required marker: CartoBoost forecasting benchmarks use fixed model menus.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/forecasting_generalization_scalable_synthetic.json",
    )
    args = parser.parse_args()
    if not args.no_hyperopt:
        parser.error("--no-hyperopt is required for benchmark integrity")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "forecasting_library_benchmark.py"),
        "--suite",
        "synthetic",
        "--no-hyperopt",
        "--model-roster",
        "scalable",
        "--lanes",
        "18",
        "--days",
        "150",
        "--horizon",
        "14",
        "--suite-folds",
        "2",
        "--seed",
        "177",
        "--cartoboost-n-estimators",
        "60",
        "--cartoboost-auto-n-estimators",
        "60",
        "--no-candidate-selection",
        "--output",
        args.output,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
