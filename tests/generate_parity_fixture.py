#!/usr/bin/env python3
"""Generate CartoBoost parity fixtures from a deterministic regression dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import CartoBoostRegressor  # noqa: E402


def build_dataset() -> tuple[list[list[float]], list[float]]:
    rows = [
        [0.10, 0.05, 0.90, 0.88, 10.0],
        [0.15, 0.07, 0.85, 0.80, 50.0],
        [0.20, 0.10, 0.80, 0.72, 120.0],
        [0.75, 0.70, 0.25, 0.20, 220.0],
        [0.80, 0.74, 0.20, 0.15, 300.0],
        [0.85, 0.79, 0.15, 0.12, 340.0],
    ]
    target = [140.0, 145.0, 155.0, 250.0, 265.0, 275.0]
    return rows, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures" / "parity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, target = build_dataset()
    model = CartoBoostRegressor(
        n_estimators=5,
        learning_rate=0.25,
        max_depth=2,
        min_samples_leaf=1,
    )
    model.fit(rows, target)

    model_path = args.output_dir / "baseline_model.cartoboost"
    model.save(model_path)
    fixture = {
        "model_path": model_path.name,
        "rows": rows,
        "target": target,
        "expected_predictions": model.predict(rows).tolist(),
    }
    (args.output_dir / "parity_fixture.json").write_text(
        json.dumps(fixture, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
