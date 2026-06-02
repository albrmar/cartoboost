#!/usr/bin/env python3
"""Create a compact validation report from prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV with target and prediction columns")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--prediction-col", default="prediction")
    parser.add_argument(
        "--task",
        choices=("regression", "binary"),
        default="regression",
        help="Metric family to report",
    )
    parser.add_argument("--output", type=Path, help="Optional Markdown report path")
    parser.add_argument("--json-output", type=Path, help="Optional JSON metrics path")
    return parser.parse_args()


def read_pairs(path: Path, target_col: str, prediction_col: str) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {target_col, prediction_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        return [
            (float(row[target_col]), float(row[prediction_col]))
            for row in reader
            if row[target_col] != "" and row[prediction_col] != ""
        ]


def regression_metrics(pairs: list[tuple[float, float]]) -> dict[str, float]:
    errors = [prediction - target for target, prediction in pairs]
    abs_errors = [abs(error) for error in errors]
    sq_errors = [error * error for error in errors]
    target_mean = mean(target for target, _ in pairs)
    total_sq = sum((target - target_mean) ** 2 for target, _ in pairs)
    residual_sq = sum(sq_errors)
    r2 = 1.0 - residual_sq / total_sq if total_sq else 0.0
    return {
        "rows": float(len(pairs)),
        "mae": mean(abs_errors),
        "rmse": math.sqrt(mean(sq_errors)),
        "bias": mean(errors),
        "r2": r2,
    }


def binary_metrics(pairs: list[tuple[float, float]]) -> dict[str, float]:
    eps = 1e-15
    predictions = [(target, min(1.0 - eps, max(eps, prediction))) for target, prediction in pairs]
    labels = [(target, 1.0 if prediction >= 0.5 else 0.0) for target, prediction in predictions]
    accuracy = mean(1.0 if target == label else 0.0 for target, label in labels)
    logloss = mean(
        -(target * math.log(prediction) + (1.0 - target) * math.log(1.0 - prediction))
        for target, prediction in predictions
    )
    return {
        "rows": float(len(pairs)),
        "accuracy": accuracy,
        "logloss": logloss,
    }


def render_markdown(path: Path, task: str, metrics: dict[str, float]) -> str:
    lines = [
        "# GeoBoost validation report",
        "",
        f"- source: `{path}`",
        f"- task: `{task}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        if key == "rows":
            lines.append(f"| {key} | {int(value)} |")
        else:
            lines.append(f"| {key} | {value:.8f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.predictions, args.target_col, args.prediction_col)
    if not pairs:
        raise ValueError("no valid prediction rows found")

    metrics = regression_metrics(pairs) if args.task == "regression" else binary_metrics(pairs)
    markdown = render_markdown(args.predictions, args.task, metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
