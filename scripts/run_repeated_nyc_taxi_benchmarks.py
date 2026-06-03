#!/usr/bin/env python3
"""Run repeated NYC taxi benchmarks and summarize GeoBoost speed ratios."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "scripts" / "run_nyc_taxi_quality_benchmarks.py"
DEFAULT_RUN_DIR = ROOT / "target" / "nyc_taxi_repeated"
DEFAULT_SUMMARY_JSON = ROOT / "docs" / "assets" / "nyc_taxi_benchmarks" / "repeated_results.json"
DEFAULT_SUMMARY_MD = ROOT / "docs" / "assets" / "nyc_taxi_benchmarks" / "repeated_results.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1")
    parser.add_argument("--sample-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="geoboost,lightgbm,xgboost,mean")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def benchmark_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--models",
        args.models,
        "--n-estimators",
        str(args.n_estimators),
        "--learning-rate",
        str(args.learning_rate),
        "--max-depth",
        str(args.max_depth),
        "--n-threads",
        str(args.n_threads),
        "--seed",
        str(args.seed),
    ]
    if args.synthetic_smoke:
        command.append("--synthetic-smoke")
    else:
        command.extend(
            [
                "--year",
                str(args.year),
                "--months",
                args.months,
                "--sample-size",
                str(args.sample_size),
            ]
        )
    if args.no_download:
        command.append("--no-download")
    return command


def run_once(args: argparse.Namespace, run_index: int) -> dict[str, Any]:
    output_dir = args.run_dir / f"run_{run_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(benchmark_command(args, output_dir), cwd=ROOT, check=True)
    return json.loads((output_dir / "results.json").read_text(encoding="utf-8"))


def collect_ratios(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, float]]] = {}
    for run_index, result in enumerate(results, start=1):
        for task_name, task in result["tasks"].items():
            for split_name, split in task["splits"].items():
                models = split["models"]
                geoboost = models.get("geoboost")
                if geoboost is None or geoboost["status"] != "ok":
                    continue
                baselines = [
                    models[name]
                    for name in ("lightgbm", "xgboost")
                    if name in models and models[name]["status"] == "ok"
                ]
                if not baselines:
                    continue
                geo_timing = geoboost["timing"]
                fastest_train = min(float(model["timing"]["train_seconds"]) for model in baselines)
                fastest_predict_rps = max(
                    float(model["timing"]["predict_rows_per_second"]) for model in baselines
                )
                key = f"{task_name}/{split_name}"
                rows.setdefault(key, []).append(
                    {
                        "run": float(run_index),
                        "geoboost_train_seconds": float(geo_timing["train_seconds"]),
                        "geoboost_predict_rows_per_second": float(
                            geo_timing["predict_rows_per_second"]
                        ),
                        "train_ratio_vs_fastest": float(geo_timing["train_seconds"])
                        / fastest_train,
                        "predict_rps_ratio_vs_fastest": float(
                            geo_timing["predict_rows_per_second"]
                        )
                        / fastest_predict_rps,
                    }
                )
    return {
        key: {
            "runs": values,
            "median_train_ratio_vs_fastest": statistics.median(
                item["train_ratio_vs_fastest"] for item in values
            ),
            "min_train_ratio_vs_fastest": min(item["train_ratio_vs_fastest"] for item in values),
            "max_train_ratio_vs_fastest": max(item["train_ratio_vs_fastest"] for item in values),
            "median_predict_rps_ratio_vs_fastest": statistics.median(
                item["predict_rps_ratio_vs_fastest"] for item in values
            ),
            "min_predict_rps_ratio_vs_fastest": min(
                item["predict_rps_ratio_vs_fastest"] for item in values
            ),
            "max_predict_rps_ratio_vs_fastest": max(
                item["predict_rps_ratio_vs_fastest"] for item in values
            ),
        }
        for key, values in sorted(rows.items())
    }


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Repeated NYC Taxi Speed Benchmark",
        "",
        "| task/split | train ratio median | train ratio min-max | "
        "predict rps ratio median | predict rps ratio min-max | gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, row in summary["ratios"].items():
        train_median = row["median_train_ratio_vs_fastest"]
        predict_median = row["median_predict_rps_ratio_vs_fastest"]
        gate = "pass" if train_median <= 5.0 and predict_median >= 0.2 else "miss"
        lines.append(
            f"| {key} | {train_median:.2f}x | "
            f"{row['min_train_ratio_vs_fastest']:.2f}x-{row['max_train_ratio_vs_fastest']:.2f}x | "
            f"{predict_median:.3f}x | "
            f"{row['min_predict_rps_ratio_vs_fastest']:.3f}x-"
            f"{row['max_predict_rps_ratio_vs_fastest']:.3f}x | {gate} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    results = [run_once(args, run_index) for run_index in range(1, args.runs + 1)]
    ratios = collect_ratios(results)
    summary = {
        "artifact_version": 1,
        "runs": args.runs,
        "target": {
            "train_ratio_vs_fastest_max": 5.0,
            "predict_rps_ratio_vs_fastest_min": 0.2,
        },
        "ratios": ratios,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.summary_md)


if __name__ == "__main__":
    main()
