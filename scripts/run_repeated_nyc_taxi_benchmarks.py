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
    parser.add_argument("--tasks", default="")
    parser.add_argument("--models", default="geoboost,geoboost_reference,lightgbm,xgboost,mean")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--geoboost-n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--geoboost-max-depth", type=int, default=4)
    parser.add_argument("--geoboost-splitters", default="axis_histogram:64")
    parser.add_argument(
        "--xgboost-tree-method",
        default="hist",
        choices=["auto", "exact", "approx", "hist"],
    )
    parser.add_argument("--xgboost-subsample", type=float, default=1.0)
    parser.add_argument("--xgboost-colsample-bytree", type=float, default=1.0)
    parser.add_argument("--zone-treatment", default="target_mean", choices=["raw", "target_mean"])
    parser.add_argument("--zone-target-smoothing", type=float, default=20.0)
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
        "--geoboost-n-estimators",
        str(args.geoboost_n_estimators),
        "--learning-rate",
        str(args.learning_rate),
        "--max-depth",
        str(args.max_depth),
        "--geoboost-max-depth",
        str(args.geoboost_max_depth),
        "--geoboost-splitters",
        args.geoboost_splitters,
        "--xgboost-tree-method",
        args.xgboost_tree_method,
        "--xgboost-subsample",
        str(args.xgboost_subsample),
        "--xgboost-colsample-bytree",
        str(args.xgboost_colsample_bytree),
        "--zone-treatment",
        args.zone_treatment,
        "--zone-target-smoothing",
        str(args.zone_target_smoothing),
        "--n-threads",
        str(args.n_threads),
        "--seed",
        str(args.seed),
    ]
    if args.tasks:
        command.extend(["--tasks", args.tasks])
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
                xgboost = models.get("xgboost")
                if xgboost is None or xgboost["status"] != "ok":
                    continue
                geoboost_reference = models.get("geoboost_reference")
                geo_timing = geoboost["timing"]
                xgb_timing = xgboost["timing"]
                geo_metrics = geoboost["metrics"]
                xgb_metrics = xgboost["metrics"]
                reference_metrics = (
                    geoboost_reference["metrics"]
                    if geoboost_reference is not None
                    and geoboost_reference["status"] == "ok"
                    else None
                )
                key = f"{task_name}/{split_name}"
                row = {
                    "run": float(run_index),
                    "geoboost_train_seconds": float(geo_timing["train_seconds"]),
                    "geoboost_predict_rows_per_second": float(
                        geo_timing["predict_rows_per_second"]
                    ),
                    "xgboost_train_seconds": float(xgb_timing["train_seconds"]),
                    "xgboost_predict_rows_per_second": float(
                        xgb_timing["predict_rows_per_second"]
                    ),
                    "train_ratio_vs_xgboost": float(geo_timing["train_seconds"])
                    / float(xgb_timing["train_seconds"]),
                    "predict_rps_ratio_vs_xgboost": float(
                        geo_timing["predict_rows_per_second"]
                    )
                    / float(xgb_timing["predict_rows_per_second"]),
                    "rmse_delta_vs_xgboost": float(geo_metrics["rmse"])
                    - float(xgb_metrics["rmse"]),
                    "r2_delta_vs_xgboost": float(geo_metrics["r2"]) - float(xgb_metrics["r2"]),
                }
                if reference_metrics is not None:
                    row["rmse_delta_vs_geoboost_reference"] = float(geo_metrics["rmse"]) - float(
                        reference_metrics["rmse"]
                    )
                    row["r2_delta_vs_geoboost_reference"] = float(geo_metrics["r2"]) - float(
                        reference_metrics["r2"]
                    )
                rows.setdefault(key, []).append(row)
    return {
        key: {
            "runs": values,
            "median_train_ratio_vs_xgboost": statistics.median(
                item["train_ratio_vs_xgboost"] for item in values
            ),
            "min_train_ratio_vs_xgboost": min(item["train_ratio_vs_xgboost"] for item in values),
            "max_train_ratio_vs_xgboost": max(item["train_ratio_vs_xgboost"] for item in values),
            "median_predict_rps_ratio_vs_xgboost": statistics.median(
                item["predict_rps_ratio_vs_xgboost"] for item in values
            ),
            "min_predict_rps_ratio_vs_xgboost": min(
                item["predict_rps_ratio_vs_xgboost"] for item in values
            ),
            "max_predict_rps_ratio_vs_xgboost": max(
                item["predict_rps_ratio_vs_xgboost"] for item in values
            ),
            "median_rmse_delta_vs_xgboost": statistics.median(
                item["rmse_delta_vs_xgboost"] for item in values
            ),
            "median_r2_delta_vs_xgboost": statistics.median(
                item["r2_delta_vs_xgboost"] for item in values
            ),
            **reference_delta_summary(values),
        }
        for key, values in sorted(rows.items())
    }


def reference_delta_summary(values: list[dict[str, float]]) -> dict[str, float | None]:
    rmse = [
        item["rmse_delta_vs_geoboost_reference"]
        for item in values
        if "rmse_delta_vs_geoboost_reference" in item
    ]
    r2 = [
        item["r2_delta_vs_geoboost_reference"]
        for item in values
        if "r2_delta_vs_geoboost_reference" in item
    ]
    return {
        "median_rmse_delta_vs_geoboost_reference": statistics.median(rmse) if rmse else None,
        "median_r2_delta_vs_geoboost_reference": statistics.median(r2) if r2 else None,
    }


def format_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def same_quality(row: dict[str, Any], tolerance: float = 1e-9) -> bool:
    rmse_ref = row["median_rmse_delta_vs_geoboost_reference"]
    r2_ref = row["median_r2_delta_vs_geoboost_reference"]
    if rmse_ref is None or r2_ref is None:
        return False
    return abs(rmse_ref) <= tolerance and abs(r2_ref) <= tolerance


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Repeated NYC Taxi Speed Benchmark",
        "",
        (
            f"- baseline estimators: {summary['model_config']['baseline_n_estimators']}; "
            f"GeoBoost candidate estimators: {summary['model_config']['geoboost_n_estimators']}"
        ),
        (
            f"- baseline max depth: {summary['model_config']['baseline_max_depth']}; "
            f"GeoBoost candidate max depth: {summary['model_config']['geoboost_max_depth']}"
        ),
        (
            f"- GeoBoost splitters: {summary['model_config']['geoboost_splitters']}; "
            f"XGBoost tree_method: {summary['model_config']['xgboost_tree_method']}"
        ),
        f"- zone treatment: {summary['model_config'].get('zone_treatment', 'raw')}",
        "- gate requires train <= XGBoost, predict rows/sec >= XGBoost, and same quality as GeoBoost reference.",
        "",
        "| task/split | train ratio vs XGBoost median | train ratio min-max | "
        "predict rps ratio vs XGBoost median | predict rps ratio min-max | "
        "RMSE delta vs Geo ref | R2 delta vs Geo ref | RMSE delta vs XGB | R2 delta vs XGB | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, row in summary["ratios"].items():
        train_median = row["median_train_ratio_vs_xgboost"]
        predict_median = row["median_predict_rps_ratio_vs_xgboost"]
        gate = (
            "pass"
            if train_median <= 1.0 and predict_median >= 1.0 and same_quality(row)
            else "miss"
        )
        rmse_ref = row["median_rmse_delta_vs_geoboost_reference"]
        r2_ref = row["median_r2_delta_vs_geoboost_reference"]
        lines.append(
            f"| {key} | {train_median:.2f}x | "
            f"{row['min_train_ratio_vs_xgboost']:.2f}x-{row['max_train_ratio_vs_xgboost']:.2f}x | "
            f"{predict_median:.3f}x | "
            f"{row['min_predict_rps_ratio_vs_xgboost']:.3f}x-"
            f"{row['max_predict_rps_ratio_vs_xgboost']:.3f}x | "
            f"{format_delta(rmse_ref)} | "
            f"{format_delta(r2_ref)} | "
            f"{row['median_rmse_delta_vs_xgboost']:.6f} | "
            f"{row['median_r2_delta_vs_xgboost']:.6f} | {gate} |"
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
            "train_ratio_vs_xgboost_max": 1.0,
            "predict_rps_ratio_vs_xgboost_min": 1.0,
            "quality_delta_vs_geoboost_reference_abs_max": 1e-9,
        },
        "model_config": {
            "baseline_n_estimators": args.n_estimators,
            "geoboost_n_estimators": args.geoboost_n_estimators,
            "baseline_max_depth": args.max_depth,
            "geoboost_max_depth": args.geoboost_max_depth,
            "geoboost_splitters": args.geoboost_splitters,
            "xgboost_tree_method": args.xgboost_tree_method,
            "xgboost_subsample": args.xgboost_subsample,
            "xgboost_colsample_bytree": args.xgboost_colsample_bytree,
            "zone_treatment": args.zone_treatment,
            "zone_target_smoothing": args.zone_target_smoothing,
        },
        "ratios": ratios,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.summary_md)


if __name__ == "__main__":
    main()
