#!/usr/bin/env python3
"""Run a 10-step NYC taxi speed optimization loop for CartoBoost.

The loop keeps the maintained dataset/split/task path but compares candidate
CartoBoost presets against the current CartoBoost preset instead of XGBoost. A
candidate is accepted only when it improves median training speed and does not
degrade RMSE or R2 on any task/split beyond the configured tolerances.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "scripts" / "run_nyc_taxi_quality_benchmarks.py"
DEFAULT_OUTPUT_DIR = ROOT / "target" / "nyc_taxi_optimization_loop"


@dataclass(frozen=True)
class Preset:
    name: str
    n_estimators: int = 100
    learning_rate: float = 0.08
    max_depth: int = 5
    splitters: str = "axis_histogram:512,periodic:24,periodic:7,sparse_set"
    min_samples_leaf: int = 1
    constant_l2: float = 0.0


BASELINE = Preset(name="baseline")


def splitters_with_bins(bins: int) -> str:
    return f"axis_histogram:{bins},periodic:24,periodic:7,sparse_set"


PRESETS = [
    Preset(name="01_bins_256", splitters=splitters_with_bins(256)),
    Preset(name="02_bins_128", splitters=splitters_with_bins(128)),
    Preset(name="03_estimators_80_lr_010", n_estimators=80, learning_rate=0.10),
    Preset(name="04_estimators_75_lr_011", n_estimators=75, learning_rate=0.11),
    Preset(name="05_depth_4", max_depth=4),
    Preset(
        name="06_depth_4_estimators_80_lr_010", n_estimators=80, learning_rate=0.10, max_depth=4
    ),
    Preset(name="07_min_leaf_5", min_samples_leaf=5),
    Preset(
        name="08_bins_256_min_leaf_5",
        splitters=splitters_with_bins(256),
        min_samples_leaf=5,
    ),
    Preset(name="09_estimators_60_lr_013", n_estimators=60, learning_rate=0.13),
    Preset(
        name="10_depth_4_estimators_60_lr_013_bins_256",
        n_estimators=60,
        learning_rate=0.13,
        max_depth=4,
        splitters=splitters_with_bins(256),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1")
    parser.add_argument("--sample-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--zone-treatment", default="target_mean", choices=["raw", "target_mean"])
    parser.add_argument("--zone-target-smoothing", type=float, default=20.0)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--rmse-tolerance", type=float, default=0.0)
    parser.add_argument("--r2-tolerance", type=float, default=0.0)
    return parser.parse_args()


def benchmark_command(args: argparse.Namespace, preset: Preset, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--models",
        "cartoboost",
        "--cartoboost-n-estimators",
        str(preset.n_estimators),
        "--learning-rate",
        str(preset.learning_rate),
        "--cartoboost-max-depth",
        str(preset.max_depth),
        "--cartoboost-splitters",
        preset.splitters,
        "--cartoboost-min-samples-leaf",
        str(preset.min_samples_leaf),
        "--cartoboost-constant-l2",
        str(preset.constant_l2),
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


def run_preset(args: argparse.Namespace, preset: Preset) -> dict[str, Any]:
    output_dir = args.output_dir / preset.name
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(benchmark_command(args, preset, output_dir), cwd=ROOT, check=True)
    result = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    return summarize_result(preset, result)


def summarize_result(preset: Preset, result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for task_name, task in result["tasks"].items():
        for split_name, split in task["splits"].items():
            model = split["models"]["cartoboost"]
            if model["status"] != "ok":
                raise RuntimeError(f"{preset.name} failed for {task_name}/{split_name}: {model}")
            rows.append(
                {
                    "key": f"{task_name}/{split_name}",
                    "rmse": float(model["metrics"]["rmse"]),
                    "r2": float(model["metrics"]["r2"]),
                    "train_seconds": float(model["timing"]["train_seconds"]),
                    "predict_rows_per_second": float(model["timing"]["predict_rows_per_second"]),
                }
            )
    return {
        "preset": preset.__dict__,
        "rows": rows,
        "median_train_seconds": statistics.median(row["train_seconds"] for row in rows),
        "median_predict_rows_per_second": statistics.median(
            row["predict_rows_per_second"] for row in rows
        ),
    }


def compare_to_baseline(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rmse_tolerance: float,
    r2_tolerance: float,
) -> dict[str, Any]:
    baseline_rows = {row["key"]: row for row in baseline["rows"]}
    comparisons = []
    for row in candidate["rows"]:
        old = baseline_rows[row["key"]]
        comparisons.append(
            {
                "key": row["key"],
                "rmse_delta": row["rmse"] - old["rmse"],
                "r2_delta": row["r2"] - old["r2"],
                "train_speedup": old["train_seconds"] / row["train_seconds"],
                "predict_rps_ratio": row["predict_rows_per_second"]
                / old["predict_rows_per_second"],
            }
        )
    quality_ok = all(
        item["rmse_delta"] <= rmse_tolerance and item["r2_delta"] >= -r2_tolerance
        for item in comparisons
    )
    median_train_speedup = statistics.median(item["train_speedup"] for item in comparisons)
    median_predict_rps_ratio = statistics.median(item["predict_rps_ratio"] for item in comparisons)
    speed_ok = median_train_speedup > 1.0
    if quality_ok and speed_ok:
        bucket = "accept"
    elif not quality_ok:
        bucket = "quality_degraded"
    elif not speed_ok:
        bucket = "speed_regressed"
    else:
        bucket = "mixed"
    return {
        "preset": candidate["preset"],
        "bucket": bucket,
        "quality_ok": quality_ok,
        "speed_ok": speed_ok,
        "median_train_speedup": median_train_speedup,
        "median_predict_rps_ratio": median_predict_rps_ratio,
        "max_rmse_delta": max(item["rmse_delta"] for item in comparisons),
        "min_r2_delta": min(item["r2_delta"] for item in comparisons),
        "comparisons": comparisons,
    }


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# NYC Taxi Optimization Loop",
        "",
        "This report runs a waterfall speed loop against the current CartoBoost preset.",
        "A candidate is accepted only if it improves median training speed and does not "
        "degrade RMSE or R2 on any task/split.",
        "",
        "## Buckets",
        "",
    ]
    bucket_counts: dict[str, int] = {}
    for item in summary["candidates"]:
        bucket_counts[item["bucket"]] = bucket_counts.get(item["bucket"], 0) + 1
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- `{bucket}`: {count}")
    lines.extend(
        [
            "",
            "## Trials",
            "",
            "| iteration | preset | bucket | median train speedup | "
            "median predict rps ratio | max RMSE delta | min R2 delta |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, item in enumerate(summary["candidates"], start=1):
        lines.append(
            f"| {idx} | {item['preset']['name']} | {item['bucket']} | "
            f"{item['median_train_speedup']:.3f}x | {item['median_predict_rps_ratio']:.3f}x | "
            f"{item['max_rmse_delta']:.6f} | {item['min_r2_delta']:.6f} |"
        )
    if summary["best_accepted"] is not None:
        best = summary["best_accepted"]
        lines.extend(
            [
                "",
                "## Best Accepted Preset",
                "",
                f"- preset: `{best['preset']['name']}`",
                f"- median training speedup: {best['median_train_speedup']:.3f}x",
                f"- median prediction throughput ratio: {best['median_predict_rps_ratio']:.3f}x",
                f"- max RMSE delta: {best['max_rmse_delta']:.6f}",
                f"- min R2 delta: {best['min_r2_delta']:.6f}",
            ]
        )
    else:
        lines.extend(["", "## Best Accepted Preset", "", "No candidate met both gates."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.rmse_tolerance < 0.0 or args.r2_tolerance < 0.0:
        raise SystemExit("quality tolerances must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_preset(args, BASELINE)
    candidates = [
        compare_to_baseline(
            baseline,
            run_preset(args, preset),
            rmse_tolerance=args.rmse_tolerance,
            r2_tolerance=args.r2_tolerance,
        )
        for preset in PRESETS
    ]
    accepted = [item for item in candidates if item["bucket"] == "accept"]
    best_accepted = (
        max(accepted, key=lambda item: item["median_train_speedup"]) if accepted else None
    )
    summary = {
        "artifact_version": 1,
        "baseline": baseline,
        "candidates": candidates,
        "best_accepted": best_accepted,
    }
    (args.output_dir / "optimization_loop.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, args.output_dir / "optimization_loop.md")


if __name__ == "__main__":
    main()
