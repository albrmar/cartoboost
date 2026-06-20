#!/usr/bin/env python3
"""Benchmark CartoBoost against explicit forecasting libraries.

The default fixture is synthetic but domain-shaped: daily pickup/dropoff lane
demand with zone IDs, route distance, airport-lane structure, borough codes,
weekly effects, and deterministic event spikes. The real-data path aggregates
NYC TLC trip records into the same lane-demand shape.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import resource
import subprocess
import sys
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from cartoboost import __version__
from cartoboost.forecasting.global_models import CartoBoostLagForecaster
from cartoboost.forecasting.local import AutoStatsBank
from cartoboost.forecasting.schema import ForecastFrame
from cartoboost.metrics.m6 import rank_probability_score
from cartoboost.metrics.wrmsse import rmsse_scale, wrmsse

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

DEFAULT_CACHE_DIR = ROOT / "data" / "nyc_taxi"
DEFAULT_FORECASTING_CACHE_DIR = ROOT / "data" / "forecasting_benchmarks"

BASE_CARTOBOOST_LAGS = [1, 7, 14, 21, 28]
BASE_CARTOBOOST_ROLLING_WINDOWS = [7, 14, 28]
EXOGENOUS_FEATURE_COLUMNS = [
    "date_dayofweek",
    "date_day",
    "date_dayofyear",
    "date_month",
    "date_elapsed_days",
    "pickup_zone",
    "dropoff_zone",
    "distance_miles",
    "airport_lane",
    "pickup_borough_code",
]
STATIC_COVARIATES = [
    "pickup_zone",
    "dropoff_zone",
    "distance_miles",
    "airport_lane",
    "pickup_borough_code",
]
FUNCTIME_MODELS = ["functime_snaive", "functime_ridge", "functime_lightgbm"]
STATSFORECAST_MODELS = [
    "statsforecast_seasonal_naive",
    "statsforecast_autoets",
    "statsforecast_autoarima",
    "statsforecast_autotheta",
    "statsforecast_autoces",
    "statsforecast_dynamic_optimized_theta",
    "statsforecast_autotbats",
]
PROPHET_MODELS = ["prophet_additive"]
EXTERNAL_TREE_MODELS = ["xgboost_lag", "lightgbm_lag"]
FORECASTING_LIBRARY_MODELS = {
    "functime": FUNCTIME_MODELS,
    "statsforecast": STATSFORECAST_MODELS,
    "prophet": PROPHET_MODELS,
    "external_trees": EXTERNAL_TREE_MODELS,
}
MODEL_LIBRARIES = {
    "cartoboost_lag": "cartoboost",
    "cartoboost_auto_forecast": "cartoboost",
    **{model: "functime" for model in FUNCTIME_MODELS},
    **{model: "statsforecast" for model in STATSFORECAST_MODELS},
    **{model: "prophet" for model in PROPHET_MODELS},
    **{model: "external_trees" for model in EXTERNAL_TREE_MODELS},
}
FORECASTING_LIBRARY_BASELINES = [
    *FUNCTIME_MODELS,
    *STATSFORECAST_MODELS,
    *PROPHET_MODELS,
    *EXTERNAL_TREE_MODELS,
]
SCALABLE_FORECASTING_LIBRARY_BASELINES = [
    *FUNCTIME_MODELS,
    *EXTERNAL_TREE_MODELS,
]
AIRPORT_ZONE_IDS = {1, 132, 138}
M4_GROUPS = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
M6_ASSETS_URL = "https://raw.githubusercontent.com/Mcompetitions/M6-methods/main/assets_m6.csv"
AUTO_ENSEMBLE_CANDIDATE = "cartoboost_validation_weighted_ensemble"
AUTO_SELECTION_MIN_RELATIVE_GAIN = 0.03
AUTO_SELECTION_ROBUST_RELATIVE_TOLERANCE = 0.05
SYNTHETIC_PROBLEMS = [
    "taxi_weekly",
    "airport_calendar_events",
    "route_mix_shift",
    "borough_monthly_pulses",
]
PROPHET_CLASS: Any | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CartoBoost global lag forecasts against forecasting libraries."
    )
    parser.add_argument(
        "--source",
        choices=["polars", "duckdb", "nyc-taxi", "m4", "m5", "m6"],
        default="polars",
    )
    parser.add_argument("--output", default="artifacts/forecasting_library_benchmark_polars.json")
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument(
        "--problem",
        choices=SYNTHETIC_PROBLEMS,
        default="taxi_weekly",
        help="Synthetic problem to run when --source is polars or duckdb.",
    )
    parser.add_argument(
        "--suite",
        nargs="?",
        const="synthetic",
        choices=["synthetic", "committed"],
        default=None,
        help=(
            "Run all synthetic forecasting problems and report aggregate rankings. "
            "Use 'committed' for the fixed committed sample suite."
        ),
    )
    parser.add_argument(
        "--no-hyperopt",
        action="store_true",
        help="Benchmark integrity marker: model menus/settings are fixed and deterministic.",
    )
    parser.add_argument(
        "--suite-folds",
        type=int,
        default=3,
        help="Rolling-origin folds per synthetic problem when --suite is set.",
    )
    parser.add_argument("--lanes", type=int, default=36)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1", help="Comma-separated month numbers, e.g. 1,2,3")
    parser.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--m4-group",
        default="Hourly",
        choices=M4_GROUPS,
    )
    parser.add_argument(
        "--m4-suite",
        action="store_true",
        help="Run all six M4 groups: Hourly, Daily, Weekly, Monthly, Quarterly, and Yearly.",
    )
    parser.add_argument(
        "--m4-series-limit",
        type=int,
        default=96,
        help=(
            "Maximum M4 series per group to score locally; use 0 for every series. "
            "The full group dataset is still downloaded."
        ),
    )
    parser.add_argument(
        "--m5-data-dir",
        type=Path,
        default=DEFAULT_FORECASTING_CACHE_DIR / "m5",
        help=(
            "Directory containing Kaggle M5 files. Requires calendar.csv and either "
            "sales_train_evaluation.csv or sales_train_validation.csv."
        ),
    )
    parser.add_argument(
        "--m5-series-limit",
        type=int,
        default=0,
        help="Maximum M5 item-store series to score; use 0 for the full bottom-level corpus.",
    )
    parser.add_argument(
        "--m5-history-days",
        type=int,
        default=365,
        help=(
            "Most recent M5 daily columns to materialize before the 28-day holdout; "
            "use 0 for every available day."
        ),
    )
    parser.add_argument(
        "--m6-assets-path",
        type=Path,
        default=DEFAULT_FORECASTING_CACHE_DIR / "m6" / "assets_m6.csv",
        help="Path to the M6 assets CSV with symbol/date/price columns.",
    )
    parser.add_argument(
        "--m6-series-limit",
        type=int,
        default=0,
        help="Maximum M6 symbols to score; use 0 for every symbol in the assets file.",
    )
    parser.add_argument(
        "--m6-horizon",
        type=int,
        default=28,
        help="Daily return holdout horizon for the M6 point-forecast proxy.",
    )
    parser.add_argument(
        "--model-roster",
        choices=["full", "scalable", "cartoboost"],
        default="full",
        help=(
            "Forecast model roster. Use scalable for full M5-style panels where "
            "per-series Prophet/StatsForecast models are impractical."
        ),
    )
    parser.add_argument(
        "--allow-full-m5-roster",
        action="store_true",
        help=(
            "Allow the full per-series library roster on the unbounded M5 corpus. "
            "Without this flag, full M5 roster runs require a positive --m5-series-limit."
        ),
    )
    parser.add_argument(
        "--no-candidate-selection",
        action="store_true",
        help="Skip inner-origin shared candidate selection for very large panels.",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--cartoboost-n-estimators", type=int, default=180)
    parser.add_argument("--cartoboost-learning-rate", type=float, default=0.06)
    parser.add_argument("--cartoboost-max-depth", type=int, default=4)
    parser.add_argument("--cartoboost-min-samples-leaf", type=int, default=8)
    args = parser.parse_args()
    validate_args(args)
    ensure_prophet_class()

    cartoboost_config = {
        "n_estimators": args.cartoboost_n_estimators,
        "learning_rate": args.cartoboost_learning_rate,
        "max_depth": args.cartoboost_max_depth,
        "min_samples_leaf": args.cartoboost_min_samples_leaf,
        "splitters": ["axis_histogram:128", "periodic:7"],
    }

    benchmark_start = perf_counter()
    if args.suite:
        return run_synthetic_suite(args, cartoboost_config, benchmark_start)
    if args.m4_suite:
        return run_m4_suite(args, cartoboost_config, benchmark_start)

    load_start = perf_counter()
    table, dataset = load_dataset(args)
    dataset["dataset_hash"] = canonical_dataset_hash(table)
    dataset_source_hashes = source_file_hashes(dataset)
    load_seconds = perf_counter() - load_start
    benchmark_horizon = int(dataset.get("horizon", args.horizon))
    season_length = int(dataset.get("season_length", 7))
    metrics, quality, timing, scored = score_models(
        table,
        horizon=benchmark_horizon,
        season_length=season_length,
        cartoboost_config=cartoboost_config,
        model_names=benchmark_model_names(args.model_roster),
        source=args.source,
        candidate_selection=not args.no_candidate_selection,
    )
    total_seconds = perf_counter() - benchmark_start
    timing = {
        "total_seconds": total_seconds,
        "load_seconds": load_seconds,
        **timing,
    }
    plots = (
        write_forecast_plots(
            scored,
            args.plot_dir,
            prefix=args.source,
            models=benchmark_model_names(args.model_roster),
        )
        if args.plot_dir
        else []
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "dataset_hash": dataset["dataset_hash"],
        "source_file_hashes": dataset_source_hashes,
        "benchmark_integrity": benchmark_integrity(args),
        "benchmark": "geotemporal_lane_demand_forecasting_libraries",
        "fixture_source": args.source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": forecasting_library_models_for_roster(args.model_roster),
        "model_libraries": MODEL_LIBRARIES,
        "dataset": dataset,
        "models": benchmark_model_names(args.model_roster),
        "model_roster": args.model_roster,
        "model_settings": {"cartoboost_lag": cartoboost_benchmark_settings(cartoboost_config)},
        "metrics": metrics,
        "quality": quality,
        "official_metrics": benchmark_objective_artifacts(
            args.source,
            train_table=table,
            scored=scored,
            model_names=benchmark_model_names(args.model_roster),
            season_length=season_length,
            cartoboost_config=cartoboost_config,
        ),
        "timing": timing,
        "resource_usage": resource_usage_snapshot(),
        "plots": plots,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"quality": quality, "timing": timing}, indent=2, sort_keys=True))
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.lanes <= 0:
        raise ValueError("--lanes must be positive")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.source in {"polars", "duckdb"} and args.days <= args.horizon + max(28, args.horizon):
        raise ValueError("--days must leave at least 28 training days before the holdout")
    if args.cartoboost_n_estimators <= 0:
        raise ValueError("--cartoboost-n-estimators must be positive")
    if args.cartoboost_max_depth <= 0:
        raise ValueError("--cartoboost-max-depth must be positive")
    if args.cartoboost_min_samples_leaf <= 0:
        raise ValueError("--cartoboost-min-samples-leaf must be positive")
    if args.suite and args.source == "nyc-taxi":
        raise ValueError("--suite is only supported for synthetic polars or duckdb sources")
    if args.suite and args.source == "m4":
        raise ValueError(
            "use --source m4 without --suite; M4 groups are already benchmark datasets"
        )
    if args.suite and args.source in {"m5", "m6"}:
        raise ValueError("--suite is only supported for synthetic polars or duckdb sources")
    if args.m4_suite and args.source != "m4":
        raise ValueError("--m4-suite requires --source m4")
    if args.m4_series_limit < 0:
        raise ValueError("--m4-series-limit must be non-negative; use 0 for every M4 series")
    if args.m5_series_limit < 0:
        raise ValueError("--m5-series-limit must be non-negative; use 0 for every M5 series")
    if args.m5_history_days < 0:
        raise ValueError("--m5-history-days must be non-negative; use 0 for every M5 day")
    if args.m6_series_limit < 0:
        raise ValueError("--m6-series-limit must be non-negative; use 0 for every M6 symbol")
    if args.m6_horizon <= 0:
        raise ValueError("--m6-horizon must be positive")
    if (
        args.source == "m5"
        and args.model_roster == "full"
        and args.m5_series_limit == 0
        and not args.allow_full_m5_roster
    ):
        raise ValueError(
            "--source m5 --model-roster full requires a positive --m5-series-limit for "
            "scientific comparison samples. To run the full per-series roster on all "
            "30,490 M5 bottom-level series, pass --allow-full-m5-roster and expect a "
            "long heavyweight benchmark."
        )
    if args.suite_folds <= 0:
        raise ValueError("--suite-folds must be positive")
    if args.suite and args.days <= args.horizon * args.suite_folds + 60:
        raise ValueError("--suite requires enough days for rolling origins and 60 training days")


def canonical_dataset_hash(table: Any) -> str:
    frame = table
    columns = sorted(str(column) for column in frame.columns)
    sort_columns = [
        column
        for column in ["lane_id", "series_id", "date", "timestamp", "horizon"]
        if column in frame.columns
    ]
    if sort_columns and hasattr(frame, "sort"):
        frame = frame.sort(sort_columns)
    frame = frame.select(columns)
    buffer = io.StringIO()
    frame.write_csv(buffer)
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def aggregate_hash(values: Any) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(value) for value in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_file_hashes(dataset: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in ["sales_file", "calendar_file", "assets_file"]:
        value = dataset.get(key)
        if not value:
            continue
        path = Path(value)
        if path.exists():
            hashes[key] = file_sha256(path)
    return hashes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def benchmark_integrity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "no_hyperopt": bool(args.no_hyperopt),
        "seed": int(args.seed),
        "model_roster": args.model_roster,
        "candidate_selection": not args.no_candidate_selection,
        "threading": {
            "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
    }


def resource_usage_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "peak_rss_mb": peak_rss_mb(usage.ru_maxrss),
    }


def peak_rss_mb(raw_maxrss: int) -> float:
    if platform.system() == "Darwin":
        return float(raw_maxrss) / (1024.0 * 1024.0)
    return float(raw_maxrss) / 1024.0


def benchmark_model_names(roster: str) -> list[str]:
    if roster == "cartoboost":
        return ["cartoboost_lag", "cartoboost_auto_forecast"]
    if roster == "scalable":
        return [
            "cartoboost_lag",
            "cartoboost_auto_forecast",
            *SCALABLE_FORECASTING_LIBRARY_BASELINES,
        ]
    return ["cartoboost_lag", "cartoboost_auto_forecast", *FORECASTING_LIBRARY_BASELINES]


def forecasting_library_models_for_roster(roster: str) -> dict[str, list[str]]:
    if roster == "cartoboost":
        return {}
    if roster == "scalable":
        return {
            "functime": FUNCTIME_MODELS,
            "external_trees": EXTERNAL_TREE_MODELS,
        }
    return FORECASTING_LIBRARY_MODELS


def run_synthetic_suite(
    args: argparse.Namespace,
    cartoboost_config: dict[str, Any],
    benchmark_start: float,
) -> int:
    results: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for problem in SYNTHETIC_PROBLEMS:
        problem_args = argparse.Namespace(**vars(args))
        problem_args.problem = problem
        load_start = perf_counter()
        table, dataset = load_synthetic_fixture(problem_args)
        dataset["dataset_hash"] = canonical_dataset_hash(table)
        load_seconds = perf_counter() - load_start
        split_results, metrics, quality, timing = score_rolling_origin_problem(
            table,
            horizon=args.horizon,
            season_length=7,
            folds=args.suite_folds,
            cartoboost_config=cartoboost_config,
            model_names=benchmark_model_names(args.model_roster),
            source="synthetic",
        )
        results[problem] = {
            "dataset": dataset,
            "splits": split_results,
            "metrics": metrics,
            "quality": quality,
        }
        timings[problem] = {
            "load_seconds": load_seconds,
            **timing,
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "dataset_hash": aggregate_hash(
            result["dataset"]["dataset_hash"] for result in results.values()
        ),
        "source_file_hashes": {},
        "benchmark_integrity": benchmark_integrity(args),
        "benchmark": "geotemporal_lane_demand_forecasting_library_suite",
        "fixture_source": args.source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": FORECASTING_LIBRARY_MODELS,
        "model_libraries": MODEL_LIBRARIES,
        "dataset": {
            "problems": SYNTHETIC_PROBLEMS,
            "series": args.lanes,
            "days": args.days,
            "horizon": args.horizon,
            "season_length": 7,
            "folds": args.suite_folds,
            "seed": args.seed,
            "domain": "synthetic NYC taxi-style forecasting problem suite",
            "split_type": "rolling_origin_last_windows",
            "static_covariates": STATIC_COVARIATES,
        },
        "models": benchmark_model_names(args.model_roster),
        "model_settings": {"cartoboost_lag": cartoboost_benchmark_settings(cartoboost_config)},
        "problems": results,
        "aggregate_quality": aggregate_suite_quality(results),
        "timing": {
            "total_seconds": perf_counter() - benchmark_start,
            "problems": timings,
        },
        "resource_usage": resource_usage_snapshot(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate_quality"], indent=2, sort_keys=True))
    return 0


def run_m4_suite(
    args: argparse.Namespace,
    cartoboost_config: dict[str, Any],
    benchmark_start: float,
) -> int:
    results: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for group in M4_GROUPS:
        group_args = argparse.Namespace(**vars(args))
        group_args.m4_group = group
        load_start = perf_counter()
        table, dataset = load_m4_fixture(group_args)
        dataset["dataset_hash"] = canonical_dataset_hash(table)
        load_seconds = perf_counter() - load_start
        metrics, quality, timing, _scored = score_models(
            table,
            horizon=int(dataset["horizon"]),
            season_length=int(dataset["season_length"]),
            cartoboost_config=cartoboost_config,
            model_names=benchmark_model_names(args.model_roster),
            source="m4",
        )
        results[group] = {
            "dataset": dataset,
            "metrics": metrics,
            "quality": quality,
        }
        timings[group] = {
            "load_seconds": load_seconds,
            **timing,
        }

    aggregate_quality = aggregate_suite_quality(results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "git_commit": read_git_commit(),
        "dataset_hash": aggregate_hash(
            result["dataset"]["dataset_hash"] for result in results.values()
        ),
        "source_file_hashes": {},
        "benchmark_integrity": benchmark_integrity(args),
        "benchmark": "m4_forecasting_library_group_suite",
        "fixture_source": args.source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": FORECASTING_LIBRARY_MODELS,
        "model_libraries": MODEL_LIBRARIES,
        "dataset": {
            "groups": M4_GROUPS,
            "source": "m4",
            "domain": "M4 forecasting competition train panels",
            "split_type": "last_official_horizon_from_training_panel",
            "series_limit_per_group": (None if args.m4_series_limit == 0 else args.m4_series_limit),
            "static_covariates": STATIC_COVARIATES,
        },
        "models": benchmark_model_names(args.model_roster),
        "model_settings": {"cartoboost_lag": cartoboost_benchmark_settings(cartoboost_config)},
        "groups": results,
        "aggregate_quality": aggregate_quality,
        "timing": {
            "total_seconds": perf_counter() - benchmark_start,
            "groups": timings,
        },
        "resource_usage": resource_usage_snapshot(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate_quality, indent=2, sort_keys=True))
    return 0


def aggregate_suite_quality(results: dict[str, Any]) -> dict[str, Any]:
    model_names = sorted(
        {model for problem in results.values() for model in problem.get("metrics", {}).keys()}
    )
    wins = dict.fromkeys(model_names, 0)
    top3 = dict.fromkeys(model_names, 0)
    rmse_ratios: dict[str, list[float]] = {model: [] for model in model_names}
    for problem in results.values():
        metrics = problem["metrics"]
        best_rmse = min(row["rmse"] for row in metrics.values())
        ranking = sorted(model_names, key=lambda name: metrics[name]["rmse"])
        for name in ranking[:3]:
            top3[name] += 1
        for name, row in metrics.items():
            if np.isclose(row["rmse"], best_rmse, rtol=1e-12):
                wins[name] += 1
            rmse_ratios[name].append(row["rmse"] / best_rmse)
    mean_rmse_ratio = {
        model: float(np.mean(values)) for model, values in rmse_ratios.items() if values
    }
    return {
        "problem_count": len(results),
        "wins_or_ties": wins,
        "top3_finishes": top3,
        "mean_rmse_ratio_to_problem_best": mean_rmse_ratio,
        "mean_rmse_ratio_ranking": sorted(mean_rmse_ratio, key=mean_rmse_ratio.__getitem__),
    }


def load_dataset(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.source in {"polars", "duckdb"}:
        return load_synthetic_fixture(args)
    if args.source == "m4":
        return load_m4_fixture(args)
    if args.source == "m5":
        return load_m5_fixture(args)
    if args.source == "m6":
        return load_m6_fixture(args)
    return load_nyc_taxi_fixture(args)


def load_synthetic_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    table = make_fixture(lanes=args.lanes, days=args.days, seed=args.seed, problem=args.problem)
    if args.source == "duckdb":
        duckdb = require_duckdb()
        con = duckdb.connect(":memory:")
        try:
            con.register("lane_demand", table.to_arrow())
            table = con.sql(
                """
                SELECT *
                FROM lane_demand
                ORDER BY lane_id, date
                """
            ).pl()
        finally:
            con.close()
    return table, {
        "series": args.lanes,
        "days": args.days,
        "horizon": args.horizon,
        "season_length": 7,
        "seed": args.seed,
        "domain": "daily NYC taxi-style pickup/dropoff lane demand",
        "source": "synthetic_fixture",
        "problem": args.problem,
        "problem_description": synthetic_problem_description(args.problem),
        "static_covariates": STATIC_COVARIATES,
    }


def load_nyc_taxi_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    from scripts.run_nyc_taxi_quality_benchmarks import (
        TLC_TRIP_RECORD_PAGE,
        clean_tlc_frame,
        ensure_parquet_files,
        ensure_zone_lookup,
        load_tlc_frame,
        parse_months,
    )

    months = parse_months(args.months)
    paths = ensure_parquet_files(
        taxi_type=args.taxi_type,
        year=args.year,
        months=months,
        cache_dir=args.cache_dir,
        no_download=args.no_download,
    )
    zone_lookup = ensure_zone_lookup(cache_dir=args.cache_dir, no_download=args.no_download)
    raw = load_tlc_frame(paths)
    clean = clean_tlc_frame(raw)
    pickup_time = pd.to_datetime(clean["tpep_pickup_datetime"])
    frame = clean.assign(
        pickup_date=pickup_time.dt.normalize(),
        lane_id=(
            "PU"
            + clean["PULocationID"].astype(int).astype(str)
            + "->DO"
            + clean["DOLocationID"].astype(int).astype(str)
        ),
    )
    lane_counts = frame.groupby("lane_id").size().sort_values(ascending=False).head(args.lanes)
    selected = frame[frame["lane_id"].isin(lane_counts.index)].copy()
    static = (
        selected.groupby("lane_id", as_index=False)
        .agg(
            pickup_zone=("PULocationID", "first"),
            dropoff_zone=("DOLocationID", "first"),
            distance_miles=("trip_distance", "mean"),
        )
        .assign(
            airport_lane=lambda data: (
                data[["pickup_zone", "dropoff_zone"]]
                .isin(AIRPORT_ZONE_IDS)
                .any(axis=1)
                .astype(float)
            ),
            pickup_borough_code=lambda data: data["pickup_zone"].map(
                lambda zone: float(zone_lookup[int(zone)].borough_code)
            ),
        )
    )
    counts = (
        selected.groupby(["lane_id", "pickup_date"], as_index=False)
        .size()
        .rename(columns={"pickup_date": "date", "size": "loads"})
    )
    dates = pd.DataFrame(
        {
            "date": pd.date_range(
                selected["pickup_date"].min(),
                selected["pickup_date"].max(),
                freq="D",
            )
        }
    )
    full_index = static[["lane_id"]].merge(dates, how="cross")
    table = (
        full_index.merge(counts, on=["lane_id", "date"], how="left")
        .merge(static, on="lane_id", how="left")
        .assign(loads=lambda data: data["loads"].fillna(0.0).astype(float))
        .sort_values(["lane_id", "date"])
    )
    result = pl.from_pandas(table).with_columns(pl.col("date").cast(pl.Datetime("us")))
    return result, {
        "series": int(static.shape[0]),
        "days": int(dates.shape[0]),
        "horizon": args.horizon,
        "season_length": 7,
        "domain": f"real daily NYC TLC {args.taxi_type} taxi pickup/dropoff lane demand",
        "source": "nyc_tlc_trip_records",
        "source_url": TLC_TRIP_RECORD_PAGE,
        "taxi_type": args.taxi_type,
        "year": args.year,
        "months": months,
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "aggregated_rows": int(result.height),
        "static_covariates": STATIC_COVARIATES,
    }


def load_m4_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    try:
        from datasetsforecast.m4 import M4
    except ImportError as exc:
        raise ImportError(
            "M4 benchmark source requires datasetsforecast; run "
            "`uv sync --group bench` after adding benchmark extras."
        ) from exc

    cache_dir = (
        DEFAULT_FORECASTING_CACHE_DIR if args.cache_dir == DEFAULT_CACHE_DIR else args.cache_dir
    )
    train, _test, info = M4.load(directory=str(cache_dir), group=args.m4_group)
    available_series_ids = sorted(train["unique_id"].unique())
    series_ids = (
        available_series_ids
        if args.m4_series_limit == 0
        else available_series_ids[: args.m4_series_limit]
    )
    data = train[train["unique_id"].isin(series_ids)].copy()
    info = info[info["unique_id"].isin(series_ids)].copy()
    category_codes = {
        category: index for index, category in enumerate(sorted(info["category"].unique()), start=1)
    }
    info = info.assign(
        series_index=lambda frame: frame["unique_id"].map(
            {series_id: index for index, series_id in enumerate(series_ids, start=1)}
        ),
        category_code=lambda frame: frame["category"].map(category_codes),
    )
    data = data.merge(info, on="unique_id", how="left")
    data = data.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    data["date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(data["ds"].astype(int), unit="D")
    data["pickup_zone"] = data["series_index"].astype(int)
    data["dropoff_zone"] = data["category_code"].astype(int)
    data["distance_miles"] = 1.0 + (data["series_index"].astype(float) % 25.0) / 5.0
    data["airport_lane"] = (data["category_code"].astype(int) % 2 == 0).astype(float)
    data["pickup_borough_code"] = data["category_code"].astype(float)
    result = pl.from_pandas(
        data.rename(columns={"unique_id": "lane_id", "y": "loads"})[
            ["lane_id", "date", "loads", *STATIC_COVARIATES]
        ]
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    horizon = m4_horizon(args.m4_group)
    season_length = m4_season_length(args.m4_group)
    return result, {
        "series": int(len(series_ids)),
        "available_series": int(len(available_series_ids)),
        "rows": int(result.height),
        "horizon": horizon,
        "season_length": season_length,
        "domain": f"M4 {args.m4_group} forecasting competition dataset",
        "source": "m4",
        "group": args.m4_group,
        "series_limit": args.m4_series_limit,
        "static_covariates": STATIC_COVARIATES,
    }


def load_m5_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pl = require_polars()
    data_dir = ensure_m5_data_dir(args.m5_data_dir, no_download=args.no_download)
    sales_path = find_m5_sales_file(data_dir)
    calendar_path = data_dir / "calendar.csv"
    if not calendar_path.exists():
        raise FileNotFoundError(
            f"M5 benchmark requires {calendar_path}; download the Kaggle M5 Accuracy files "
            "and point --m5-data-dir at the extracted directory."
        )

    sales = pl.read_csv(sales_path, n_rows=args.m5_series_limit or None)
    calendar = pl.read_csv(calendar_path)
    required_sales_columns = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing_sales = [column for column in required_sales_columns if column not in sales.columns]
    if missing_sales:
        raise ValueError(f"M5 sales file is missing required columns: {missing_sales}")
    if "id" not in sales.columns:
        sales = sales.with_columns(
            pl.concat_str(["item_id", "store_id"], separator="_").alias("id")
        )
    if "date" not in calendar.columns:
        raise ValueError("M5 calendar file is missing required column: date")
    if "d" not in calendar.columns:
        calendar = calendar.with_row_index("d_index").with_columns(
            pl.format("d_{}", pl.col("d_index") + 1).alias("d")
        )

    available_series = count_m5_series(sales_path)
    value_columns = sorted(
        [column for column in sales.columns if column.startswith("d_")],
        key=lambda value: int(value.split("_", 1)[1]),
    )
    if len(value_columns) <= 28:
        raise ValueError("M5 sales file must contain more than 28 daily observations per series")
    materialized_value_columns = (
        value_columns
        if args.m5_history_days == 0
        else value_columns[-max(args.m5_history_days, 29) :]
    )

    id_columns = ["id", *required_sales_columns]
    long = unpivot_frame(
        sales,
        index=id_columns,
        on=materialized_value_columns,
        variable_name="d",
        value_name="loads",
    )
    calendar = calendar.select(
        "d",
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).cast(pl.Datetime("us")),
    )

    lookup_frame = m5_static_lookup(sales)
    result = (
        long.join(calendar, on="d", how="inner")
        .join(lookup_frame, on=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
        .select(
            pl.col("id").alias("lane_id"),
            "date",
            pl.col("loads").cast(pl.Float64),
            *STATIC_COVARIATES,
        )
        .sort(["lane_id", "date"])
    )
    if result.height != sales.height * len(materialized_value_columns):
        raise RuntimeError(
            "M5 calendar join dropped rows; check that calendar.csv covers all d_* columns"
        )
    return result, {
        "series": int(sales.height),
        "available_series": int(available_series),
        "rows": int(result.height),
        "days": int(len(materialized_value_columns)),
        "available_days": int(len(value_columns)),
        "horizon": 28,
        "season_length": 7,
        "domain": "M5 Forecasting Accuracy Walmart item-store unit sales",
        "source": "m5_kaggle_local_files",
        "source_url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data",
        "mirror_url": "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip",
        "sales_file": str(sales_path),
        "calendar_file": str(calendar_path),
        "series_limit": int(args.m5_series_limit),
        "history_days": int(args.m5_history_days),
        "split_type": "last_28_days_from_training_or_evaluation_file",
        "static_covariates": STATIC_COVARIATES,
    }


def count_m5_series(sales_path: Path) -> int:
    with sales_path.open("rb") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def ensure_m5_data_dir(data_dir: Path, *, no_download: bool) -> Path:
    if m5_data_files_exist(data_dir):
        return data_dir
    nested_data_dir = data_dir / "datasets"
    if m5_data_files_exist(nested_data_dir):
        return nested_data_dir
    if no_download:
        return data_dir
    try:
        from datasetsforecast.m5 import M5
    except ImportError as exc:
        raise ImportError(
            "M5 benchmark download requires datasetsforecast; run `uv sync --group bench`."
        ) from exc
    M5.download(str(data_dir.parent))
    if m5_data_files_exist(nested_data_dir):
        return nested_data_dir
    if m5_data_files_exist(data_dir):
        return data_dir
    raise FileNotFoundError(
        f"M5 public mirror download completed but required CSVs were not found under {data_dir}"
    )


def m5_data_files_exist(data_dir: Path) -> bool:
    return (data_dir / "calendar.csv").exists() and (
        (data_dir / "sales_train_evaluation.csv").exists()
        or (data_dir / "sales_train_validation.csv").exists()
    )


def find_m5_sales_file(data_dir: Path) -> Path:
    candidates = [
        data_dir / "sales_train_evaluation.csv",
        data_dir / "sales_train_validation.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "M5 benchmark requires sales_train_evaluation.csv or sales_train_validation.csv "
        f"under {data_dir}; download the Kaggle M5 Accuracy files first."
    )


def unpivot_frame(
    frame: Any,
    *,
    index: list[str],
    on: list[str],
    variable_name: str,
    value_name: str,
) -> Any:
    if hasattr(frame, "unpivot"):
        return frame.unpivot(
            index=index,
            on=on,
            variable_name=variable_name,
            value_name=value_name,
        )
    return frame.melt(
        id_vars=index,
        value_vars=on,
        variable_name=variable_name,
        value_name=value_name,
    )


def m5_static_lookup(sales: Any) -> Any:
    pl = require_polars()
    base = sales.select("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
    lookups = [
        code_lookup(base, "store_id", "pickup_zone_code"),
        code_lookup(base, "item_id", "dropoff_zone_code"),
        code_lookup(base, "dept_id", "dept_code"),
        code_lookup(base, "cat_id", "cat_code"),
        code_lookup(base, "state_id", "state_code"),
    ]
    for lookup in lookups:
        base = base.join(lookup, on=lookup.columns[0], how="left")
    return base.with_columns(
        pl.col("pickup_zone_code").cast(pl.Float64).alias("pickup_zone"),
        pl.col("dropoff_zone_code").cast(pl.Float64).alias("dropoff_zone"),
        (1.0 + (pl.col("dept_code").cast(pl.Float64) % 20.0) / 4.0).alias("distance_miles"),
        (pl.col("cat_code") % 2).cast(pl.Float64).alias("airport_lane"),
        pl.col("state_code").cast(pl.Float64).alias("pickup_borough_code"),
    ).select("id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *STATIC_COVARIATES)


def code_lookup(frame: Any, column: str, output: str) -> Any:
    pl = require_polars()
    values = sorted(frame.select(pl.col(column).unique()).to_series().to_list())
    return pl.DataFrame({column: values, output: list(range(1, len(values) + 1))})


def load_m6_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    assets_path = ensure_m6_assets_file(args.m6_assets_path, no_download=args.no_download)
    raw = pd.read_csv(assets_path)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required_columns = {"symbol", "date", "price"}
    missing = sorted(required_columns - set(raw.columns))
    if missing:
        raise ValueError(f"M6 assets file is missing required columns: {missing}")
    raw = raw[["symbol", "date", "price"]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
    raw["price"] = pd.to_numeric(raw["price"], errors="raise")
    raw = raw.dropna(subset=["symbol", "date", "price"]).sort_values(["symbol", "date"])
    if raw.empty:
        raise ValueError("M6 assets file did not contain any usable symbol/date/price rows")

    available_symbols = sorted(raw["symbol"].astype(str).unique())
    selected_symbols = (
        available_symbols
        if args.m6_series_limit == 0
        else available_symbols[: args.m6_series_limit]
    )
    raw = raw[raw["symbol"].astype(str).isin(selected_symbols)].copy()
    result = build_m6_daily_return_panel(raw, selected_symbols)
    day_count = result.select(pl.col("date").unique()).height
    if day_count <= args.m6_horizon + 60:
        raise ValueError("M6 assets file does not leave enough daily observations for the holdout")
    return result, {
        "series": int(len(selected_symbols)),
        "available_series": int(len(available_symbols)),
        "rows": int(result.height),
        "days": int(day_count),
        "horizon": int(args.m6_horizon),
        "season_length": 7,
        "domain": "M6 financial competition assets daily return point-forecast proxy",
        "source": "m6_methods_assets_csv",
        "source_url": M6_ASSETS_URL,
        "assets_file": str(assets_path),
        "series_limit": int(args.m6_series_limit),
        "split_type": f"last_{args.m6_horizon}_calendar_days_from_daily_return_panel",
        "official_metric_note": (
            "M6 official scoring used probability rank buckets, RPS, and investment return. "
            "This benchmark scores daily point return forecasts with the shared CartoBoost "
            "library RMSE/MAE/WAPE harness."
        ),
        "static_covariates": STATIC_COVARIATES,
    }


def ensure_m6_assets_file(path: Path, *, no_download: bool) -> Path:
    if path.exists():
        return path
    if no_download:
        raise FileNotFoundError(
            f"M6 benchmark requires {path}; remove --no-download to fetch {M6_ASSETS_URL} "
            "or provide --m6-assets-path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(M6_ASSETS_URL, path)
    if not path.exists():
        raise FileNotFoundError(f"failed to download M6 assets file to {path}")
    return path


def build_m6_daily_return_panel(raw: Any, selected_symbols: list[str]) -> Any:
    pd = require_pandas_for_benchmark()
    pl = require_polars()
    symbol_codes = {symbol: index for index, symbol in enumerate(selected_symbols, start=1)}
    pieces = []
    for symbol in selected_symbols:
        group = raw[raw["symbol"].astype(str) == symbol].sort_values("date")
        if group.empty:
            continue
        date_index = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        prices = (
            group.drop_duplicates("date")
            .set_index("date")["price"]
            .sort_index()
            .reindex(date_index)
            .ffill()
        )
        returns = prices.pct_change().fillna(0.0)
        code = symbol_codes[symbol]
        pieces.append(
            pd.DataFrame(
                {
                    "lane_id": symbol,
                    "date": date_index,
                    "loads": returns.to_numpy(dtype=float),
                    "pickup_zone": float(code),
                    "dropoff_zone": float((code % 11) + 1),
                    "distance_miles": np.log1p(prices.to_numpy(dtype=float)),
                    "airport_lane": float(code % 2),
                    "pickup_borough_code": float((code % 5) + 1),
                }
            )
        )
    if not pieces:
        raise ValueError("M6 assets file did not contain any selected symbols")
    return (
        pl.from_pandas(pd.concat(pieces, ignore_index=True))
        .with_columns(
            pl.col("date").cast(pl.Datetime("us")),
            pl.col("loads").fill_nan(0.0).fill_null(0.0),
            pl.col("distance_miles").fill_nan(0.0).fill_null(0.0),
        )
        .sort(["lane_id", "date"])
    )


def m4_horizon(group: str) -> int:
    horizons = {
        "Hourly": 48,
        "Daily": 14,
        "Weekly": 13,
        "Monthly": 18,
        "Quarterly": 8,
        "Yearly": 6,
    }
    return horizons[group]


def m4_season_length(group: str) -> int:
    season_lengths = {
        "Hourly": 24,
        "Daily": 1,
        "Weekly": 1,
        "Monthly": 12,
        "Quarterly": 4,
        "Yearly": 1,
    }
    return season_lengths[group]


def synthetic_problem_description(problem: str) -> str:
    descriptions = {
        "taxi_weekly": "Weekly lane demand with slow drift and deterministic airport events.",
        "airport_calendar_events": (
            "Airport pickup/dropoff lanes receive repeated day-of-month surges."
        ),
        "route_mix_shift": (
            "Longer routes and airport lanes have horizon-relevant route-mix swings."
        ),
        "borough_monthly_pulses": (
            "Pickup borough codes drive repeated monthly taxi-demand pulses."
        ),
    }
    return descriptions[problem]


def make_fixture(*, lanes: int, days: int, seed: int, problem: str) -> Any:
    pl = require_polars()
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1)
    rows = []
    for lane_idx in range(lanes):
        pickup_zone = 101 + lane_idx
        dropoff_zone = 201 + ((lane_idx * 7) % lanes)
        distance = 1.5 + (lane_idx % 9) * 0.8
        airport_lane = float(lane_idx % 11 == 0)
        pickup_borough_code = float(lane_idx % 5)
        base = 12.0 + 0.35 * distance + 5.0 * airport_lane + 1.2 * pickup_borough_code
        lane_effect = 2.0 * np.sin(lane_idx / 3.0)
        lane_noise = rng.normal(loc=0.0, scale=0.03)
        for day in range(days):
            timestamp = start + timedelta(days=day)
            weekly = [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 2.0][timestamp.weekday()]
            slow_drift = 0.04 * day
            airport_event = synthetic_airport_event(problem, airport_lane, timestamp, day)
            route_event = synthetic_route_event(problem, distance, airport_lane, timestamp, day)
            borough_event = synthetic_borough_event(problem, pickup_borough_code, timestamp)
            quarterly_event = (
                2.5 if problem == "taxi_weekly" and day % 91 in {12, 13, 14, 15} else 0.0
            )
            deterministic_noise = ((lane_idx * 17 + day * 13) % 11 - 5) * 0.12
            demand = max(
                0.0,
                base
                + lane_effect
                + weekly
                + slow_drift
                + airport_event
                + route_event
                + borough_event
                + quarterly_event
                + deterministic_noise,
            )
            rows.append(
                {
                    "lane_id": f"PU{pickup_zone}->DO{dropoff_zone}",
                    "date": timestamp,
                    "loads": float(demand + lane_noise),
                    "pickup_zone": pickup_zone,
                    "dropoff_zone": dropoff_zone,
                    "distance_miles": float(distance),
                    "airport_lane": airport_lane,
                    "pickup_borough_code": pickup_borough_code,
                }
            )
    return pl.DataFrame(rows)


def synthetic_airport_event(
    problem: str, airport_lane: float, timestamp: datetime, day: int
) -> float:
    if not airport_lane:
        return 0.0
    if problem == "airport_calendar_events":
        return 8.0 if timestamp.day in {5, 6, 20, 21} else 0.0
    if problem == "route_mix_shift":
        return 4.5 if timestamp.weekday() in {0, 4, 6} else 0.0
    return 4.0 if day % 28 in {5, 6, 7} else 0.0


def synthetic_route_event(
    problem: str,
    distance: float,
    airport_lane: float,
    timestamp: datetime,
    day: int,
) -> float:
    if problem != "route_mix_shift":
        return 0.0
    long_route = distance >= 5.5
    if long_route and day % 14 in {10, 11, 12, 13}:
        return 5.5
    if airport_lane and timestamp.day in {1, 2, 15, 16}:
        return 3.5
    return 0.0


def synthetic_borough_event(problem: str, pickup_borough_code: float, timestamp: datetime) -> float:
    if problem != "borough_monthly_pulses":
        return 0.0
    if int(pickup_borough_code) in {1, 3} and timestamp.day in {8, 9, 10}:
        return 7.0
    if int(pickup_borough_code) in {2, 4} and timestamp.day in {23, 24, 25}:
        return 5.0
    return 0.0


def score_models(
    table: Any,
    *,
    horizon: int,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str] | None = None,
    source: str = "synthetic",
    candidate_selection: bool = True,
    cutoff: Any | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, Any], Any]:
    pl = require_polars()
    if model_names is None:
        model_names = benchmark_model_names("full")
    train, test, cutoff = train_test_split_for_cutoff(table, horizon=horizon, cutoff=cutoff)
    if train.is_empty() or test.is_empty():
        raise ValueError("benchmark split produced empty train or test data")

    actual = (
        test.sort(["lane_id", "date"])
        .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
        .select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
            "horizon",
            pl.col("loads").alias("actual"),
        )
    )
    predictions, timing = forecast_model_roster(
        train,
        horizon,
        season_length=season_length,
        cartoboost_config=cartoboost_config,
        model_names=model_names,
        source=source,
    )
    if candidate_selection:
        predictions, selection_timing = apply_shared_candidate_selection(
            train,
            horizon,
            season_length=season_length,
            source=source,
            raw_predictions=predictions,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
        )
    else:
        selection_timing = {
            "calibration_seconds": 0.0,
            "inner_origin_count": 0.0,
            "selected_candidates": {model: model for model in model_names},
            "disabled": True,
        }
    scored = actual.join(predictions, on=["series_id", "timestamp", "horizon"], how="inner")
    if scored.height != actual.height:
        raise RuntimeError("forecast alignment dropped rows")

    metrics = {
        model: evaluate_metrics(scored, model, train, season_length=season_length)
        for model in model_names
    }
    quality = quality_summary(metrics, model_names=model_names)
    timing["candidate_selection"] = selection_timing
    return metrics, quality, timing, scored


def train_test_split_for_cutoff(
    table: Any,
    *,
    horizon: int,
    cutoff: Any | None,
) -> tuple[Any, Any, Any]:
    pl = require_polars()
    timestamps = table.select(pl.col("date").unique().sort()).to_series().to_list()
    if cutoff is None:
        cutoff = timestamps[-horizon]
    try:
        start_index = timestamps.index(cutoff)
    except ValueError as exc:
        raise ValueError(f"cutoff {cutoff!r} is not present in benchmark timestamps") from exc
    end_index = start_index + horizon
    if end_index > len(timestamps):
        raise ValueError("cutoff leaves fewer timestamps than requested horizon")
    validation_timestamps = timestamps[start_index:end_index]
    train = table.filter(pl.col("date") < cutoff)
    test = table.filter(pl.col("date").is_in(validation_timestamps))
    return train, test, cutoff


def auto_selection_objective(source: str) -> str:
    if source == "m5":
        return "wrmsse"
    if source == "m6":
        return "rank_probability_score_then_rmse"
    if source == "m4":
        return "owa_proxy"
    return "rmse"


def forecast_objective_loss(
    objective: str,
    *,
    train: Any,
    scored: Any,
    prediction_col: str,
    season_length: int,
) -> float:
    if objective == "wrmsse":
        artifact = m5_wrmsse_artifact(
            train,
            scored,
            model_names=[prediction_col],
            seasonal_period=1,
        )
        value = artifact["model_scores"][prediction_col]
        return math.inf if value is None else float(value)
    if objective == "rank_probability_score":
        artifact = m6_rps_artifact(scored, model_names=[prediction_col])
        return float(artifact["models"][prediction_col]["mean_rps"])
    if objective == "rank_probability_score_then_rmse":
        rps = forecast_objective_loss(
            "rank_probability_score",
            train=train,
            scored=scored,
            prediction_col=prediction_col,
            season_length=season_length,
        )
        rmse = rmse_expr(scored, prediction_col)
        return float(rps + 1.0e-6 * rmse)
    if objective == "owa_proxy":
        metrics = evaluate_metrics(scored, prediction_col, train, season_length=season_length)
        return float(0.5 * metrics["mase"] + 0.5 * metrics["smape"])
    return rmse_expr(scored, prediction_col)


def rolling_origin_cutoffs(table: Any, *, horizon: int, folds: int) -> list[Any]:
    timestamps = table.select(require_polars().col("date").unique().sort()).to_series().to_list()
    required = horizon * folds + 1
    if len(timestamps) <= required:
        raise ValueError("not enough timestamps for requested rolling-origin folds")
    start = len(timestamps) - horizon * folds
    return [timestamps[start + fold * horizon] for fold in range(folds)]


def score_rolling_origin_problem(
    table: Any,
    *,
    horizon: int,
    season_length: int,
    folds: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    source: str = "synthetic",
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, Any], dict[str, Any]]:
    split_results: dict[str, Any] = {}
    timing: dict[str, Any] = {"splits": {}}
    cutoffs = rolling_origin_cutoffs(table, horizon=horizon, folds=folds)
    for fold_index, cutoff in enumerate(cutoffs, start=1):
        split_name = f"rolling_origin_{fold_index}"
        metrics, quality, split_timing, _scored = score_models(
            table,
            horizon=horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            source=source,
            cutoff=cutoff,
        )
        split_results[split_name] = {
            "cutoff": str(cutoff),
            "metrics": metrics,
            "quality": quality,
        }
        timing["splits"][split_name] = split_timing
    aggregate_metrics = aggregate_split_metrics(split_results)
    return split_results, aggregate_metrics, quality_summary(aggregate_metrics), timing


def aggregate_split_metrics(split_results: dict[str, Any]) -> dict[str, dict[str, float]]:
    first_split = next(iter(split_results.values()))
    model_names = list(first_split["metrics"])
    aggregate: dict[str, dict[str, float]] = {}
    metric_names = ["mae", "rmse", "mase", "wape", "smape", "bias"]
    for model in model_names:
        aggregate[model] = {
            metric: float(
                np.mean(
                    [
                        split["metrics"][model][metric]
                        for split in split_results.values()
                        if metric in split["metrics"][model]
                    ]
                )
            )
            for metric in metric_names
        }
    return aggregate


def combine_forecast_frames(frames: list[Any]) -> Any:
    normalized = [normalize_forecast_frame(frame) for frame in frames]
    combined = normalized[0]
    for frame in normalized[1:]:
        combined = combined.join(frame, on=["series_id", "timestamp", "horizon"], how="inner")
    return combined


def normalize_forecast_frame(frame: Any) -> Any:
    pl = require_polars()
    return frame.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


def quality_summary(
    metrics: dict[str, dict[str, float]],
    *,
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    if model_names is None:
        model_names = list(metrics)
    cartoboost_models = [name for name in model_names if MODEL_LIBRARIES.get(name) == "cartoboost"]
    library_models = [name for name in model_names if name not in cartoboost_models]
    best_cartoboost_model = min(cartoboost_models, key=lambda name: metrics[name]["rmse"])
    cartoboost_rmse = metrics[best_cartoboost_model]["rmse"]
    best_rmse = min(row["rmse"] for row in metrics.values())
    tied_best_models = [
        name for name, row in metrics.items() if np.isclose(row["rmse"], best_rmse, rtol=1e-12)
    ]
    summary: dict[str, Any] = {
        "winner": "tie" if len(tied_best_models) > 1 else tied_best_models[0],
        "comparison_libraries": [
            library
            for library, names in FORECASTING_LIBRARY_MODELS.items()
            if any(name in model_names for name in names)
        ],
        "forecasting_library_models": {
            library: [name for name in names if name in model_names]
            for library, names in FORECASTING_LIBRARY_MODELS.items()
            if any(name in model_names for name in names)
        },
        "model_libraries": MODEL_LIBRARIES,
        "best_rmse": best_rmse,
        "tied_best_models": tied_best_models,
        "rmse_ranking": sorted(metrics, key=lambda name: metrics[name]["rmse"]),
        "mae_ranking": sorted(metrics, key=lambda name: metrics[name]["mae"]),
        "wape_ranking": sorted(metrics, key=lambda name: metrics[name]["wape"]),
        "best_cartoboost_model": best_cartoboost_model,
        "cartoboost_rmse": cartoboost_rmse,
        "cartoboost_mae": metrics[best_cartoboost_model]["mae"],
        "cartoboost_wape": metrics[best_cartoboost_model]["wape"],
    }
    if library_models:
        best_library_model = min(library_models, key=lambda name: metrics[name]["rmse"])
        library_rmse = metrics[best_library_model]["rmse"]
        summary.update(
            {
                "best_forecasting_library": MODEL_LIBRARIES[best_library_model],
                "best_forecasting_library_model": best_library_model,
                "best_forecasting_library_rmse": library_rmse,
                "best_forecasting_library_mae": metrics[best_library_model]["mae"],
                "best_forecasting_library_wape": metrics[best_library_model]["wape"],
                "rmse_delta_vs_best_forecasting_library": cartoboost_rmse - library_rmse,
                "rmse_ratio_vs_best_forecasting_library": cartoboost_rmse / library_rmse,
                "rmse_reduction_vs_best_forecasting_library": 1.0 - cartoboost_rmse / library_rmse,
                "mae_delta_vs_best_forecasting_library": metrics[best_cartoboost_model]["mae"]
                - metrics[best_library_model]["mae"],
                "mae_reduction_vs_best_forecasting_library": 1.0
                - metrics[best_cartoboost_model]["mae"] / metrics[best_library_model]["mae"],
                "wape_reduction_vs_best_forecasting_library": 1.0
                - metrics[best_cartoboost_model]["wape"] / metrics[best_library_model]["wape"],
            }
        )
    for library, library_model_names in FORECASTING_LIBRARY_MODELS.items():
        available_models = [name for name in library_model_names if name in metrics]
        if not available_models:
            continue
        best_model = min(available_models, key=lambda name: metrics[name]["rmse"])
        summary[f"best_{library}_method"] = best_model
        summary[f"best_{library}_rmse"] = metrics[best_model]["rmse"]
    return summary


def benchmark_objective_artifacts(
    source: str,
    *,
    train_table: Any,
    scored: Any,
    model_names: list[str],
    season_length: int,
    cartoboost_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source == "m5":
        return {
            "primary_metric": "wrmsse",
            "objective": "M5 Forecasting Accuracy level-aware weighted RMSSE",
            "m5": m5_wrmsse_artifact(
                train_table,
                scored,
                model_names=model_names,
                seasonal_period=1,
            ),
        }
    if source == "m6":
        calibration_scored = m6_validation_scored_frame(
            train_table,
            scored,
            model_names=model_names,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
        )
        return {
            "primary_metric": "rank_probability_score",
            "objective": "M6-style rank-probability scoring plus deterministic decisions",
            "m6": m6_rps_artifact(
                scored,
                model_names=model_names,
                calibration_scored=calibration_scored,
            ),
        }
    return {}


def m5_wrmsse_artifact(
    table: Any,
    scored: Any,
    *,
    model_names: list[str],
    seasonal_period: int,
) -> dict[str, Any]:
    pl = require_polars()
    cutoff = scored.select(pl.col("timestamp").min()).item()
    train = table.filter(pl.col("date") < cutoff)
    if train.is_empty():
        raise ValueError("M5 WRMSSE artifact requires pre-origin training rows")

    metadata = train.select("lane_id", *STATIC_COVARIATES).unique(subset=["lane_id"])
    actual = scored.select("series_id", "timestamp", "horizon", "actual").unique()
    levels = [
        ("total", None),
        ("state", "pickup_borough_code"),
        ("store", "pickup_zone"),
        ("item", "dropoff_zone"),
        ("item_store", None),
    ]
    level_artifacts = {
        level_name: m5_wrmsse_level_artifact(
            train,
            actual,
            scored,
            metadata,
            level_name=level_name,
            group_column=group_column,
            model_names=model_names,
            seasonal_period=seasonal_period,
        )
        for level_name, group_column in levels
    }
    model_scores: dict[str, float] = {}
    for model in model_names:
        available_scores = [
            level["models"][model]["wrmsse"]
            for level in level_artifacts.values()
            if model in level["models"] and level["models"][model]["wrmsse"] is not None
        ]
        model_scores[model] = float(np.mean(available_scores)) if available_scores else None
    ranking = sorted(
        model_scores,
        key=lambda name: (
            math.inf if model_scores[name] is None else model_scores[name],
            name,
        ),
    )
    return {
        "seasonal_period": int(seasonal_period),
        "level_order": list(level_artifacts),
        "levels": level_artifacts,
        "model_scores": model_scores,
        "ranking": ranking,
        "notes": [
            "Weights use recent unit-sales volume from the benchmark panel because sell prices "
            "are not present in the shared forecasting frame.",
            "Flat zero-scale series are reported under skipped_zero_scale_series rather than "
            "assigned an artificial denominator.",
        ],
    }


def m5_wrmsse_level_artifact(
    train: Any,
    actual: Any,
    scored: Any,
    metadata: Any,
    *,
    level_name: str,
    group_column: str | None,
    model_names: list[str],
    seasonal_period: int,
) -> dict[str, Any]:
    pl = require_polars()
    train_level = m5_level_frame(train, level_name=level_name, group_column=group_column)
    actual_level = m5_level_scored_frame(
        actual,
        metadata,
        level_name=level_name,
        group_column=group_column,
        value_column="actual",
    )
    ids = sorted(actual_level.select("level_id").unique().to_series().to_list())
    train_rows = {
        str(row["level_id"]): row["loads"]
        for row in train_level.sort(["level_id", "date"])
        .group_by("level_id", maintain_order=True)
        .agg(pl.col("loads"))
        .iter_rows(named=True)
    }
    actual_rows = {
        str(row["level_id"]): row["actual"]
        for row in actual_level.sort(["level_id", "horizon"])
        .group_by("level_id", maintain_order=True)
        .agg(pl.col("actual"))
        .iter_rows(named=True)
    }
    weights = m5_level_weights(train_level, ids)
    valid_ids: list[str] = []
    skipped: list[str] = []
    for level_id in ids:
        try:
            rmsse_scale(train_rows[str(level_id)], seasonal_period=seasonal_period)
        except ValueError:
            skipped.append(str(level_id))
        else:
            valid_ids.append(str(level_id))

    model_artifacts: dict[str, Any] = {}
    for model in model_names:
        pred_level = m5_level_scored_frame(
            scored.select("series_id", "timestamp", "horizon", pl.col(model).alias("forecast")),
            metadata,
            level_name=level_name,
            group_column=group_column,
            value_column="forecast",
        )
        pred_rows = {
            str(row["level_id"]): row["forecast"]
            for row in pred_level.sort(["level_id", "horizon"])
            .group_by("level_id", maintain_order=True)
            .agg(pl.col("forecast"))
            .iter_rows(named=True)
        }
        if valid_ids:
            result = wrmsse(
                [train_rows[level_id] for level_id in valid_ids],
                [actual_rows[level_id] for level_id in valid_ids],
                [pred_rows[level_id] for level_id in valid_ids],
                [weights[level_id] for level_id in valid_ids],
                seasonal_period=seasonal_period,
                series_ids=valid_ids,
                return_breakdown=True,
            )
            model_artifacts[model] = {
                "wrmsse": float(result["wrmsse"]),
                "series_count": len(valid_ids),
                "series": result["series"],
            }
        else:
            model_artifacts[model] = {
                "wrmsse": None,
                "series_count": 0,
                "series": [],
            }
    return {
        "series_count": len(ids),
        "scored_series_count": len(valid_ids),
        "skipped_zero_scale_series": skipped,
        "models": model_artifacts,
    }


def m5_level_frame(train: Any, *, level_name: str, group_column: str | None) -> Any:
    pl = require_polars()
    if level_name == "total":
        return (
            train.with_columns(pl.lit("total").alias("level_id"))
            .group_by(["level_id", "date"], maintain_order=True)
            .agg(pl.col("loads").sum())
        )
    if level_name == "item_store":
        return train.with_columns(pl.col("lane_id").cast(pl.Utf8).alias("level_id")).select(
            "level_id", "date", "loads"
        )
    if group_column is None:
        raise ValueError(f"M5 level {level_name!r} requires a group column")
    return (
        train.with_columns(pl.col(group_column).cast(pl.Utf8).alias("level_id"))
        .group_by(["level_id", "date"], maintain_order=True)
        .agg(pl.col("loads").sum())
    )


def m5_level_scored_frame(
    frame: Any,
    metadata: Any,
    *,
    level_name: str,
    group_column: str | None,
    value_column: str,
) -> Any:
    pl = require_polars()
    if level_name == "total":
        return (
            frame.with_columns(pl.lit("total").alias("level_id"))
            .group_by(["level_id", "timestamp", "horizon"], maintain_order=True)
            .agg(pl.col(value_column).sum())
        )
    joined = frame.join(metadata, left_on="series_id", right_on="lane_id", how="left")
    if level_name == "item_store":
        return joined.with_columns(pl.col("series_id").cast(pl.Utf8).alias("level_id")).select(
            "level_id", "timestamp", "horizon", value_column
        )
    if group_column is None:
        raise ValueError(f"M5 level {level_name!r} requires a group column")
    return (
        joined.with_columns(pl.col(group_column).cast(pl.Utf8).alias("level_id"))
        .group_by(["level_id", "timestamp", "horizon"], maintain_order=True)
        .agg(pl.col(value_column).sum())
    )


def m5_level_weights(train_level: Any, ids: list[Any]) -> dict[str, float]:
    pl = require_polars()
    weight_rows = (
        train_level.sort(["level_id", "date"])
        .group_by("level_id", maintain_order=True)
        .agg(pl.col("loads").tail(28).sum().alias("weight"))
    )
    weights = {
        str(row["level_id"]): float(row["weight"]) for row in weight_rows.iter_rows(named=True)
    }
    if sum(max(weights.get(str(level_id), 0.0), 0.0) for level_id in ids) <= 0.0:
        return {str(level_id): 1.0 for level_id in ids}
    return {str(level_id): max(weights.get(str(level_id), 0.0), 0.0) for level_id in ids}


def m6_validation_scored_frame(
    table: Any,
    scored: Any,
    *,
    model_names: list[str],
    season_length: int,
    cartoboost_config: dict[str, Any] | None,
) -> Any | None:
    if cartoboost_config is None:
        return None
    pl = require_polars()
    cutoff = scored.select(pl.col("timestamp").min()).item()
    horizon = int(scored.select(pl.col("horizon").max()).item())
    pre_holdout = table.filter(pl.col("date") < cutoff)
    timestamps = pre_holdout.select(pl.col("date").unique().sort()).to_series().to_list()
    if len(timestamps) < horizon * 2:
        return None

    validation_timestamps = timestamps[-horizon:]
    validation_start = validation_timestamps[0]
    validation_train = pre_holdout.filter(pl.col("date") < validation_start)
    validation_test = pre_holdout.filter(pl.col("date").is_in(validation_timestamps))
    if validation_train.is_empty() or validation_test.is_empty():
        return None
    try:
        validation_raw, _timing = forecast_model_roster(
            validation_train,
            horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            source="m6",
        )
        validation_predictions, _selection_timing = apply_shared_candidate_selection(
            validation_train,
            horizon,
            season_length=season_length,
            source="m6",
            raw_predictions=validation_raw,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
        )
    except Exception:
        return None

    actual = (
        validation_test.sort(["lane_id", "date"])
        .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
        .select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
            "horizon",
            pl.col("loads").alias("actual"),
        )
    )
    validation_scored = actual.join(
        validation_predictions,
        on=["series_id", "timestamp", "horizon"],
        how="inner",
    )
    return None if validation_scored.is_empty() else validation_scored


def m6_rps_artifact(
    scored: Any,
    *,
    model_names: list[str],
    calibration_scored: Any | None = None,
) -> dict[str, Any]:
    summaries = {
        model: m6_model_rps_summary(
            scored,
            prediction_col=model,
            calibration_scored=calibration_scored,
        )
        for model in model_names
    }
    ranking = sorted(model_names, key=lambda name: (summaries[name]["mean_rps"], name))
    return {
        "rank_bucket_count": 5,
        "models": summaries,
        "ranking": ranking,
        "notes": [
            "Rank probabilities are deterministic five-bucket distributions derived from "
            "predicted cumulative holdout returns.",
            "Decision rows are deterministic long/short selections for auditability; they "
            "are not an official M6 submission file.",
        ],
    }


def m6_model_rps_summary(
    scored: Any,
    *,
    prediction_col: str,
    calibration_scored: Any | None = None,
) -> dict[str, Any]:
    pl = require_polars()
    returns = (
        scored.group_by("series_id", maintain_order=True)
        .agg(
            pl.col("actual").sum().alias("actual_return"),
            pl.col(prediction_col).sum().alias("predicted_return"),
        )
        .sort("series_id")
    )
    actual_values = returns["actual_return"].to_list()
    predicted_values = returns["predicted_return"].to_list()
    actual_buckets = rank_buckets(actual_values, bucket_count=5)
    predicted_buckets = rank_buckets(predicted_values, bucket_count=5)
    rows = []
    rps_values = []
    calibration = m6_calibration_from_scored(
        calibration_scored,
        prediction_col=prediction_col,
        bucket_count=5,
    )
    for idx, row in enumerate(returns.iter_rows(named=True)):
        probabilities = calibrated_rank_bucket_probabilities(
            predicted_buckets[idx],
            bucket_count=5,
            calibration=calibration,
        )
        rps = rank_probability_score(probabilities, actual_buckets[idx])
        rps_values.append(rps)
        rows.append(
            {
                "series_id": str(row["series_id"]),
                "actual_return": float(row["actual_return"]),
                "predicted_return": float(row["predicted_return"]),
                "observed_rank_bucket": int(actual_buckets[idx]),
                "predicted_rank_bucket": int(predicted_buckets[idx]),
                "rank_probabilities": probabilities,
                "rps": float(rps),
            }
        )
    decisions = m6_decision_rows(rows)
    return {
        "mean_rps": float(np.mean(rps_values)) if rps_values else float("nan"),
        "asset_count": len(rows),
        "rank_probability_calibration": calibration["metadata"],
        "assets": rows,
        "decisions": decisions,
        "decision_return": float(sum(row["actual_return"] * row["weight"] for row in decisions)),
    }


def m6_calibration_from_scored(
    calibration_scored: Any | None,
    *,
    prediction_col: str,
    bucket_count: int,
) -> dict[str, Any]:
    if calibration_scored is None or prediction_col not in calibration_scored.columns:
        return m6_rank_probability_calibration(
            [],
            [],
            bucket_count=bucket_count,
            validation_support=0,
        )
    pl = require_polars()
    returns = (
        calibration_scored.group_by("series_id", maintain_order=True)
        .agg(
            pl.col("actual").sum().alias("actual_return"),
            pl.col(prediction_col).sum().alias("predicted_return"),
        )
        .sort("series_id")
    )
    if returns.is_empty():
        return m6_rank_probability_calibration(
            [],
            [],
            bucket_count=bucket_count,
            validation_support=0,
        )
    actual_buckets = rank_buckets(returns["actual_return"].to_list(), bucket_count=bucket_count)
    predicted_buckets = rank_buckets(
        returns["predicted_return"].to_list(),
        bucket_count=bucket_count,
    )
    return m6_rank_probability_calibration(
        actual_buckets,
        predicted_buckets,
        bucket_count=bucket_count,
        validation_support=len(actual_buckets),
    )


def rank_buckets(values: list[float], *, bucket_count: int) -> list[int]:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    buckets = [0] * len(values)
    for rank, idx in enumerate(order):
        buckets[idx] = min(bucket_count - 1, int(rank * bucket_count / max(len(values), 1)))
    return buckets


def rank_bucket_probabilities(predicted_bucket: int, *, bucket_count: int) -> list[float]:
    if predicted_bucket < 0 or predicted_bucket >= bucket_count:
        raise ValueError("predicted_bucket must be inside bucket_count")
    weights = []
    for bucket in range(bucket_count):
        distance = abs(bucket - predicted_bucket)
        weights.append(1.0 / (1.0 + distance * distance))
    total = sum(weights)
    return [float(weight / total) for weight in weights]


def m6_rank_probability_calibration(
    actual_buckets: list[int],
    predicted_buckets: list[int],
    *,
    bucket_count: int,
    validation_support: int,
) -> dict[str, Any]:
    if len(actual_buckets) != len(predicted_buckets):
        raise ValueError("actual and predicted bucket lists must have the same length")
    prior = 1.0
    rows = [[prior for _bucket in range(bucket_count)] for _bucket in range(bucket_count)]
    for actual_bucket, predicted_bucket in zip(actual_buckets, predicted_buckets, strict=True):
        if 0 <= predicted_bucket < bucket_count and 0 <= actual_bucket < bucket_count:
            rows[predicted_bucket][actual_bucket] += 1.0
    probabilities = []
    for row in rows:
        total = sum(row)
        probabilities.append([float(value / total) for value in row])

    support = max(int(validation_support), 0)
    shrinkage = float(support / (support + bucket_count * 20.0)) if support else 0.0
    return {
        "probabilities": probabilities,
        "shrinkage": shrinkage,
        "metadata": {
            "method": "dirichlet_confusion_with_uniform_shrinkage",
            "bucket_count": bucket_count,
            "validation_support": support,
            "dirichlet_prior": prior,
            "shrinkage_to_confusion": shrinkage,
            "fallback": "none" if support else "uniform_when_no_validation_support",
        },
    }


def calibrated_rank_bucket_probabilities(
    predicted_bucket: int,
    *,
    bucket_count: int,
    calibration: dict[str, Any],
) -> list[float]:
    if predicted_bucket < 0 or predicted_bucket >= bucket_count:
        raise ValueError("predicted_bucket must be inside bucket_count")
    uniform = [1.0 / bucket_count for _bucket in range(bucket_count)]
    row = calibration["probabilities"][predicted_bucket]
    shrinkage = float(calibration["shrinkage"])
    return [
        float(shrinkage * row_value + (1.0 - shrinkage) * uniform_value)
        for row_value, uniform_value in zip(row, uniform, strict=True)
    ]


def m6_decision_rows(asset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not asset_rows:
        return []
    ordered = sorted(asset_rows, key=lambda row: (row["predicted_return"], row["series_id"]))
    side_count = max(1, len(ordered) // 5)
    shorts = ordered[:side_count]
    longs = ordered[-side_count:]
    long_weight = 0.5 / len(longs)
    short_weight = -0.5 / len(shorts)
    decisions = [
        {
            "series_id": row["series_id"],
            "side": "short",
            "weight": float(short_weight),
            "actual_return": float(row["actual_return"]),
            "predicted_return": float(row["predicted_return"]),
        }
        for row in shorts
    ]
    decisions.extend(
        {
            "series_id": row["series_id"],
            "side": "long",
            "weight": float(long_weight),
            "actual_return": float(row["actual_return"]),
            "predicted_return": float(row["predicted_return"]),
        }
        for row in longs
    )
    return sorted(decisions, key=lambda row: (row["side"], row["series_id"]))


def forecast_model_roster(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    source: str = "synthetic",
) -> tuple[Any, dict[str, Any]]:
    forecast_frames = []
    model_timing: dict[str, Any] = {}
    if "cartoboost_lag" in model_names:
        cartoboost_predictions, cartoboost_timing = cartoboost_raw_forecast(
            train,
            horizon,
            season_length=season_length,
            config=cartoboost_config,
            prediction_col="cartoboost_lag",
        )
        forecast_frames.append(cartoboost_predictions)
        model_timing["cartoboost_lag"] = cartoboost_timing
    if "cartoboost_auto_forecast" in model_names:
        auto_predictions, auto_timing = cartoboost_forecast(
            train,
            horizon,
            season_length=season_length,
            config=cartoboost_config,
            prediction_col="cartoboost_auto_forecast",
        )
        if include_autostats_candidate(source=source, season_length=season_length, horizon=horizon):
            autostats_predictions, autostats_timing = cartoboost_autostats_forecast(
                train,
                horizon,
                season_length=season_length,
                prediction_col="cartoboost_autostats_bank",
            )
            auto_predictions = auto_predictions.join(
                autostats_predictions,
                on=["series_id", "timestamp", "horizon"],
                how="inner",
            )
            auto_timing["autostats_candidate"] = autostats_timing
        if source == "m6":
            point_auto_predictions, point_auto_timing = cartoboost_forecast(
                train,
                horizon,
                season_length=season_length,
                config=cartoboost_config,
                prediction_col="cartoboost_m6_point_auto",
            )
            auto_predictions = auto_predictions.join(
                point_auto_predictions,
                on=["series_id", "timestamp", "horizon"],
                how="inner",
            )
            auto_timing["m6_point_candidate"] = point_auto_timing
        forecast_frames.append(auto_predictions)
        model_timing["cartoboost_auto_forecast"] = auto_timing
    if any(model in model_names for model in FUNCTIME_MODELS):
        functime_predictions, functime_timing = functime_forecasts(
            train,
            horizon,
            season_length=season_length,
            lightgbm_config=cartoboost_config,
        )
        forecast_frames.append(
            functime_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in FUNCTIME_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {model: timing for model, timing in functime_timing.items() if model in model_names}
        )
    if any(model in model_names for model in STATSFORECAST_MODELS):
        statsforecast_predictions, statsforecast_timing = statsforecast_forecasts(
            train,
            horizon,
            season_length=season_length,
        )
        forecast_frames.append(
            statsforecast_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in STATSFORECAST_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {
                model: timing
                for model, timing in statsforecast_timing.items()
                if model in model_names
            }
        )
    if any(model in model_names for model in PROPHET_MODELS):
        prophet_predictions, prophet_timing = prophet_forecasts(
            train,
            horizon,
            season_length=season_length,
        )
        forecast_frames.append(
            prophet_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in PROPHET_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {model: timing for model, timing in prophet_timing.items() if model in model_names}
        )
    if any(model in model_names for model in EXTERNAL_TREE_MODELS):
        tree_predictions, tree_timing = external_tree_lag_forecasts(
            train,
            horizon,
            season_length=season_length,
            config=cartoboost_config,
        )
        forecast_frames.append(
            tree_predictions.select(
                "series_id",
                "timestamp",
                "horizon",
                *[model for model in EXTERNAL_TREE_MODELS if model in model_names],
            )
        )
        model_timing.update(
            {model: timing for model, timing in tree_timing.items() if model in model_names}
        )
    predictions = combine_forecast_frames(forecast_frames)
    timing = {"models": model_timing}
    return predictions, timing


def apply_shared_candidate_selection(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    source: str,
    raw_predictions: Any,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
) -> tuple[Any, dict[str, Any]]:
    pl = require_polars()
    started = perf_counter()
    timestamps = train.select(pl.col("date").unique().sort()).to_series().to_list()
    if len(timestamps) <= horizon + 2:
        return raw_predictions, {
            "calibration_seconds": perf_counter() - started,
            "inner_origin_count": 0.0,
            "selected_candidates": {model: model for model in model_names},
        }

    selector_model_names = candidate_selection_model_names(model_names)
    if not selector_model_names:
        return raw_predictions, {
            "calibration_seconds": perf_counter() - started,
            "inner_origin_count": 0.0,
            "selected_candidates": {model: model for model in model_names},
            "selection_error": "no candidate-selectable models in roster",
        }

    inner_scores = shared_candidate_validation_scores(
        train,
        horizon,
        season_length=season_length,
        source=source,
        cartoboost_config=cartoboost_config,
        model_names=selector_model_names,
    )
    if not inner_scores:
        return raw_predictions, {
            "calibration_seconds": perf_counter() - started,
            "inner_origin_count": 0.0,
            "selected_candidates": {model: model for model in model_names},
            "selection_error": "inner forecast roster failed or produced no aligned rows",
        }

    selected: dict[str, str] = {}
    inner_losses: dict[str, dict[str, float]] = {}
    candidate_losses_by_model: dict[str, dict[str, float]] = {}
    objective = auto_selection_objective(source)
    for model in model_names:
        if model != "cartoboost_auto_forecast":
            selected[model] = model
            if model in inner_scores:
                raw_loss = float(np.mean(inner_scores[model]))
                inner_losses[model] = {
                    "raw": raw_loss,
                    "selected": raw_loss,
                }
                candidate_losses_by_model[model] = {model: raw_loss}
            continue
        eligible_candidates = selectable_candidate_names(model, source=source)
        candidate_scores = {
            candidate: float(np.mean(losses))
            for candidate, losses in inner_scores.items()
            if candidate == model or candidate in eligible_candidates
        }
        if model not in candidate_scores:
            candidate_scores[model] = math.inf
        base_loss = candidate_scores[model]
        best_candidate = candidate_choice_for_source(candidate_scores, source=source)
        lag_loss = candidate_scores.get("cartoboost_lag", math.inf)
        if (
            requires_lag_spine(source=source, season_length=season_length, horizon=horizon)
            and math.isfinite(lag_loss)
            and math.isfinite(candidate_scores[best_candidate])
            and lag_loss <= candidate_scores[best_candidate] * 1.25
        ):
            best_candidate = "cartoboost_lag"
        best_loss = candidate_scores[best_candidate]
        if (
            best_candidate != model
            and source not in {"m5", "m6"}
            and not (source in {"synthetic", "m4"} and best_candidate == "cartoboost_lag")
            and base_loss > 0.0
            and best_loss < base_loss
            and 1.0 - best_loss / base_loss < 0.01
        ):
            best_candidate = model
        selected[model] = best_candidate
        inner_losses[model] = {
            "raw": base_loss,
            "selected": candidate_scores[best_candidate],
        }
        candidate_losses_by_model[model] = dict(sorted(candidate_scores.items()))

    outer_candidates = add_shared_candidate_columns(
        train,
        horizon,
        season_length=season_length,
        predictions=raw_predictions,
        source=source,
    )
    selected_columns = [
        pl.col(selected[model]).alias(model)
        if selected[model] in selectable_candidate_names(model, source=source)
        else pl.col(model)
        for model in model_names
    ]
    selected_predictions = outer_candidates.select(
        "series_id",
        "timestamp",
        "horizon",
        *selected_columns,
    )
    return selected_predictions, {
        "calibration_seconds": perf_counter() - started,
        "inner_origin_count": float(max(len(losses) for losses in inner_scores.values())),
        "objective": objective,
        "selected_candidates": selected,
        "inner_losses": inner_losses,
        "inner_candidate_losses": candidate_losses_by_model,
        "inner_rmse": inner_losses if objective == "rmse" else {},
    }


def candidate_selection_model_names(model_names: list[str]) -> list[str]:
    if "cartoboost_auto_forecast" not in model_names:
        return []
    selector_models = ["cartoboost_auto_forecast"]
    if "cartoboost_lag" in model_names:
        selector_models.insert(0, "cartoboost_lag")
    return selector_models


def shared_candidate_validation_scores(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    source: str,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
) -> dict[str, list[float]]:
    pl = require_polars()
    timestamps = train.select(pl.col("date").unique().sort()).to_series().to_list()
    cutoffs = shared_candidate_validation_cutoffs(timestamps, horizon=horizon)
    objective = auto_selection_objective(source)
    scores: dict[str, list[float]] = {}
    candidate_names = sorted(
        {
            candidate
            for model in model_names
            for candidate in [model, *selectable_candidate_names(model, source=source)]
        }
    )
    for cutoff in cutoffs:
        inner_train = train.filter(pl.col("date") < cutoff)
        validation_timestamps = timestamps[
            timestamps.index(cutoff) : timestamps.index(cutoff) + horizon
        ]
        inner_test = train.filter(pl.col("date").is_in(validation_timestamps))
        if inner_train.is_empty() or inner_test.is_empty():
            continue
        try:
            inner_raw, _inner_timing = candidate_selection_forecast_roster(
                inner_train,
                horizon,
                season_length=season_length,
                cartoboost_config=cartoboost_config,
                model_names=model_names,
                source=source,
            )
        except Exception:
            continue
        actual = (
            inner_test.sort(["lane_id", "date"])
            .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
            .select(
                pl.col("lane_id").alias("series_id"),
                pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
                "horizon",
                pl.col("loads").alias("actual"),
            )
        )
        inner_candidates = add_shared_candidate_columns(
            inner_train,
            horizon,
            season_length=season_length,
            predictions=inner_raw,
            source=source,
        )
        scored = actual.join(
            inner_candidates,
            on=["series_id", "timestamp", "horizon"],
            how="inner",
        )
        if scored.is_empty():
            continue
        for candidate in candidate_names:
            if candidate not in scored.columns:
                continue
            loss = forecast_objective_loss(
                objective,
                train=inner_train,
                scored=scored,
                prediction_col=candidate,
                season_length=season_length,
            )
            if math.isfinite(loss):
                scores.setdefault(candidate, []).append(loss)
    return scores


def candidate_selection_forecast_roster(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    cartoboost_config: dict[str, Any],
    model_names: list[str],
    source: str,
) -> tuple[Any, dict[str, Any]]:
    if source != "m6" or "cartoboost_auto_forecast" not in model_names:
        return forecast_model_roster(
            train,
            horizon,
            season_length=season_length,
            cartoboost_config=cartoboost_config,
            model_names=model_names,
            source=source,
        )

    pl = require_polars()
    frames = []
    timing: dict[str, Any] = {"models": {}}
    if "cartoboost_lag" in model_names:
        lag_predictions, lag_timing = cartoboost_raw_forecast(
            train,
            horizon,
            season_length=season_length,
            config=cartoboost_config,
            prediction_col="cartoboost_lag",
        )
        frames.append(lag_predictions)
        timing["models"]["cartoboost_lag"] = lag_timing
    auto_config = cartoboost_auto_config(
        cartoboost_config,
        season_length=season_length,
        horizon=horizon,
    )
    auto_predictions, auto_timing = cartoboost_raw_forecast(
        train,
        horizon,
        season_length=season_length,
        config=auto_config,
        prediction_col="cartoboost_auto_forecast",
    )
    auto_predictions = auto_predictions.with_columns(
        pl.col("cartoboost_auto_forecast").alias("cartoboost_m6_point_auto")
    )
    frames.append(auto_predictions)
    timing["models"]["cartoboost_auto_forecast"] = {
        **auto_timing,
        "selector_mode": "raw_auto_no_nested_calibration",
    }
    return combine_forecast_frames(frames), timing


def shared_candidate_validation_cutoffs(timestamps: list[Any], *, horizon: int) -> list[Any]:
    if len(timestamps) <= horizon + 2:
        return []
    max_origins = 3
    cutoffs: list[Any] = []
    for origin in range(max_origins, 0, -1):
        index = len(timestamps) - origin * horizon
        min_train = max(14, horizon)
        if index >= min_train and index + horizon <= len(timestamps):
            cutoffs.append(timestamps[index])
    return cutoffs


def selectable_candidate_names(model: str, *, source: str) -> list[str]:
    candidates = shared_candidate_names()
    if model == "cartoboost_auto_forecast":
        candidates = ["cartoboost_lag", *candidates]
    if source == "m4" and model == "cartoboost_auto_forecast":
        candidates = [*candidates, "cartoboost_autostats_bank"]
    if source == "m5" and model == "cartoboost_auto_forecast":
        candidates = [
            *candidates,
            "cartoboost_autostats_bank",
            "shared_m5_calendar_autostats_blend",
            "shared_m5_phase14_total_reconciled_020",
            "shared_m5_phase14_total_reconciled_035",
            "shared_m5_phase14_total_reconciled_050",
            "shared_m5_total_reconciled_auto",
            "shared_m5_state_reconciled_auto",
            "shared_m5_store_reconciled_auto",
        ]
    if source == "m6" and model == "cartoboost_auto_forecast":
        candidates = [*candidates, "cartoboost_m6_point_auto"]
    return candidates


def robust_candidate_choice(candidate_scores: dict[str, float]) -> str:
    finite_scores = {
        candidate: loss
        for candidate, loss in candidate_scores.items()
        if math.isfinite(loss) and loss >= 0.0
    }
    if not finite_scores:
        return min(candidate_scores, key=candidate_scores.__getitem__)
    best_loss = min(finite_scores.values())
    tolerance = best_loss * (1.0 + AUTO_SELECTION_ROBUST_RELATIVE_TOLERANCE)
    close_candidates = {
        candidate: loss for candidate, loss in finite_scores.items() if loss <= tolerance
    }
    return min(
        close_candidates,
        key=lambda candidate: (candidate_complexity_rank(candidate), close_candidates[candidate]),
    )


def candidate_choice_for_source(candidate_scores: dict[str, float], *, source: str) -> str:
    if source in {"m5", "m6"}:
        finite_scores = {
            candidate: loss
            for candidate, loss in candidate_scores.items()
            if math.isfinite(loss) and loss >= 0.0
        }
        if finite_scores:
            return min(finite_scores, key=lambda candidate: (finite_scores[candidate], candidate))
    return robust_candidate_choice(candidate_scores)


def candidate_complexity_rank(candidate: str) -> int:
    ranks = {
        "cartoboost_lag": 0,
        "cartoboost_autostats_bank": 1,
        "shared_seasonal_base": 2,
        "shared_half_drift": 3,
        "shared_drift": 4,
        "shared_seasonal_drift": 5,
        "shared_seasonal_cycle_drift_050": 6,
        "shared_seasonal_cycle_drift_075": 7,
        "cartoboost_auto_forecast": 8,
        AUTO_ENSEMBLE_CANDIDATE: 9,
        "shared_calendar_dom": 10,
        "shared_calendar_phase14": 11,
        "shared_m5_calendar_autostats_blend": 12,
        "shared_m5_phase14_total_reconciled_020": 13,
        "shared_m5_phase14_total_reconciled_035": 14,
        "shared_m5_phase14_total_reconciled_050": 15,
        "shared_m5_total_reconciled_auto": 16,
        "shared_m5_state_reconciled_auto": 17,
        "shared_m5_store_reconciled_auto": 18,
        "cartoboost_m6_point_auto": 19,
    }
    return ranks.get(candidate, 20)


def include_autostats_candidate(*, source: str, season_length: int, horizon: int) -> bool:
    if source == "m5":
        return True
    return source == "m4" and season_length in {1, 4, 12} and horizon <= 24


def m4_requires_lag_spine(*, season_length: int, horizon: int) -> bool:
    return season_length in {12, 24} or (season_length == 1 and horizon > 6)


def requires_lag_spine(*, source: str, season_length: int, horizon: int) -> bool:
    if source == "synthetic":
        return True
    if source == "m4":
        return m4_requires_lag_spine(season_length=season_length, horizon=horizon)
    return False


def shared_candidate_names() -> list[str]:
    return [
        "shared_seasonal_base",
        "shared_calendar_dom",
        "shared_calendar_phase14",
        "shared_drift",
        "shared_half_drift",
        "shared_seasonal_drift",
        "shared_seasonal_cycle_drift_050",
        "shared_seasonal_cycle_drift_075",
    ]


def add_shared_candidate_columns(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    predictions: Any,
    source: str,
) -> Any:
    pl = require_polars()
    shared_frames = [
        seasonal_naive_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col="shared_seasonal_base",
        ),
        calendar_profile_forecast_frame(
            train,
            horizon,
            prediction_col="shared_calendar_dom",
            mode="day_of_month",
        ),
        calendar_profile_forecast_frame(
            train,
            horizon,
            prediction_col="shared_calendar_phase14",
            mode="phase14",
        ),
        trend_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col="shared_drift",
            mode="drift",
        ),
        trend_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col="shared_half_drift",
            mode="half_drift",
        ),
        trend_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col="shared_seasonal_drift",
            mode="seasonal_drift",
        ),
        trend_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col="shared_seasonal_cycle_drift_050",
            mode="seasonal_cycle_drift_050",
        ),
        trend_forecast_frame(
            train,
            horizon,
            season_length=season_length,
            prediction_col="shared_seasonal_cycle_drift_075",
            mode="seasonal_cycle_drift_075",
        ),
    ]
    combined = predictions
    for frame in shared_frames:
        combined = combined.join(frame, on=["series_id", "timestamp", "horizon"], how="inner")
    if source == "m5" and "cartoboost_auto_forecast" in combined.columns:
        if "cartoboost_autostats_bank" in combined.columns:
            combined = combined.with_columns(
                (
                    0.35 * pl.col("cartoboost_auto_forecast")
                    + 0.50 * pl.col("shared_calendar_phase14")
                    + 0.15 * pl.col("cartoboost_autostats_bank")
                ).alias("shared_m5_calendar_autostats_blend")
            )
            combined = add_m5_phase14_total_reconciled_candidates(
                combined,
                base_col="shared_m5_calendar_autostats_blend",
                target_col="shared_calendar_phase14",
            )
        for group_column, prediction_col in [
            (None, "shared_m5_total_reconciled_auto"),
            ("pickup_borough_code", "shared_m5_state_reconciled_auto"),
            ("pickup_zone", "shared_m5_store_reconciled_auto"),
        ]:
            reconciled = m5_autostats_reconciled_forecast_frame(
                train,
                horizon,
                season_length=season_length,
                predictions=combined,
                group_column=group_column,
                base_col="cartoboost_auto_forecast",
                prediction_col=prediction_col,
            )
            combined = combined.join(
                reconciled,
                on=["series_id", "timestamp", "horizon"],
                how="inner",
            )
    return combined


def add_m5_phase14_total_reconciled_candidates(
    frame: Any,
    *,
    base_col: str,
    target_col: str,
) -> Any:
    pl = require_polars()
    sums = frame.group_by(["timestamp", "horizon"], maintain_order=True).agg(
        pl.col(base_col).sum().alias("__m5_phase_base_sum"),
        pl.col(target_col).sum().alias("__m5_phase_target_sum"),
    )
    reconciled_col = "__m5_phase_total_reconciled"
    with_reconciled = (
        frame.join(sums, on=["timestamp", "horizon"], how="left")
        .with_columns(
            pl.when(pl.col("__m5_phase_base_sum").abs() > 1.0e-12)
            .then(
                pl.col(base_col) * (pl.col("__m5_phase_target_sum") / pl.col("__m5_phase_base_sum"))
            )
            .otherwise(pl.col(base_col))
            .alias(reconciled_col)
        )
        .with_columns(
            *[
                ((1.0 - gamma) * pl.col(base_col) + gamma * pl.col(reconciled_col)).alias(
                    f"shared_m5_phase14_total_reconciled_{suffix}"
                )
                for gamma, suffix in [(0.20, "020"), (0.35, "035"), (0.50, "050")]
            ]
        )
    )
    return with_reconciled.drop("__m5_phase_base_sum", "__m5_phase_target_sum", reconciled_col)


def m5_autostats_reconciled_forecast_frame(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    predictions: Any,
    group_column: str | None,
    base_col: str,
    prediction_col: str,
) -> Any:
    pl = require_polars()
    group_key = "__m5_reconcile_group"
    if group_column is None:
        grouped_train = train.with_columns(pl.lit("total").alias(group_key))
    else:
        grouped_train = train.with_columns(pl.col(group_column).cast(pl.Utf8).alias(group_key))
    aggregate_train = (
        grouped_train.group_by([group_key, "date"], maintain_order=True)
        .agg(pl.col("loads").sum())
        .rename({group_key: "lane_id"})
        .sort(["lane_id", "date"])
    )
    aggregate_target, _timing = cartoboost_autostats_forecast(
        aggregate_train,
        horizon,
        season_length=season_length,
        prediction_col="__m5_reconcile_target",
    )
    aggregate_target = aggregate_target.with_columns(
        pl.col("series_id").cast(pl.Utf8).alias(group_key),
        pl.max_horizontal(pl.col("__m5_reconcile_target"), pl.lit(0.0)).alias(
            "__m5_reconcile_target"
        ),
    ).select(group_key, "timestamp", "horizon", "__m5_reconcile_target")

    metadata = train.select("lane_id", *STATIC_COVARIATES).unique(subset=["lane_id"])
    base = predictions.select("series_id", "timestamp", "horizon", base_col)
    if group_column is None:
        bottom = base.with_columns(pl.lit("total").alias(group_key))
    else:
        bottom = (
            base.join(metadata, left_on="series_id", right_on="lane_id", how="left")
            .with_columns(pl.col(group_column).cast(pl.Utf8).alias(group_key))
            .select("series_id", "timestamp", "horizon", base_col, group_key)
        )
    group_sums = bottom.group_by([group_key, "timestamp", "horizon"], maintain_order=True).agg(
        pl.col(base_col).sum().alias("__m5_reconcile_base_sum")
    )
    return (
        bottom.join(group_sums, on=[group_key, "timestamp", "horizon"], how="left")
        .join(aggregate_target, on=[group_key, "timestamp", "horizon"], how="left")
        .with_columns(
            pl.when(
                pl.col("__m5_reconcile_target").is_not_null()
                & (pl.col("__m5_reconcile_base_sum").abs() > 1.0e-12)
            )
            .then(pl.col("__m5_reconcile_target") / pl.col("__m5_reconcile_base_sum"))
            .otherwise(1.0)
            .alias("__m5_reconcile_scale")
        )
        .with_columns(
            pl.max_horizontal(
                pl.col(base_col) * pl.col("__m5_reconcile_scale"),
                pl.lit(0.0),
            ).alias(prediction_col)
        )
        .select("series_id", "timestamp", "horizon", prediction_col)
    )


def cartoboost_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    config: dict[str, Any],
    prediction_col: str = "cartoboost_auto_forecast",
) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    auto_config = cartoboost_auto_config(config, season_length=season_length, horizon=horizon)
    raw_forecast, timing = cartoboost_raw_forecast(
        train,
        horizon,
        season_length=season_length,
        config=auto_config,
        prediction_col="cartoboost_raw",
    )
    seasonal_base = seasonal_naive_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_base",
    )
    (
        selected_candidate,
        residual_alpha,
        ensemble_weights,
        calibration_timing,
    ) = calibrate_cartoboost_candidate(
        train,
        horizon,
        season_length=season_length,
        config=auto_config,
    )
    calendar_dom = calendar_profile_forecast_frame(
        train,
        horizon,
        prediction_col="cartoboost_calendar_dom",
        mode="day_of_month",
    )
    calendar_phase14 = calendar_profile_forecast_frame(
        train,
        horizon,
        prediction_col="cartoboost_calendar_phase14",
        mode="phase14",
    )
    drift = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_drift",
        mode="drift",
    )
    half_drift = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_half_drift",
        mode="half_drift",
    )
    seasonal_drift = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_drift",
        mode="seasonal_drift",
    )
    seasonal_cycle_drift_050 = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_cycle_drift_050",
        mode="seasonal_cycle_drift_050",
    )
    seasonal_cycle_drift_075 = trend_forecast_frame(
        train,
        horizon,
        season_length=season_length,
        prediction_col="cartoboost_seasonal_cycle_drift_075",
        mode="seasonal_cycle_drift_075",
    )
    candidates = (
        raw_forecast.join(seasonal_base, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(calendar_dom, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(calendar_phase14, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(half_drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_cycle_drift_050, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_cycle_drift_075, on=["series_id", "timestamp", "horizon"], how="inner")
        .with_columns(
            (
                pl.col("cartoboost_seasonal_base")
                + residual_alpha * (pl.col("cartoboost_raw") - pl.col("cartoboost_seasonal_base"))
            ).alias("cartoboost_residual_blend"),
            (
                0.5 * pl.col("cartoboost_seasonal_base") + 0.5 * pl.col("cartoboost_calendar_dom")
            ).alias("cartoboost_calendar_dom_blend"),
            (
                0.5 * pl.col("cartoboost_seasonal_base")
                + 0.5 * pl.col("cartoboost_calendar_phase14")
            ).alias("cartoboost_calendar_phase14_blend"),
        )
        .with_columns(weighted_candidate_expr(ensemble_weights).alias(AUTO_ENSEMBLE_CANDIDATE))
    )
    blended = candidates.select(
        "series_id",
        "timestamp",
        "horizon",
        pl.col(selected_candidate).alias(prediction_col),
    )
    timing = {
        **timing,
        **calibration_timing,
        "selected_candidate": selected_candidate,
        "residual_alpha": residual_alpha,
        "ensemble_weights": ensemble_weights,
        "auto_config": auto_config,
        "total_seconds": timing["total_seconds"] + calibration_timing["calibration_seconds"],
    }
    timing["fit_predict_seconds"] = timing["fit_seconds"] + timing["predict_seconds"]
    return blended, timing


def cartoboost_auto_config(
    config: dict[str, Any],
    *,
    season_length: int,
    horizon: int,
) -> dict[str, Any]:
    auto = dict(config)
    auto["n_estimators"] = max(int(config["n_estimators"]), 360)
    auto["max_depth"] = max(int(config["max_depth"]), 5)
    auto["min_samples_leaf"] = min(int(config["min_samples_leaf"]), 6)
    if horizon >= 24 or season_length in {4, 7, 12}:
        auto["max_depth"] = max(auto["max_depth"], 6)
        auto["min_samples_leaf"] = min(auto["min_samples_leaf"], 4)
    return auto


def calibrate_cartoboost_candidate(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    config: dict[str, Any],
) -> tuple[str, float, dict[str, float], dict[str, Any]]:
    pl = require_polars()
    started = perf_counter()
    timestamps = train.select(pl.col("date").unique().sort()).to_series().to_list()
    if len(timestamps) <= max(horizon + 14, 60):
        return (
            "cartoboost_raw",
            1.0,
            {"cartoboost_raw": 1.0},
            {
                "calibration_seconds": perf_counter() - started,
                "inner_origin_count": 0.0,
                "inner_base_rmse": 0.0,
                "inner_raw_rmse": 0.0,
                "inner_blended_rmse": 0.0,
                "inner_raw_relative_rmse_gain": 0.0,
            },
        )
    cutoff = timestamps[-horizon]
    inner_train = train.filter(pl.col("date") < cutoff)
    inner_test = train.filter(pl.col("date") >= cutoff)
    try:
        raw, _timing = cartoboost_raw_forecast(
            inner_train,
            horizon,
            season_length=season_length,
            config=config,
            prediction_col="cartoboost_raw",
        )
        base = seasonal_naive_forecast_frame(
            inner_train,
            horizon,
            season_length=season_length,
            prediction_col="cartoboost_seasonal_base",
        )
        calendar_dom = calendar_profile_forecast_frame(
            inner_train,
            horizon,
            prediction_col="cartoboost_calendar_dom",
            mode="day_of_month",
        )
        calendar_phase14 = calendar_profile_forecast_frame(
            inner_train,
            horizon,
            prediction_col="cartoboost_calendar_phase14",
            mode="phase14",
        )
        drift = trend_forecast_frame(
            inner_train,
            horizon,
            season_length=season_length,
            prediction_col="cartoboost_drift",
            mode="drift",
        )
        half_drift = trend_forecast_frame(
            inner_train,
            horizon,
            season_length=season_length,
            prediction_col="cartoboost_half_drift",
            mode="half_drift",
        )
        seasonal_drift = trend_forecast_frame(
            inner_train,
            horizon,
            season_length=season_length,
            prediction_col="cartoboost_seasonal_drift",
            mode="seasonal_drift",
        )
        seasonal_cycle_drift_050 = trend_forecast_frame(
            inner_train,
            horizon,
            season_length=season_length,
            prediction_col="cartoboost_seasonal_cycle_drift_050",
            mode="seasonal_cycle_drift_050",
        )
        seasonal_cycle_drift_075 = trend_forecast_frame(
            inner_train,
            horizon,
            season_length=season_length,
            prediction_col="cartoboost_seasonal_cycle_drift_075",
            mode="seasonal_cycle_drift_075",
        )
    except Exception:
        return (
            "cartoboost_raw",
            1.0,
            {"cartoboost_raw": 1.0},
            {
                "calibration_seconds": perf_counter() - started,
                "inner_origin_count": 1.0,
                "inner_base_rmse": 0.0,
                "inner_raw_rmse": 0.0,
                "inner_blended_rmse": 0.0,
                "inner_raw_relative_rmse_gain": 0.0,
            },
        )
    actual = (
        inner_test.sort(["lane_id", "date"])
        .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
        .select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
            "horizon",
            pl.col("loads").alias("actual"),
        )
    )
    scored = (
        actual.join(raw, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(base, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(calendar_dom, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(calendar_phase14, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(half_drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_drift, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_cycle_drift_050, on=["series_id", "timestamp", "horizon"], how="inner")
        .join(seasonal_cycle_drift_075, on=["series_id", "timestamp", "horizon"], how="inner")
        .with_columns(
            (
                0.5 * pl.col("cartoboost_seasonal_base") + 0.5 * pl.col("cartoboost_calendar_dom")
            ).alias("cartoboost_calendar_dom_blend"),
            (
                0.5 * pl.col("cartoboost_seasonal_base")
                + 0.5 * pl.col("cartoboost_calendar_phase14")
            ).alias("cartoboost_calendar_phase14_blend"),
        )
        .select(
            "actual",
            "cartoboost_raw",
            "cartoboost_seasonal_base",
            "cartoboost_calendar_dom",
            "cartoboost_calendar_phase14",
            "cartoboost_calendar_dom_blend",
            "cartoboost_calendar_phase14_blend",
            "cartoboost_drift",
            "cartoboost_half_drift",
            "cartoboost_seasonal_drift",
            "cartoboost_seasonal_cycle_drift_050",
            "cartoboost_seasonal_cycle_drift_075",
        )
    )
    if scored.is_empty():
        return (
            "cartoboost_raw",
            1.0,
            {"cartoboost_raw": 1.0},
            {
                "calibration_seconds": perf_counter() - started,
                "inner_origin_count": 1.0,
                "inner_base_rmse": 0.0,
                "inner_raw_rmse": 0.0,
                "inner_blended_rmse": 0.0,
                "inner_raw_relative_rmse_gain": 0.0,
            },
        )
    base_rmse = rmse_expr(scored, "cartoboost_seasonal_base")
    raw_rmse = rmse_expr(scored, "cartoboost_raw")
    best_alpha = 1.0
    best_rmse = raw_rmse
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        candidate = scored.with_columns(
            (
                pl.col("cartoboost_seasonal_base")
                + alpha * (pl.col("cartoboost_raw") - pl.col("cartoboost_seasonal_base"))
            ).alias("candidate")
        )
        candidate_rmse = rmse_expr(candidate, "candidate")
        if candidate_rmse < best_rmse:
            best_rmse = candidate_rmse
            best_alpha = alpha
    raw_gain = 1.0 - raw_rmse / base_rmse if base_rmse > 0.0 else 0.0
    blended_gain = 1.0 - best_rmse / raw_rmse if raw_rmse > 0.0 else 0.0
    if blended_gain < AUTO_SELECTION_MIN_RELATIVE_GAIN:
        best_alpha = 1.0
        best_rmse = raw_rmse
    scored = scored.with_columns(
        (
            pl.col("cartoboost_seasonal_base")
            + best_alpha * (pl.col("cartoboost_raw") - pl.col("cartoboost_seasonal_base"))
        ).alias("cartoboost_residual_blend")
    )
    candidate_scores = {
        "cartoboost_raw": raw_rmse,
        "cartoboost_seasonal_base": base_rmse,
        "cartoboost_residual_blend": best_rmse,
        "cartoboost_calendar_dom": rmse_expr(scored, "cartoboost_calendar_dom"),
        "cartoboost_calendar_phase14": rmse_expr(scored, "cartoboost_calendar_phase14"),
        "cartoboost_calendar_dom_blend": rmse_expr(scored, "cartoboost_calendar_dom_blend"),
        "cartoboost_calendar_phase14_blend": rmse_expr(
            scored,
            "cartoboost_calendar_phase14_blend",
        ),
        "cartoboost_drift": rmse_expr(scored, "cartoboost_drift"),
        "cartoboost_half_drift": rmse_expr(scored, "cartoboost_half_drift"),
        "cartoboost_seasonal_drift": rmse_expr(scored, "cartoboost_seasonal_drift"),
        "cartoboost_seasonal_cycle_drift_050": rmse_expr(
            scored,
            "cartoboost_seasonal_cycle_drift_050",
        ),
        "cartoboost_seasonal_cycle_drift_075": rmse_expr(
            scored,
            "cartoboost_seasonal_cycle_drift_075",
        ),
    }
    ensemble_weights = validation_ensemble_weights(candidate_scores)
    scored = scored.with_columns(
        weighted_candidate_expr(ensemble_weights).alias(AUTO_ENSEMBLE_CANDIDATE)
    )
    candidate_scores[AUTO_ENSEMBLE_CANDIDATE] = rmse_expr(scored, AUTO_ENSEMBLE_CANDIDATE)
    selected_candidate = min(candidate_scores, key=candidate_scores.__getitem__)
    selected_gain = 1.0 - candidate_scores[selected_candidate] / raw_rmse if raw_rmse > 0.0 else 0.0
    if selected_candidate != "cartoboost_raw" and selected_gain < AUTO_SELECTION_MIN_RELATIVE_GAIN:
        selected_candidate = "cartoboost_raw"
        ensemble_weights = {"cartoboost_raw": 1.0}
    return (
        selected_candidate,
        best_alpha,
        ensemble_weights,
        {
            "calibration_seconds": perf_counter() - started,
            "inner_origin_count": 1.0,
            "inner_base_rmse": base_rmse,
            "inner_raw_rmse": raw_rmse,
            "inner_blended_rmse": best_rmse,
            "inner_raw_relative_rmse_gain": raw_gain,
            "inner_blended_relative_rmse_gain": blended_gain,
            "inner_selected_relative_rmse_gain": selected_gain,
            "inner_validation_ensemble_rmse": candidate_scores[AUTO_ENSEMBLE_CANDIDATE],
            "inner_calendar_dom_rmse": candidate_scores["cartoboost_calendar_dom"],
            "inner_calendar_phase14_rmse": candidate_scores["cartoboost_calendar_phase14"],
            "inner_drift_rmse": candidate_scores["cartoboost_drift"],
            "inner_half_drift_rmse": candidate_scores["cartoboost_half_drift"],
            "inner_seasonal_drift_rmse": candidate_scores["cartoboost_seasonal_drift"],
            "inner_seasonal_cycle_drift_050_rmse": candidate_scores[
                "cartoboost_seasonal_cycle_drift_050"
            ],
            "inner_seasonal_cycle_drift_075_rmse": candidate_scores[
                "cartoboost_seasonal_cycle_drift_075"
            ],
        },
    )


def validation_ensemble_weights(candidate_scores: dict[str, float]) -> dict[str, float]:
    finite_scores = [
        (name, score)
        for name, score in candidate_scores.items()
        if math.isfinite(score) and score > 0.0
    ]
    if not finite_scores:
        return {"cartoboost_raw": 1.0}
    ranked = sorted(finite_scores, key=lambda item: (item[1], item[0]))[:4]
    inv = [(name, 1.0 / max(score * score, 1.0e-12)) for name, score in ranked]
    total = sum(weight for _name, weight in inv)
    if total <= 0.0 or not math.isfinite(total):
        return {ranked[0][0]: 1.0}
    return {name: weight / total for name, weight in inv}


def weighted_candidate_expr(weights: dict[str, float]) -> Any:
    pl = require_polars()
    expr = None
    for name, weight in sorted(weights.items()):
        term = float(weight) * pl.col(name)
        expr = term if expr is None else expr + term
    return expr if expr is not None else pl.col("cartoboost_raw")


def rmse_expr(frame: Any, prediction_col: str) -> float:
    pl = require_polars()
    return float(
        frame.select(((pl.col(prediction_col) - pl.col("actual")).pow(2).mean()).sqrt()).item()
    )


def cartoboost_raw_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    config: dict[str, Any],
    prediction_col: str,
) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    feature_start = perf_counter()
    model_params = cartoboost_native_forecaster_params(
        season_length,
        horizon,
        config,
        train=train,
    )
    training_frame = train.select("lane_id", "date", "loads").to_pandas()
    if not isinstance(training_frame, pd.DataFrame):
        raise TypeError("CartoBoost native benchmark training conversion did not return pandas")
    feature_seconds = perf_counter() - feature_start
    model = CartoBoostLagForecaster(
        time_col="date",
        target_col="loads",
        panel_cols=["lane_id"],
        frequency="D",
        **model_params,
    )
    fit_start = perf_counter()
    model.fit(training_frame)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    result = model.predict(horizon)
    predictions = pl.DataFrame(
        result.predictions(),
        schema=["series_id", "timestamp", "horizon", "model", prediction_col],
        orient="row",
    ).select(
        "series_id",
        pl.col("timestamp").str.to_datetime().cast(pl.Datetime("us")).alias("timestamp"),
        "horizon",
        prediction_col,
    )
    predict_seconds = perf_counter() - predict_start
    feature_count = len(model.metadata_.get("feature_names", []))
    timing = {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
        "feature_count": float(feature_count),
    }
    return predictions, timing


def cartoboost_autostats_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    prediction_col: str,
) -> tuple[Any, dict[str, Any]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    feature_start = perf_counter()
    training_frame = train.select("lane_id", "date", "loads").to_pandas()
    if not isinstance(training_frame, pd.DataFrame):
        raise TypeError("CartoBoost native benchmark training conversion did not return pandas")
    frame = ForecastFrame.from_pandas(
        training_frame,
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
        allow_irregular=True,
    )
    validation_window = max(1, min(int(horizon), 8))
    feature_seconds = perf_counter() - feature_start

    model = AutoStatsBank(
        season_length=max(int(season_length), 1),
        validation_window=validation_window,
    )
    fit_start = perf_counter()
    model.fit(frame)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    result = model.predict(horizon)
    predictions = pl.DataFrame(
        result.predictions(),
        schema=["series_id", "timestamp", "horizon", "model", prediction_col],
        orient="row",
    ).select(
        "series_id",
        pl.col("timestamp").str.to_datetime().cast(pl.Datetime("us")).alias("timestamp"),
        "horizon",
        prediction_col,
    )
    predict_seconds = perf_counter() - predict_start
    metadata = model.metadata_
    timing = {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
        "validation_window": float(validation_window),
        "metadata": metadata,
    }
    return predictions, timing


def external_tree_lag_forecasts(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    config: dict[str, Any],
) -> tuple[Any, dict[str, dict[str, float]]]:
    try:
        import lightgbm as lgb
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "external tree lag baselines require xgboost and lightgbm; run `uv sync --group bench`."
        ) from exc

    tree_params = cartoboost_tree_regularization(season_length, horizon, config)
    model_specs = {
        "xgboost_lag": xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=tree_params["max_depth"],
            min_child_weight=tree_params["min_samples_leaf"],
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
            random_state=0,
        ),
        "lightgbm_lag": lgb.LGBMRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=tree_params["max_depth"],
            min_child_samples=tree_params["min_samples_leaf"],
            verbosity=-1,
            random_state=0,
            n_jobs=1,
        ),
    }
    forecasts = []
    timings = {}
    for name, model in model_specs.items():
        forecast, timing = external_tree_lag_forecast(
            train,
            horizon,
            season_length=season_length,
            model=model,
            prediction_col=name,
        )
        forecasts.append(forecast)
        timings[name] = timing
    return combine_forecast_frames(forecasts), timings


def external_tree_lag_forecast(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    model: Any,
    prediction_col: str,
) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    feature_start = perf_counter()
    history_features = build_history_features(train, season_length=season_length)
    feature_columns = select_cartoboost_feature_columns(
        history_features,
        season_length=season_length,
    )
    feature_frame = history_features.drop_nulls(feature_columns)
    x = feature_frame.select(feature_columns).to_numpy()
    target_mode = cartoboost_target_mode(season_length, horizon)
    if target_mode == "delta_from_last":
        y = (feature_frame["loads"] - feature_frame["loads_lag_1"]).to_numpy()
    else:
        y = feature_frame.select("loads").to_numpy().ravel()
    feature_seconds = perf_counter() - feature_start

    fit_start = perf_counter()
    model.fit(x, y)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    history = train.clone()
    history_schema = history.schema
    forecast_frames = []
    for step in range(1, horizon + 1):
        future = next_future_rows(history)
        future_features = build_future_features(
            history,
            future,
            season_length=season_length,
        ).drop_nulls(feature_columns)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            raw_predictions = model.predict(future_features.select(feature_columns).to_numpy())
        if target_mode == "delta_from_last":
            predictions = raw_predictions + future_features["loads_lag_1"].to_numpy()
        else:
            predictions = raw_predictions
        step_forecast = future_features.select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").alias("timestamp"),
            pl.lit(step).alias("horizon"),
        ).with_columns(pl.Series(prediction_col, predictions))
        forecast_frames.append(step_forecast)
        predicted_future = future_features.with_columns(pl.Series(prediction_col, predictions))
        append_frame = predicted_future.select(
            pl.col("lane_id").cast(history_schema["lane_id"]),
            pl.col("date").cast(history_schema["date"]),
            pl.col(prediction_col).alias("loads").cast(history_schema["loads"]),
            *[pl.col(column).cast(history_schema[column]) for column in STATIC_COVARIATES],
        )
        history = pl.concat([history, append_frame], how="vertical")
    predict_seconds = perf_counter() - predict_start
    return pl.concat(forecast_frames, how="vertical"), {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
        "feature_count": float(len(feature_columns)),
        "target_mode_delta": float(target_mode == "delta_from_last"),
    }


def cartoboost_native_forecaster_params(
    season_length: int,
    horizon: int,
    config: dict[str, Any],
    *,
    train: Any,
) -> dict[str, Any]:
    max_lag, max_window = cartoboost_supported_history_limits(train)
    lags = [lag for lag in cartoboost_lag_values(season_length) if lag <= max_lag]
    rolling_windows = [
        window for window in cartoboost_rolling_windows(season_length) if window <= max_window
    ]
    difference_lags = [
        lag for lag in cartoboost_difference_lags(season_length) if lag <= max_lag - 1
    ]
    rolling_trend_windows = list(rolling_windows)
    if not lags:
        lags = [1]
    return {
        "lags": lags,
        "rolling_windows": rolling_windows,
        "difference_lags": difference_lags,
        "rolling_trend_windows": rolling_trend_windows,
        "calendar_features": True,
        "target_mode": cartoboost_target_mode(season_length, horizon),
        "n_estimators": config["n_estimators"],
        "learning_rate": config["learning_rate"],
        **cartoboost_tree_regularization(season_length, horizon, config),
        "splitters": config.get("splitters"),
    }


def cartoboost_benchmark_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config)
    settings["native_target_mode_policy"] = (
        "delta_from_last when season_length == 12, horizon == 13, or horizon >= 24; level otherwise"
    )
    settings["native_tree_regularization_policy"] = (
        "use at least max_depth=5 and at most min_samples_leaf=6 when "
        "horizon >= 24 or season_length == 4; otherwise use configured values"
    )
    settings["native_feature_policy"] = (
        "season-aware lags, rolling means, lag deltas, and rolling trends capped to "
        "the shortest training series"
    )
    return settings


def cartoboost_tree_regularization(
    season_length: int,
    horizon: int,
    config: dict[str, Any],
) -> dict[str, int]:
    max_depth = int(config["max_depth"])
    min_samples_leaf = int(config["min_samples_leaf"])
    if horizon >= 24 or season_length == 4:
        max_depth = max(max_depth, 5)
        min_samples_leaf = min(min_samples_leaf, 6)
    return {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }


def cartoboost_target_mode(season_length: int, horizon: int) -> str:
    if season_length == 12 or horizon == 13 or horizon >= 24:
        return "delta_from_last"
    return "level"


def cartoboost_supported_history_limits(train: Any) -> tuple[int, int]:
    pl = require_polars()
    min_history = int(train.group_by("lane_id").len().select(pl.col("len").min()).item())
    if min_history < 2:
        raise ValueError("CartoBoost lag benchmark requires at least two rows per series")
    max_lag = max(1, min_history - 1)
    return max_lag, max_lag


def seasonal_naive_forecast_frame(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    prediction_col: str,
) -> Any:
    pl = require_polars()
    history = train.clone()
    history_schema = history.schema
    forecast_frames = []
    for step in range(1, horizon + 1):
        rows = []
        for row in next_future_rows(history).iter_rows(named=True):
            lane_history = history.filter(pl.col("lane_id") == row["lane_id"]).sort("date")
            values = lane_history["loads"].to_list()
            prediction = float(
                values[-season_length] if len(values) >= season_length else values[-1]
            )
            rows.append({**row, prediction_col: prediction, "horizon": step})
        future = pl.DataFrame(rows)
        forecast_frames.append(
            future.select(
                pl.col("lane_id").alias("series_id"),
                pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
                "horizon",
                prediction_col,
            )
        )
        append_frame = future.select(
            pl.col("lane_id").cast(history_schema["lane_id"]),
            pl.col("date").cast(history_schema["date"]),
            pl.col(prediction_col).alias("loads").cast(history_schema["loads"]),
            *[pl.col(column).cast(history_schema[column]) for column in STATIC_COVARIATES],
        )
        history = pl.concat([history, append_frame], how="vertical")
    return pl.concat(forecast_frames, how="vertical")


def calendar_profile_forecast_frame(
    train: Any,
    horizon: int,
    *,
    prediction_col: str,
    mode: str,
) -> Any:
    pl = require_polars()
    history = train.clone()
    history_schema = history.schema
    forecast_frames = []
    for step in range(1, horizon + 1):
        rows = []
        for row in next_future_rows(history).iter_rows(named=True):
            lane_history = history.filter(pl.col("lane_id") == row["lane_id"]).sort("date")
            values = [float(value) for value in lane_history["loads"].to_list()]
            prediction = calendar_profile_prediction(
                lane_history,
                values,
                row["date"],
                mode=mode,
            )
            rows.append({**row, prediction_col: prediction, "horizon": step})
        future = pl.DataFrame(rows)
        forecast_frames.append(
            future.select(
                pl.col("lane_id").alias("series_id"),
                pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
                "horizon",
                prediction_col,
            )
        )
        append_frame = future.select(
            pl.col("lane_id").cast(history_schema["lane_id"]),
            pl.col("date").cast(history_schema["date"]),
            pl.col(prediction_col).alias("loads").cast(history_schema["loads"]),
            *[pl.col(column).cast(history_schema[column]) for column in STATIC_COVARIATES],
        )
        history = pl.concat([history, append_frame], how="vertical")
    return pl.concat(forecast_frames, how="vertical")


def trend_forecast_frame(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    prediction_col: str,
    mode: str,
) -> Any:
    pl = require_polars()
    history = train.clone()
    history_schema = history.schema
    forecast_frames = []
    for step in range(1, horizon + 1):
        rows = []
        for row in next_future_rows(history).iter_rows(named=True):
            lane_history = history.filter(pl.col("lane_id") == row["lane_id"]).sort("date")
            values = [float(value) for value in lane_history["loads"].to_list()]
            prediction = trend_prediction(
                values,
                step=step,
                season_length=season_length,
                mode=mode,
            )
            rows.append({**row, prediction_col: max(0.0, prediction), "horizon": step})
        future = pl.DataFrame(rows)
        forecast_frames.append(
            future.select(
                pl.col("lane_id").alias("series_id"),
                pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
                "horizon",
                prediction_col,
            )
        )
        append_frame = future.select(
            pl.col("lane_id").cast(history_schema["lane_id"]),
            pl.col("date").cast(history_schema["date"]),
            pl.col(prediction_col).alias("loads").cast(history_schema["loads"]),
            *[pl.col(column).cast(history_schema[column]) for column in STATIC_COVARIATES],
        )
        history = pl.concat([history, append_frame], how="vertical")
    return pl.concat(forecast_frames, how="vertical")


def trend_prediction(
    values: list[float],
    *,
    step: int,
    season_length: int,
    mode: str,
) -> float:
    if not values:
        raise ValueError("trend forecast requires non-empty lane history")
    if len(values) == 1:
        return values[-1]
    slope = (values[-1] - values[0]) / (len(values) - 1)
    if mode == "drift":
        return values[-1] + step * slope
    if mode == "half_drift":
        return values[-1] + 0.5 * step * slope
    if mode == "seasonal_drift":
        baseline = (
            values[-season_length]
            if season_length > 1 and len(values) >= season_length
            else values[-1]
        )
        return baseline + step * slope
    if mode.startswith("seasonal_cycle_drift_"):
        if season_length <= 1 or len(values) < 2 * season_length:
            return values[-1]
        alpha = float(mode.rsplit("_", maxsplit=1)[-1]) / 100.0
        baseline = values[-season_length]
        seasonal_slope = (values[-season_length] - values[-2 * season_length]) / season_length
        return baseline + alpha * step * seasonal_slope
    raise ValueError(f"unsupported trend forecast mode {mode!r}")


def calendar_profile_prediction(
    lane_history: Any,
    values: list[float],
    timestamp: datetime,
    *,
    mode: str,
) -> float:
    if not values:
        raise ValueError("calendar profile requires non-empty lane history")
    fallback = float(values[-7] if len(values) >= 7 else values[-1])
    if mode == "day_of_month":
        matches = lane_history.filter(lane_history["date"].dt.day() == timestamp.day)[
            "loads"
        ].to_list()
        return float(np.mean(matches)) if matches else fallback
    if mode == "phase14":
        future_phase = len(values) % 14
        matches = [value for index, value in enumerate(values) if index % 14 == future_phase]
        return float(np.mean(matches)) if matches else fallback
    raise ValueError(f"unsupported calendar profile mode {mode!r}")


def unique_positive_ints(values: list[int]) -> list[int]:
    seen = set()
    result = []
    for value in values:
        if value > 0 and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def cartoboost_lag_values(season_length: int) -> list[int]:
    seasonal_lags: list[int] = []
    if season_length > 1:
        seasonal_lags = [
            season_length - 1,
            season_length,
            season_length + 1,
            2 * season_length,
        ]
    return unique_positive_ints([*BASE_CARTOBOOST_LAGS, *seasonal_lags])


def cartoboost_rolling_windows(season_length: int) -> list[int]:
    seasonal_windows: list[int] = []
    if season_length > 1:
        seasonal_windows = [season_length, 2 * season_length]
    return unique_positive_ints([*BASE_CARTOBOOST_ROLLING_WINDOWS, *seasonal_windows])


def cartoboost_difference_lags(season_length: int) -> list[int]:
    return [lag for lag in cartoboost_lag_values(season_length) if lag > 1]


def cartoboost_target_feature_specs(season_length: int) -> list[tuple[str, int]]:
    specs = [(f"loads_lag_{lag}", lag) for lag in cartoboost_lag_values(season_length)]
    specs.extend(
        (f"loads_roll_{window}", window) for window in cartoboost_rolling_windows(season_length)
    )
    specs.extend(
        (f"loads_delta_lag_{lag}", lag) for lag in cartoboost_difference_lags(season_length)
    )
    specs.extend(
        (f"loads_roll_trend_{window}", 2 * window)
        for window in cartoboost_rolling_windows(season_length)
    )
    return specs


def select_cartoboost_feature_columns(feature_frame: Any, *, season_length: int) -> list[str]:
    target_specs = cartoboost_target_feature_specs(season_length)
    ranked_drop_candidates = sorted(target_specs, key=lambda item: item[1], reverse=True)
    for drop_count in range(len(ranked_drop_candidates) + 1):
        dropped = {name for name, _cost in ranked_drop_candidates[:drop_count]}
        target_columns = [name for name, _cost in target_specs if name not in dropped]
        columns = [*target_columns, *EXOGENOUS_FEATURE_COLUMNS]
        if feature_frame.drop_nulls(columns).height > 0:
            return columns
    raise ValueError("CartoBoost lag benchmark has no complete lag feature rows")


def build_history_features(frame: Any, *, season_length: int) -> Any:
    pl = require_polars()
    lags = cartoboost_lag_values(season_length)
    rolling_windows = cartoboost_rolling_windows(season_length)
    difference_lags = cartoboost_difference_lags(season_length)
    return frame.sort(["lane_id", "date"]).with_columns(
        *[pl.col("loads").shift(lag).over("lane_id").alias(f"loads_lag_{lag}") for lag in lags],
        *[
            pl.col("loads")
            .shift(1)
            .rolling_mean(window)
            .over("lane_id")
            .alias(f"loads_roll_{window}")
            for window in rolling_windows
        ],
        *[
            (
                pl.col("loads").shift(1).over("lane_id")
                - pl.col("loads").shift(lag).over("lane_id")
            ).alias(f"loads_delta_lag_{lag}")
            for lag in difference_lags
        ],
        *[
            (
                pl.col("loads").shift(1).rolling_mean(window).over("lane_id")
                - pl.col("loads").shift(window + 1).rolling_mean(window).over("lane_id")
            ).alias(f"loads_roll_trend_{window}")
            for window in rolling_windows
        ],
        date_dayofweek=pl.col("date").dt.weekday().cast(pl.Float64),
        date_day=pl.col("date").dt.day().cast(pl.Float64),
        date_dayofyear=pl.col("date").dt.ordinal_day().cast(pl.Float64),
        date_month=pl.col("date").dt.month().cast(pl.Float64),
        date_elapsed_days=pl.int_range(pl.len()).over("lane_id").cast(pl.Float64),
    )


def next_future_rows(history: Any) -> Any:
    pl = require_polars()
    return (
        history.sort(["lane_id", "date"])
        .group_by("lane_id", maintain_order=True)
        .tail(1)
        .with_columns((pl.col("date") + pl.duration(days=1)).alias("date"))
        .select("lane_id", "date", *STATIC_COVARIATES)
    )


def build_future_features(history: Any, future: Any, *, season_length: int) -> Any:
    pl = require_polars()
    lags = cartoboost_lag_values(season_length)
    rolling_windows = cartoboost_rolling_windows(season_length)
    difference_lags = cartoboost_difference_lags(season_length)
    pieces = []
    for row in future.iter_rows(named=True):
        lane_history = history.filter(pl.col("lane_id") == row["lane_id"]).sort("date")
        values = dict(row)
        loads = lane_history["loads"].to_list()
        for lag in lags:
            values[f"loads_lag_{lag}"] = float(loads[-lag]) if len(loads) >= lag else None
        for window in rolling_windows:
            values[f"loads_roll_{window}"] = (
                float(np.mean(loads[-window:])) if len(loads) >= window else None
            )
        for lag in difference_lags:
            values[f"loads_delta_lag_{lag}"] = (
                float(loads[-1] - loads[-lag]) if len(loads) >= lag else None
            )
        for window in rolling_windows:
            values[f"loads_roll_trend_{window}"] = (
                float(np.mean(loads[-window:]) - np.mean(loads[-2 * window : -window]))
                if len(loads) >= 2 * window
                else None
            )
        timestamp = values["date"]
        values["date_dayofweek"] = float(timestamp.weekday() + 1)
        values["date_day"] = float(timestamp.day)
        values["date_dayofyear"] = float(timestamp.timetuple().tm_yday)
        values["date_month"] = float(timestamp.month)
        values["date_elapsed_days"] = float(len(loads))
        pieces.append(values)
    return pl.DataFrame(pieces)


def external_autoreg_lag_depth(*, season_length: int, min_series_length: int) -> int:
    max_supported = max(1, min_series_length - 1)
    structural_lags = [28]
    if season_length > 1:
        structural_lags.extend([season_length, 2 * season_length])
    return min(max_supported, max(structural_lags))


def functime_forecasts(
    train: Any,
    horizon: int,
    *,
    season_length: int,
    lightgbm_config: dict[str, Any],
) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    try:
        from functime.forecasting import lightgbm, ridge, snaive
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires functime; run `uv sync --group bench`."
        ) from exc

    y = train.select(
        pl.col("lane_id").alias("entity"),
        pl.col("date").alias("time"),
        pl.col("loads").alias("target"),
    )
    min_series_length = int(train.group_by("lane_id").len().select(pl.col("len").min()).item())
    autoreg_lags = external_autoreg_lag_depth(
        season_length=season_length,
        min_series_length=min_series_length,
    )
    model_specs = {
        "functime_snaive": snaive(freq="1d", sp=season_length),
        "functime_ridge": ridge(freq="1d", lags=autoreg_lags),
        "functime_lightgbm": lightgbm(
            freq="1d",
            lags=autoreg_lags,
            n_estimators=lightgbm_config["n_estimators"],
            learning_rate=lightgbm_config["learning_rate"],
            max_depth=lightgbm_config["max_depth"],
            min_child_samples=lightgbm_config["min_samples_leaf"],
            verbosity=-1,
        ),
    }
    forecasts = []
    timings = {}
    for name, model in model_specs.items():
        fit_start = perf_counter()
        model.fit(y)
        fit_seconds = perf_counter() - fit_start
        predict_start = perf_counter()
        forecast = (
            model.predict(horizon)
            .rename({"entity": "series_id", "time": "timestamp", "target": name})
            .sort(["series_id", "timestamp"])
            .with_columns((pl.int_range(pl.len()).over("series_id") + 1).alias("horizon"))
            .select("series_id", "timestamp", "horizon", name)
        )
        predict_seconds = perf_counter() - predict_start
        forecasts.append(forecast)
        timings[name] = {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "fit_predict_seconds": fit_seconds + predict_seconds,
        }

    return combine_forecast_frames(forecasts), timings


def configure_prophet_seasonality(model: Any, *, season_length: int) -> None:
    if season_length <= 1 or season_length == 7:
        return
    fourier_order = min(10, max(3, season_length // 2))
    model.add_seasonality(
        name=f"structural_period_{season_length}",
        period=float(season_length),
        fourier_order=fourier_order,
        prior_scale=10.0,
        mode="additive",
    )


def statsforecast_forecasts(
    train: Any,
    horizon: int,
    *,
    season_length: int,
) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import (
            AutoARIMA,
            AutoCES,
            AutoETS,
            AutoTBATS,
            AutoTheta,
            DynamicOptimizedTheta,
            SeasonalNaive,
        )
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires statsforecast; run `uv sync --group bench`."
        ) from exc

    y = (
        train.select(
            pl.col("lane_id").alias("unique_id"),
            pl.col("date").alias("ds"),
            pl.col("loads").alias("y"),
        )
        .sort(["unique_id", "ds"])
        .to_pandas()
    )
    model_specs = {
        "statsforecast_seasonal_naive": SeasonalNaive(season_length=season_length),
        "statsforecast_autoets": AutoETS(season_length=season_length, model="ZZZ"),
        "statsforecast_autoarima": AutoARIMA(season_length=season_length),
        "statsforecast_autotheta": AutoTheta(season_length=season_length),
        "statsforecast_autoces": AutoCES(season_length=season_length),
        "statsforecast_dynamic_optimized_theta": DynamicOptimizedTheta(season_length=season_length),
        "statsforecast_autotbats": AutoTBATS(season_length=season_length),
    }
    forecasts = []
    timings = {}
    for name, model in model_specs.items():
        forecast_runner = StatsForecast(models=[model], freq="D", n_jobs=1)
        fit_start = perf_counter()
        forecast = forecast_runner.forecast(df=y, h=horizon)
        fit_predict_seconds = perf_counter() - fit_start
        value_columns = [column for column in forecast.columns if column not in {"unique_id", "ds"}]
        if len(value_columns) != 1:
            raise RuntimeError(
                f"StatsForecast model {name} returned forecast columns {value_columns!r}"
            )
        forecast_frame = (
            pl.from_pandas(
                forecast.rename(
                    columns={
                        "unique_id": "series_id",
                        "ds": "timestamp",
                        value_columns[0]: name,
                    }
                )
            )
            .sort(["series_id", "timestamp"])
            .with_columns((pl.int_range(pl.len()).over("series_id") + 1).alias("horizon"))
            .select("series_id", "timestamp", "horizon", name)
        )
        forecasts.append(forecast_frame)
        timings[name] = {
            "fit_seconds": fit_predict_seconds,
            "predict_seconds": 0.0,
            "fit_predict_seconds": fit_predict_seconds,
        }
    del pd
    return combine_forecast_frames(forecasts), timings


def prophet_forecasts(
    train: Any,
    horizon: int,
    *,
    season_length: int,
) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    prophet_class = ensure_prophet_class()

    fit_seconds = 0.0
    predict_seconds = 0.0
    forecast_frames = []
    grouped = train.sort(["lane_id", "date"]).group_by("lane_id", maintain_order=True)
    for series_id, group in grouped:
        series_frame = group.select(
            pl.col("date").alias("ds"),
            pl.col("loads").alias("y"),
        ).to_pandas()
        fit_start = perf_counter()
        model = prophet_class(
            weekly_seasonality=season_length == 7,
            daily_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode="additive",
            stan_backend="CMDSTANPY",
        )
        configure_prophet_seasonality(model, season_length=season_length)
        model.fit(series_frame)
        fit_seconds += perf_counter() - fit_start
        future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        predict_start = perf_counter()
        forecast = model.predict(future)
        predict_seconds += perf_counter() - predict_start
        forecast_frames.append(
            pl.from_pandas(
                pd.DataFrame(
                    {
                        "series_id": [series_id[0] if isinstance(series_id, tuple) else series_id]
                        * horizon,
                        "timestamp": forecast["ds"],
                        "horizon": np.arange(1, horizon + 1, dtype=int),
                        "prophet_additive": forecast["yhat"].to_numpy(dtype=float),
                    }
                )
            )
        )
    timing = {
        "prophet_additive": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "fit_predict_seconds": fit_seconds + predict_seconds,
        }
    }
    return pl.concat(forecast_frames, how="vertical"), timing


def evaluate_metrics(
    scored: Any,
    prediction_col: str,
    train: Any,
    *,
    season_length: int,
) -> dict[str, float]:
    pl = require_polars()
    error_frame = scored.select(
        error=pl.col(prediction_col) - pl.col("actual"),
        abs_error=(pl.col(prediction_col) - pl.col("actual")).abs(),
        actual_abs=pl.col("actual").abs(),
        smape_den=(pl.col(prediction_col).abs() + pl.col("actual").abs()),
    )
    mae = float(error_frame.select(pl.col("abs_error").mean()).item())
    rmse = float(error_frame.select((pl.col("error").pow(2).mean()).sqrt()).item())
    sse = float(error_frame.select(pl.col("error").pow(2).sum()).item())
    actual_mean = float(scored.select(pl.col("actual").mean()).item())
    sst = float(scored.select((pl.col("actual") - actual_mean).pow(2).sum()).item())
    r2 = 1.0 if sse <= 1.0e-12 else 0.0 if sst <= 1.0e-12 else 1.0 - sse / sst
    actual_abs_sum = float(error_frame.select(pl.col("actual_abs").sum()).item())
    abs_error_sum = float(error_frame.select(pl.col("abs_error").sum()).item())
    wape = 0.0 if abs_error_sum <= 1.0e-12 else abs_error_sum / max(actual_abs_sum, 1.0e-12)
    smape_value = (
        error_frame.filter(pl.col("smape_den") > 0)
        .select((2.0 * pl.col("abs_error") / pl.col("smape_den")).mean())
        .item()
    )
    smape = 0.0 if smape_value is None else float(smape_value)
    bias = float(error_frame.select(pl.col("error").mean()).item())
    train_scale = (
        train.sort(["lane_id", "date"])
        .with_columns(
            (pl.col("loads") - pl.col("loads").shift(season_length).over("lane_id"))
            .abs()
            .alias("d")
        )
        .select(pl.col("d").mean())
        .item()
    )
    mase_denom = max(float(train_scale or 0.0), 1.0e-12)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "mase": mae / mase_denom,
        "wape": wape,
        "smape": smape,
        "bias": bias,
    }


def write_forecast_plots(
    scored: Any, plot_dir: Path, *, prefix: str, models: list[str] | None = None
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "forecasting benchmark plotting requires matplotlib; run `uv sync`."
        ) from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    if models is None:
        models = benchmark_model_names("full")
    frame = scored.sort(["series_id", "timestamp"]).to_pandas()
    paths: list[Path] = []

    metric_path = plot_dir / f"{prefix}_tool_metric_comparison.png"
    rmse_values = [
        float(np.sqrt(np.mean((frame[model].to_numpy() - frame["actual"].to_numpy()) ** 2)))
        for model in models
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(models, rmse_values)
    ax.set_ylabel("RMSE")
    ax.set_title("Forecast RMSE by tool")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(metric_path, dpi=160)
    plt.close(fig)
    paths.append(metric_path)

    horizon_path = plot_dir / f"{prefix}_horizon_rmse_by_tool.png"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in models:
        by_horizon = (
            frame.assign(error=(frame[model] - frame["actual"]) ** 2)
            .groupby("horizon")["error"]
            .mean()
        )
        ax.plot(by_horizon.index, np.sqrt(by_horizon.to_numpy()), marker="o", label=model)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(horizon_path, dpi=160)
    plt.close(fig)
    paths.append(horizon_path)

    lines_path = plot_dir / f"{prefix}_forecast_lines.png"
    top_series = (
        frame.groupby("series_id")["actual"].sum().sort_values(ascending=False).head(3).index
    )
    fig, axes = plt.subplots(len(top_series), 1, figsize=(10, 2.8 * len(top_series)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, series_id in zip(axes, top_series, strict=True):
        subset = frame[frame["series_id"] == series_id].sort_values("timestamp")
        ax.plot(subset["timestamp"], subset["actual"], marker="o", label="actual", linewidth=2)
        for model in models[:4]:
            ax.plot(subset["timestamp"], subset[model], marker=".", label=model, alpha=0.8)
        ax.set_title(str(series_id))
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(lines_path, dpi=160)
    plt.close(fig)
    paths.append(lines_path)

    scatter_path = plot_dir / f"{prefix}_actual_vs_predicted.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in models[:5]:
        ax.scatter(frame["actual"], frame[model], s=12, alpha=0.45, label=model)
    low = float(min(frame["actual"].min(), *(frame[model].min() for model in models[:5])))
    high = float(max(frame["actual"].max(), *(frame[model].max() for model in models[:5])))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)
    paths.append(scatter_path)

    return [str(path) for path in paths]


def require_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires polars; run `uv sync --group bench`."
        ) from exc
    return pl


def require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires duckdb for --source duckdb; run "
            "`uv sync --group bench`."
        ) from exc
    return duckdb


def require_pandas_for_benchmark() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires pandas; run `uv sync --group bench`."
        ) from exc
    return pd


def ensure_prophet_class() -> Any:
    global PROPHET_CLASS
    if PROPHET_CLASS is not None:
        return PROPHET_CLASS
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires prophet; run `uv sync --group bench`."
        ) from exc
    PROPHET_CLASS = Prophet
    return PROPHET_CLASS


if __name__ == "__main__":
    raise SystemExit(main())
