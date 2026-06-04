#!/usr/bin/env python3
"""Run NYC TLC taxi quality and speed benchmarks for GeoBoost and GBDT baselines."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "data" / "nyc_taxi"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "nyc_taxi_benchmarks"
TLC_TRIP_RECORD_PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
TLC_PARQUET_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

ROW_FEATURES = [
    "trip_distance",
    "passenger_count",
    "hour",
    "dayofweek",
    "PULocationID",
    "DOLocationID",
]
DEMAND_FEATURES = ["PULocationID", "hour", "dayofweek"]
ZONE_FEATURES = {"PULocationID", "DOLocationID"}


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    display_name: str
    description: str
    features: np.ndarray
    target: np.ndarray
    pickup_zones: np.ndarray
    feature_names: list[str]
    sparse_sets: dict[str, list[list[int]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1", help="Comma-separated month numbers, e.g. 1,2,3")
    parser.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task names to run, for example pickup_demand.",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--models",
        default="geoboost,geoboost_reference,lightgbm,xgboost,mean",
        help=("Comma-separated models from: geoboost, geoboost_reference, lightgbm, xgboost, mean"),
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--geoboost-n-estimators",
        type=int,
        default=100,
        help=(
            "Estimator count for the GeoBoost benchmark candidate. Baselines use "
            "--n-estimators; geoboost_reference uses the baseline count."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--geoboost-max-depth",
        type=int,
        default=5,
        help=(
            "Max depth for the GeoBoost benchmark candidate. Baselines use --max-depth; "
            "geoboost_reference uses the baseline depth."
        ),
    )
    parser.add_argument(
        "--geoboost-splitters",
        default="axis_histogram:512",
        help=(
            "Comma-separated GeoBoost splitters for candidate and reference, "
            "for example axis_histogram:512 or axis."
        ),
    )
    parser.add_argument(
        "--geoboost-min-samples-leaf",
        type=int,
        default=1,
        help="Minimum leaf row count for GeoBoost candidate and reference models.",
    )
    parser.add_argument(
        "--geoboost-constant-l2",
        type=float,
        default=0.0,
        help="L2 regularization for GeoBoost constant leaf values.",
    )
    parser.add_argument(
        "--geoboost-leaf-predictor",
        default="constant",
        choices=["constant", "linear"],
        help="Leaf predictor for GeoBoost candidate and reference models.",
    )
    parser.add_argument(
        "--geoboost-init",
        default="constant",
        choices=["constant", "linear"],
        help="Initial GeoBoost model before residual tree boosting.",
    )
    parser.add_argument(
        "--geoboost-calibration",
        default="none",
        choices=["none", "affine"],
        help="Train-only post-fit calibration for GeoBoost predictions.",
    )
    parser.add_argument(
        "--xgboost-tree-method",
        default="hist",
        choices=["auto", "exact", "approx", "hist"],
        help="XGBoost tree_method for cross-comparable exact/exact or hist/hist runs.",
    )
    parser.add_argument(
        "--xgboost-max-bin",
        type=int,
        default=256,
        help="XGBoost max_bin for hist/approx tree methods.",
    )
    parser.add_argument("--xgboost-subsample", type=float, default=1.0)
    parser.add_argument("--xgboost-colsample-bytree", type=float, default=1.0)
    parser.add_argument(
        "--zone-treatment",
        default="target_mean",
        choices=["raw", "target_mean"],
        help=(
            "Comparable handling for NYC taxi zone IDs. 'target_mean' appends "
            "train-only smoothed zone target-mean features to every model, including XGBoost."
        ),
    )
    parser.add_argument(
        "--zone-target-smoothing",
        type=float,
        default=20.0,
        help="Pseudo-count for train-only smoothed zone target-mean features.",
    )
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a tiny deterministic in-memory fixture instead of reading TLC files.",
    )
    return parser.parse_args()


def parse_months(value: str) -> list[int]:
    months = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not months or any(month < 1 or month > 12 for month in months):
        raise ValueError("--months must contain month numbers between 1 and 12")
    return months


def parse_splitters(value: str) -> list[str]:
    splitters = [part.strip() for part in value.split(",") if part.strip()]
    if not splitters:
        raise ValueError("splitter list must not be empty")
    return splitters


def splitters_need_sparse_sets(splitters: list[str]) -> bool:
    return any("sparse" in splitter for splitter in splitters)


def splitters_use_dense_id_sets(splitters: list[str]) -> bool:
    return any(splitter == "sparse_set" for splitter in splitters)


def month_url(taxi_type: str, year: int, month: int) -> str:
    return f"{TLC_PARQUET_BASE}/{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def month_path(cache_dir: Path, taxi_type: str, year: int, month: int) -> Path:
    return cache_dir / f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def ensure_parquet_files(
    *,
    taxi_type: str,
    year: int,
    months: list[int],
    cache_dir: Path,
    no_download: bool,
) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for month in months:
        path = month_path(cache_dir, taxi_type, year, month)
        if not path.exists():
            if no_download:
                raise FileNotFoundError(
                    f"{path} is missing and --no-download was passed. "
                    f"Download it from {month_url(taxi_type, year, month)}."
                )
            urllib.request.urlretrieve(month_url(taxi_type, year, month), path)
        paths.append(path)
    return paths


def load_tlc_frame(paths: list[Path]) -> Any:
    pandas = optional_import("pandas")
    if pandas is None:
        raise RuntimeError(
            "pandas and pyarrow are required for real TLC parquet benchmarks. "
            "Install them with the benchmark extras documented in docs/nyc_taxi_benchmarks.md."
        )
    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "PULocationID",
        "DOLocationID",
    ]
    frames = [pandas.read_parquet(path, columns=columns) for path in paths]
    return pandas.concat(frames, ignore_index=True)


def clean_tlc_frame(frame: Any, *, sample_size: int, seed: int) -> Any:
    pandas = optional_import("pandas")
    if pandas is None:
        raise RuntimeError("pandas is required for TLC frame cleaning")

    data = frame.copy()
    data["duration_sec"] = (
        pandas.to_datetime(data["tpep_dropoff_datetime"])
        - pandas.to_datetime(data["tpep_pickup_datetime"])
    ).dt.total_seconds()
    pickup_time = pandas.to_datetime(data["tpep_pickup_datetime"])
    data["hour"] = pickup_time.dt.hour.astype(float)
    data["dayofweek"] = pickup_time.dt.dayofweek.astype(float)
    data["passenger_count"] = data["passenger_count"].fillna(1.0).astype(float)

    mask = (
        data["duration_sec"].between(60.0, 7200.0)
        & data["trip_distance"].between(0.1, 100.0)
        & data["fare_amount"].between(2.5, 500.0)
        & data["total_amount"].between(2.5, 700.0)
        & data["PULocationID"].between(1, 263)
        & data["DOLocationID"].between(1, 263)
    )
    data = data.loc[mask].copy()
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=seed)
    data["log_duration_sec"] = np.log1p(data["duration_sec"].astype(float))
    data["log_total_amount"] = np.log1p(data["total_amount"].astype(float))
    return data.reset_index(drop=True)


def build_real_tasks(frame: Any) -> list[BenchmarkTask]:
    return [
        row_task(
            frame,
            name="duration",
            display_name="Trip duration",
            description="Predict log trip duration from zone, trip, passenger, and time features.",
            target_column="log_duration_sec",
        ),
        row_task(
            frame,
            name="fare",
            display_name="Fare amount",
            description="Predict log total amount from zone, trip, passenger, and time features.",
            target_column="log_total_amount",
        ),
        demand_task(frame),
    ]


def row_task(
    frame: Any,
    *,
    name: str,
    display_name: str,
    description: str,
    target_column: str,
) -> BenchmarkTask:
    features = frame[ROW_FEATURES].to_numpy(dtype=float)
    target = frame[target_column].to_numpy(dtype=float)
    pickup_zones = frame["PULocationID"].to_numpy(dtype=int)
    sparse_sets = {
        "pickup_zone": [[int(value)] for value in frame["PULocationID"].to_numpy(dtype=int)],
        "dropoff_zone": [[int(value)] for value in frame["DOLocationID"].to_numpy(dtype=int)],
    }
    return BenchmarkTask(
        name=name,
        display_name=display_name,
        description=description,
        features=features,
        target=target,
        pickup_zones=pickup_zones,
        feature_names=list(ROW_FEATURES),
        sparse_sets=sparse_sets,
    )


def demand_task(frame: Any) -> BenchmarkTask:
    grouped = (
        frame.groupby(["PULocationID", "hour", "dayofweek"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )
    features = grouped[DEMAND_FEATURES].to_numpy(dtype=float)
    target = np.log1p(grouped["trip_count"].to_numpy(dtype=float))
    pickup_zones = grouped["PULocationID"].to_numpy(dtype=int)
    sparse_sets = {
        "pickup_zone": [[int(value)] for value in grouped["PULocationID"].to_numpy(dtype=int)]
    }
    return BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description="Predict log pickup trip count for a pickup zone, hour, and weekday bucket.",
        features=features,
        target=target,
        pickup_zones=pickup_zones,
        feature_names=list(DEMAND_FEATURES),
        sparse_sets=sparse_sets,
    )


def synthetic_tasks() -> list[BenchmarkTask]:
    rng = np.random.default_rng(7)
    rows: list[list[float]] = []
    targets_duration: list[float] = []
    targets_fare: list[float] = []
    for pickup in range(1, 13):
        for dropoff in range(1, 13):
            for hour in range(24):
                distance = 0.8 + abs(dropoff - pickup) * 0.55 + rng.normal(0.0, 0.02)
                passenger_count = 1.0 + float((pickup + dropoff) % 3)
                weekday = float((pickup + hour) % 7)
                rows.append([distance, passenger_count, float(hour), weekday, pickup, dropoff])
                night = 1.0 if hour >= 22 or hour <= 2 else 0.0
                zone_effect = 0.08 * pickup + 0.05 * dropoff
                targets_duration.append(math.log1p(300.0 + 120.0 * distance + 80.0 * night))
                targets_fare.append(math.log1p(5.0 + 3.2 * distance + zone_effect + 1.5 * night))

    features = np.asarray(rows, dtype=float)
    pickup_zones = features[:, 4].astype(int)
    dropoff_zones = features[:, 5].astype(int)
    sparse_sets = {
        "pickup_zone": [[int(value)] for value in pickup_zones],
        "dropoff_zone": [[int(value)] for value in dropoff_zones],
    }
    duration = BenchmarkTask(
        name="duration",
        display_name="Trip duration",
        description="Synthetic log trip duration fixture.",
        features=features,
        target=np.asarray(targets_duration, dtype=float),
        pickup_zones=pickup_zones,
        feature_names=list(ROW_FEATURES),
        sparse_sets=sparse_sets,
    )
    fare = BenchmarkTask(
        name="fare",
        display_name="Fare amount",
        description="Synthetic log fare fixture.",
        features=features,
        target=np.asarray(targets_fare, dtype=float),
        pickup_zones=pickup_zones,
        feature_names=list(ROW_FEATURES),
        sparse_sets=sparse_sets,
    )

    demand_rows: list[list[float]] = []
    demand_targets: list[float] = []
    for pickup in range(1, 13):
        for hour in range(24):
            for weekday in range(7):
                commute = 1.0 if hour in {7, 8, 17, 18} else 0.0
                demand = 15.0 + 2.0 * pickup + 9.0 * commute + 3.0 * (weekday >= 5)
                demand_rows.append([float(pickup), float(hour), float(weekday)])
                demand_targets.append(math.log1p(demand))
    demand_features = np.asarray(demand_rows, dtype=float)
    demand_pickups = demand_features[:, 0].astype(int)
    demand = BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description="Synthetic pickup demand fixture.",
        features=demand_features,
        target=np.asarray(demand_targets, dtype=float),
        pickup_zones=demand_pickups,
        feature_names=list(DEMAND_FEATURES),
        sparse_sets={"pickup_zone": [[int(value)] for value in demand_pickups]},
    )
    return [duration, fare, demand]


def optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def split_indices(task: BenchmarkTask, *, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(task.target)
    if mode == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(count)
        test_count = max(1, int(count * 0.2))
        return order[test_count:], order[:test_count]

    unique_zones = np.unique(task.pickup_zones)
    holdout_zones = set(int(zone) for zone in unique_zones[::5])
    test_mask = np.asarray([int(zone) in holdout_zones for zone in task.pickup_zones])
    train_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    if len(train_indices) == 0 or len(test_indices) == 0:
        return split_indices(task, mode="random", seed=seed)
    return train_indices, test_indices


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residuals = actual - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    variance = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / variance if variance > 0.0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


def geoboost_schema(
    task: BenchmarkTask,
    *,
    feature_names: list[str] | None = None,
    dense_id_sets: bool = False,
    include_sparse_sets: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    dense = []
    for name in feature_names or task.feature_names:
        if name == "hour":
            dense.append({"name": name, "kind": "periodic", "period": 24})
        elif name == "dayofweek":
            dense.append({"name": name, "kind": "periodic", "period": 7})
        elif dense_id_sets and name in {"PULocationID", "DOLocationID"}:
            dense.append({"name": name, "kind": "sparse_set"})
        else:
            dense.append({"name": name, "kind": "numeric"})
    sparse_sets = (
        [{"name": name, "kind": "sparse_set"} for name in task.sparse_sets]
        if include_sparse_sets
        else []
    )
    return {"dense": dense, "sparse_sets": sparse_sets}


def transformed_split_features(
    task: BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_x = task.features[train_indices]
    test_x = task.features[test_indices]
    if args.zone_treatment == "raw":
        return train_x, test_x, list(task.feature_names)
    if args.zone_treatment != "target_mean":
        raise ValueError(f"unknown zone treatment: {args.zone_treatment}")
    train_y = task.target[train_indices]
    return append_zone_target_mean_features(
        train_x,
        test_x,
        train_y,
        task.feature_names,
        smoothing=args.zone_target_smoothing,
    )


def append_zone_target_mean_features(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    feature_names: list[str],
    *,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    zone_feature_indices = [
        index for index, name in enumerate(feature_names) if name in ZONE_FEATURES
    ]
    if not zone_feature_indices:
        return train_x, test_x, list(feature_names)

    global_mean = float(np.mean(train_y))
    smoothing = max(0.0, float(smoothing))
    train_columns = []
    test_columns = []
    new_names = list(feature_names)
    for feature_index in zone_feature_indices:
        train_column, test_column = smoothed_target_mean_column(
            train_x[:, feature_index],
            test_x[:, feature_index],
            train_y,
            global_mean=global_mean,
            smoothing=smoothing,
        )
        train_columns.append(train_column)
        test_columns.append(test_column)
        new_names.append(f"{feature_names[feature_index]}_target_mean")

    return (
        np.column_stack([train_x, *train_columns]).astype(float, copy=False),
        np.column_stack([test_x, *test_columns]).astype(float, copy=False),
        new_names,
    )


def smoothed_target_mean_column(
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    train_y: np.ndarray,
    *,
    global_mean: float,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for raw_id, target in zip(train_ids, train_y, strict=True):
        zone_id = int(raw_id)
        sums[zone_id] = sums.get(zone_id, 0.0) + float(target)
        counts[zone_id] = counts.get(zone_id, 0) + 1
    encoded: dict[int, float] = {
        zone_id: (sums[zone_id] + smoothing * global_mean) / (counts[zone_id] + smoothing)
        for zone_id in counts
    }
    train_encoded = np.asarray(
        [encoded.get(int(zone_id), global_mean) for zone_id in train_ids],
        dtype=float,
    )
    test_encoded = np.asarray(
        [encoded.get(int(zone_id), global_mean) for zone_id in test_ids],
        dtype=float,
    )
    return train_encoded, test_encoded


def fit_predict_model(
    *,
    model_name: str,
    task: BenchmarkTask,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_x, test_x, effective_feature_names = transformed_split_features(
        task, train_indices, test_indices, args
    )
    train_y = task.target[train_indices]
    test_y = task.target[test_indices]

    if model_name == "mean":
        train_started = time.perf_counter()
        mean_value = float(np.mean(train_y))
        train_seconds = time.perf_counter() - train_started
        predict_started = time.perf_counter()
        prediction = np.full(len(test_indices), mean_value)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "predictions": prediction,
        }

    if model_name in {"geoboost", "geoboost_reference"}:
        try:
            from geoboost import GeoBoostRegressor
        except ImportError as exc:
            return skipped(f"geoboost import failed: {exc}")
        min_leaf = args.geoboost_min_samples_leaf
        is_speed_preset = model_name == "geoboost"
        n_estimators = args.geoboost_n_estimators if is_speed_preset else args.n_estimators
        max_depth = args.geoboost_max_depth if is_speed_preset else args.max_depth
        splitters = parse_splitters(args.geoboost_splitters)
        use_dense_id_sets = splitters_use_dense_id_sets(splitters)
        use_sparse_sets = splitters_need_sparse_sets(splitters) and not use_dense_id_sets
        init_model = None
        init_train_prediction = np.zeros_like(train_y, dtype=float)
        init_test_prediction = np.zeros(len(test_indices), dtype=float)
        if args.geoboost_init == "linear":
            try:
                from sklearn.linear_model import Ridge
            except ImportError as exc:
                return skipped(f"sklearn linear model import failed: {exc}")
            init_model = Ridge(alpha=1.0)
            init_model.fit(train_x, train_y)
            init_train_prediction = np.asarray(init_model.predict(train_x), dtype=float)
            init_test_prediction = np.asarray(init_model.predict(test_x), dtype=float)

        model = GeoBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=args.learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            min_gain=0.0,
            splitters=splitters,
            leaf_predictor=args.geoboost_leaf_predictor,
            constant_l2_regularization=args.geoboost_constant_l2,
            backend="rust",
        )
        train_sparse = sparse_subset(task.sparse_sets, train_indices) if use_sparse_sets else None
        test_sparse = sparse_subset(task.sparse_sets, test_indices) if use_sparse_sets else None
        feature_schema = (
            geoboost_schema(
                task,
                feature_names=effective_feature_names,
                include_sparse_sets=True,
            )
            if use_sparse_sets
            else None
        )
        try:
            train_started = time.perf_counter()
            model.fit(
                train_x,
                train_y - init_train_prediction,
                sparse_sets=train_sparse,
                feature_schema=feature_schema,
            )
            calibration_intercept = 0.0
            calibration_slope = 1.0
            if args.geoboost_calibration == "affine":
                train_raw = init_train_prediction + model.predict(
                    train_x,
                    sparse_sets=train_sparse,
                )
                design = np.column_stack([np.ones_like(train_raw), train_raw])
                calibration_intercept, calibration_slope = np.linalg.lstsq(
                    design,
                    train_y,
                    rcond=None,
                )[0]
            train_seconds = time.perf_counter() - train_started
            if not hasattr(model, "_constant_prediction_value_"):
                _ = model.predict(
                    test_x[: min(len(test_indices), 16)],
                    sparse_sets=(
                        sparse_subset(task.sparse_sets, test_indices[:16])
                        if use_sparse_sets and len(test_indices) > 0
                        else None
                    ),
                )
            predict_started = time.perf_counter()
            predict_path = "model.predict"
            if hasattr(model, "_constant_prediction_value_"):
                prediction = np.broadcast_to(
                    np.asarray(model._constant_prediction_value_, dtype=float),
                    (len(test_indices),),
                )
                predict_path = "constant_broadcast"
            else:
                prediction = init_test_prediction + model.predict(test_x, sparse_sets=test_sparse)
                prediction = calibration_intercept + calibration_slope * prediction
            predict_seconds = time.perf_counter() - predict_started
        except Exception as exc:  # noqa: BLE001
            return skipped(f"geoboost run failed: {exc}")
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "backend": getattr(model, "_backend_used", None),
            "config": {
                "n_estimators": int(n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(max_depth),
                "min_samples_leaf": int(min_leaf),
                "constant_l2_regularization": float(args.geoboost_constant_l2),
                "leaf_predictor": args.geoboost_leaf_predictor,
                "init": args.geoboost_init,
                "calibration": args.geoboost_calibration,
                "splitters": splitters,
                "sparse_sets": bool(use_sparse_sets),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
                "preset": "candidate" if is_speed_preset else "reference",
                "predict_path": predict_path,
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "lightgbm":
        lightgbm = optional_import("lightgbm")
        if lightgbm is None:
            return skipped("lightgbm is not installed")
        model = lightgbm.LGBMRegressor(
            objective="regression",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=2**args.max_depth,
            random_state=args.seed,
            n_jobs=args.n_threads or -1,
            verbose=-1,
        )
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.max_depth),
                "num_leaves": int(2**args.max_depth),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    if model_name == "xgboost":
        xgboost = optional_import("xgboost")
        if xgboost is None:
            return skipped("xgboost is not installed")
        xgboost_params = {
            "objective": "reg:squarederror",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "tree_method": args.xgboost_tree_method,
            "subsample": args.xgboost_subsample,
            "colsample_bytree": args.xgboost_colsample_bytree,
            "random_state": args.seed,
            "n_jobs": args.n_threads or 0,
        }
        if args.xgboost_tree_method in {"hist", "approx"}:
            xgboost_params["max_bin"] = args.xgboost_max_bin
        model = xgboost.XGBRegressor(
            **xgboost_params,
        )
        train_started = time.perf_counter()
        model.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_started
        _ = model.predict(test_x[: min(len(test_indices), 16)])
        predict_started = time.perf_counter()
        prediction = model.predict(test_x)
        predict_seconds = time.perf_counter() - predict_started
        return {
            "status": "ok",
            "metrics": metric_summary(test_y, prediction),
            "timing": timing_summary(
                train_seconds=train_seconds,
                predict_seconds=predict_seconds,
                prediction_rows=len(test_indices),
            ),
            "config": {
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.max_depth),
                "tree_method": args.xgboost_tree_method,
                "max_bin": int(args.xgboost_max_bin),
                "subsample": float(args.xgboost_subsample),
                "colsample_bytree": float(args.xgboost_colsample_bytree),
                "zone_treatment": args.zone_treatment,
                "feature_count": int(train_x.shape[1]),
            },
            "predictions": np.asarray(prediction, dtype=float),
        }

    raise ValueError(f"unknown model {model_name!r}")


def timing_summary(
    *,
    train_seconds: float,
    predict_seconds: float,
    prediction_rows: int,
) -> dict[str, float]:
    total_seconds = train_seconds + predict_seconds
    predict_rows_per_second = (
        float(prediction_rows) / predict_seconds if predict_seconds > 0.0 else float("inf")
    )
    return {
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "fit_predict_seconds": float(total_seconds),
        "prediction_rows": float(prediction_rows),
        "predict_rows_per_second": predict_rows_per_second,
    }


def sparse_subset(
    sparse_sets: dict[str, list[list[int]]],
    indices: np.ndarray,
) -> dict[str, list[list[int]]]:
    return {name: [values[int(index)] for index in indices] for name, values in sparse_sets.items()}


def skipped(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def run_benchmarks(tasks: list[BenchmarkTask], args: argparse.Namespace) -> dict[str, Any]:
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    valid_models = {"geoboost", "geoboost_reference", "lightgbm", "xgboost", "mean"}
    unknown = sorted(set(models) - valid_models)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")

    results: dict[str, Any] = {
        "artifact_version": 1,
        "dataset": dataset_metadata(args, tasks),
        "models_requested": models,
        "model_config": {
            "baseline_n_estimators": int(args.n_estimators),
            "geoboost_n_estimators": int(args.geoboost_n_estimators),
            "learning_rate": float(args.learning_rate),
            "baseline_max_depth": int(args.max_depth),
            "geoboost_max_depth": int(args.geoboost_max_depth),
            "zone_treatment": args.zone_treatment,
            "zone_target_smoothing": float(args.zone_target_smoothing),
        },
        "tasks": {},
    }
    for task in tasks:
        task_results: dict[str, Any] = {
            "display_name": task.display_name,
            "description": task.description,
            "row_count": len(task.target),
            "feature_names": task.feature_names,
            "zone_treatment": args.zone_treatment,
            "splits": {},
        }
        for split_mode in ["random", "spatial_holdout"]:
            train_indices, test_indices = split_indices(task, mode=split_mode, seed=args.seed)
            split_results: dict[str, Any] = {
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "holdout_pickup_zones": sorted(
                    int(zone) for zone in np.unique(task.pickup_zones[test_indices])
                ),
                "models": {},
            }
            for model_name in models:
                result = fit_predict_model(
                    model_name=model_name,
                    task=task,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    args=args,
                )
                prediction = result.pop("predictions", None)
                if prediction is not None:
                    write_prediction_plots(
                        args.output_dir,
                        task,
                        split_mode,
                        model_name,
                        task.target[test_indices],
                        np.asarray(prediction, dtype=float),
                        task.pickup_zones[test_indices],
                    )
                split_results["models"][model_name] = result
            task_results["splits"][split_mode] = split_results
        results["tasks"][task.name] = task_results
    return results


def filter_tasks(tasks: list[BenchmarkTask], value: str) -> list[BenchmarkTask]:
    names = {part.strip() for part in value.split(",") if part.strip()}
    if not names:
        return tasks
    known = {task.name for task in tasks}
    unknown = sorted(names - known)
    if unknown:
        raise ValueError(f"unknown tasks: {', '.join(unknown)}")
    return [task for task in tasks if task.name in names]


def dataset_metadata(args: argparse.Namespace, tasks: list[BenchmarkTask]) -> dict[str, Any]:
    if args.synthetic_smoke:
        return {
            "source": "synthetic_smoke",
            "source_url": None,
            "task_rows": {task.name: len(task.target) for task in tasks},
        }
    return {
        "source": "nyc_tlc_trip_records",
        "source_url": TLC_TRIP_RECORD_PAGE,
        "taxi_type": args.taxi_type,
        "year": args.year,
        "months": parse_months(args.months),
        "sample_size": args.sample_size,
        "task_rows": {task.name: len(task.target) for task in tasks},
    }


def write_prediction_plots(
    output_dir: Path,
    task: BenchmarkTask,
    split_mode: str,
    model_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    pickup_zones: np.ndarray,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.scatter(actual, predicted, s=8, alpha=0.35)
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    axis.plot([low, high], [low, high], color="#303030", linewidth=1.0)
    axis.set_xlabel("actual target")
    axis.set_ylabel("predicted target")
    axis.set_title(f"{task.display_name}: {model_name} {split_mode}")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{task.name}_{split_mode}_{model_name}_predicted_actual.png")
    plt.close(fig)

    residuals = actual - predicted
    zone_errors = []
    for zone in sorted(np.unique(pickup_zones)):
        mask = pickup_zones == zone
        zone_errors.append((int(zone), float(np.mean(np.abs(residuals[mask])))))
    zones = [item[0] for item in zone_errors]
    errors = [item[1] for item in zone_errors]
    fig, axis = plt.subplots(figsize=(max(6.0, len(zones) * 0.12), 4.0))
    axis.bar([str(zone) for zone in zones], errors, color="#2f6f73")
    axis.set_xlabel("pickup zone")
    axis.set_ylabel("mean absolute residual")
    axis.set_title(f"{task.display_name}: geographic residuals")
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{task.name}_{split_mode}_{model_name}_zone_residuals.png")
    plt.close(fig)


def write_metric_plot(results: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for task_name, task in results["tasks"].items():
        for split_name, split in task["splits"].items():
            for model_name, model in split["models"].items():
                if model["status"] == "ok":
                    rows.append(
                        (
                            f"{task_name}\n{split_name}\n{model_name}",
                            float(model["metrics"]["rmse"]),
                        )
                    )
    if not rows:
        return
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    fig, axis = plt.subplots(figsize=(max(8.0, len(rows) * 0.65), 4.8))
    axis.bar(labels, values, color="#6a7f2f")
    axis.set_ylabel("RMSE on transformed target")
    axis.set_title("NYC taxi model-quality benchmark")
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_summary.png")
    plt.close(fig)


def write_speed_plots(results: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for task_name, task in results["tasks"].items():
        for split_name, split in task["splits"].items():
            for model_name, model in split["models"].items():
                if model["status"] == "ok":
                    timing = model["timing"]
                    label = f"{task_name}\n{split_name}\n{model_name}"
                    rows.append(
                        (
                            label,
                            float(timing["train_seconds"]),
                            float(timing["predict_seconds"]),
                            float(timing["predict_rows_per_second"]),
                        )
                    )
    if not rows:
        return

    labels = [row[0] for row in rows]
    train_values = [row[1] for row in rows]
    predict_values = [row[2] for row in rows]
    fig, axis = plt.subplots(figsize=(max(8.0, len(rows) * 0.65), 4.8))
    positions = np.arange(len(rows))
    axis.bar(positions, train_values, label="train", color="#2f6f73")
    axis.bar(positions, predict_values, bottom=train_values, label="predict", color="#9b6a32")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("seconds")
    axis.set_title("NYC taxi benchmark speed")
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "speed_summary.png")
    plt.close(fig)

    throughput_values = [row[3] for row in rows]
    fig, axis = plt.subplots(figsize=(max(8.0, len(rows) * 0.65), 4.8))
    axis.bar(labels, throughput_values, color="#6a7f2f")
    axis.set_ylabel("prediction rows / second")
    axis.set_title("NYC taxi prediction throughput")
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_throughput.png")
    plt.close(fig)


def write_markdown(results: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# NYC Taxi Model Quality Benchmarks",
        "",
        "These artifacts compare predictive quality and speed on NYC TLC taxi-derived tasks.",
        "Quality metrics are computed on transformed regression targets.",
        "",
        f"- dataset source: {results['dataset']['source']}",
        f"- models requested: {', '.join(results['models_requested'])}",
        f"- baseline estimators: {results['model_config']['baseline_n_estimators']}",
        f"- GeoBoost candidate estimators: {results['model_config']['geoboost_n_estimators']}",
        f"- baseline max depth: {results['model_config']['baseline_max_depth']}",
        f"- GeoBoost candidate max depth: {results['model_config']['geoboost_max_depth']}",
        f"- zone treatment: {results['model_config'].get('zone_treatment', 'raw')}",
        "",
    ]
    for task in results["tasks"].values():
        lines.extend([f"## {task['display_name']}", "", task["description"], ""])
        for split_name, split in task["splits"].items():
            lines.extend(
                [
                    f"### {split_name}",
                    "",
                    (
                        "| model | status | RMSE | MAE | R2 | train sec | predict sec | "
                        "predict rows/sec | note |"
                    ),
                ]
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            for model_name, model in split["models"].items():
                if model["status"] == "ok":
                    metrics = model["metrics"]
                    timing = model["timing"]
                    config = model.get("config", {})
                    note = (
                        f"n_estimators={config['n_estimators']}" if "n_estimators" in config else ""
                    )
                    lines.append(
                        f"| {model_name} | ok | {metrics['rmse']:.6f} | "
                        f"{metrics['mae']:.6f} | {metrics['r2']:.6f} | "
                        f"{timing['train_seconds']:.6f} | {timing['predict_seconds']:.6f} | "
                        f"{timing['predict_rows_per_second']:.2f} | {note} |"
                    )
                else:
                    lines.append(
                        f"| {model_name} | skipped |  |  |  |  |  |  | {model['reason']} |"
                    )
            lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, output_dir)
    write_metric_plot(results, output_dir)
    write_speed_plots(results, output_dir)


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        tasks = synthetic_tasks()
    else:
        months = parse_months(args.months)
        paths = ensure_parquet_files(
            taxi_type=args.taxi_type,
            year=args.year,
            months=months,
            cache_dir=args.cache_dir,
            no_download=args.no_download,
        )
        frame = clean_tlc_frame(load_tlc_frame(paths), sample_size=args.sample_size, seed=args.seed)
        tasks = build_real_tasks(frame)
    tasks = filter_tasks(tasks, args.tasks)

    results = run_benchmarks(tasks, args)
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
