from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from cartoboost import __version__
from cartoboost.forecasting import (
    CartoBoostLagForecaster,
    ForecastFrame,
    ForecastMetricSet,
    NaiveForecaster,
    OptimizedThetaForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
    WeightedEnsembleForecaster,
)


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    frame: ForecastFrame
    description: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic forecasting benchmarks.")
    parser.add_argument("--output", default="artifacts/forecasting_benchmark.json")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--panel-series", type=int, default=24)
    args = parser.parse_args()

    if args.days < 180:
        raise ValueError("--days must be at least 180 for the forecasting benchmark")
    if args.horizon < 7:
        raise ValueError("--horizon must be at least 7")
    if args.folds < 1:
        raise ValueError("--folds must be positive")

    datasets = _fixtures(days=args.days, panel_series=args.panel_series)
    model_names = list(_models().keys())
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "benchmark": "deterministic_long_horizon_forecasting",
        "protocol": {
            "days": args.days,
            "horizon": args.horizon,
            "folds": args.folds,
            "panel_series": args.panel_series,
            "split_type": "rolling_origin_last_windows",
            "primary_metric": "rmse",
            "leakage_rule": "train timestamps are strictly before validation timestamps",
            "model_selection": "fixed settings; no hyperparameter search",
        },
        "models": model_names,
        "model_settings": _model_settings(),
        "datasets": [
            {
                "name": dataset.name,
                "description": dataset.description,
                "rows": dataset.frame.n_rows,
                "series_count": max(1, len(dataset.frame.series_ids)),
                "frequency": dataset.frame.freq,
            }
            for dataset in datasets
        ],
        "metrics": {},
    }

    for dataset in datasets:
        payload["metrics"][dataset.name] = _score_dataset(
            dataset.frame,
            horizon=args.horizon,
            folds=args.folds,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _fixtures(days: int, panel_series: int) -> list[BenchmarkDataset]:
    return [
        BenchmarkDataset(
            "trend_only_long",
            _single_fixture(days, lambda i: 40.0 + 0.18 * i),
            "Two-year daily single series with stable trend and no seasonality.",
        ),
        BenchmarkDataset(
            "weekly_annual_seasonal",
            _single_fixture(
                days,
                lambda i: (
                    65.0
                    + 0.05 * i
                    + 8.0 * np.sin(2.0 * np.pi * i / 7.0)
                    + 5.0 * np.sin(2.0 * np.pi * i / 365.25)
                ),
            ),
            "Daily single series with weekly and annual seasonality.",
        ),
        BenchmarkDataset(
            "intermittent_sparse",
            _single_fixture(
                days,
                lambda i: 0.0 if i % 6 else 12.0 + 2.0 * ((i // 6) % 4) + 0.02 * i,
            ),
            "Sparse intermittent daily demand with slow drift.",
        ),
        BenchmarkDataset(
            "regime_shift",
            _single_fixture(
                days,
                lambda i: (
                    35.0 + 0.08 * i + 10.0 * (i >= days // 2) + 4.0 * np.sin(2.0 * np.pi * i / 7.0)
                ),
            ),
            "Single series with weekly seasonality and a mid-history level shift.",
        ),
        BenchmarkDataset(
            "panel_lanes_long",
            _panel_fixture(days, panel_series),
            "Multi-series pickup/dropoff lane demand panel with lane offsets and calendar effects.",
        ),
        BenchmarkDataset(
            "noisy_geotemporal_panel",
            _noisy_panel_fixture(days, panel_series),
            "Panel demand with deterministic bursts, lane heterogeneity, and structured noise.",
        ),
    ]


def _single_fixture(days: int, fn: Callable[[int], float]) -> ForecastFrame:
    rows = [
        {
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "loads": float(fn(i)),
        }
        for i in range(days)
    ]
    return ForecastFrame.from_pandas(
        pd.DataFrame(rows),
        timestamp_col="date",
        target_col="loads",
        freq="D",
    )


def _panel_fixture(days: int, series_count: int) -> ForecastFrame:
    rows = []
    origins = ["PULocationID_132", "PULocationID_138", "PULocationID_161", "PULocationID_236"]
    destinations = ["DOLocationID_230", "DOLocationID_170", "DOLocationID_48", "DOLocationID_263"]
    for idx in range(series_count):
        origin = origins[idx % len(origins)]
        destination = destinations[(idx // len(origins)) % len(destinations)]
        lane = f"{origin}->{destination}#{idx:02d}"
        base = 25.0 + 2.5 * (idx % 6) + 1.2 * (idx // 6)
        for day in range(days):
            weekly = 4.0 * np.sin(2.0 * np.pi * (day + idx) / 7.0)
            annual = 2.5 * np.sin(2.0 * np.pi * (day + 13 * idx) / 365.25)
            airport = 6.0 if "132" in origin or "138" in origin else 0.0
            rows.append(
                {
                    "lane_id": lane,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "loads": float(base + airport + 0.03 * day + weekly + annual),
                }
            )
    return ForecastFrame.from_pandas(
        pd.DataFrame(rows),
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
    )


def _noisy_panel_fixture(days: int, series_count: int) -> ForecastFrame:
    rows = []
    for idx in range(series_count):
        lane = f"PULocationID_{100 + idx}->DOLocationID_{200 + (idx * 7) % 80}"
        base = 30.0 + (idx % 8) * 3.0
        for day in range(days):
            weekly = 3.5 * np.sin(2.0 * np.pi * (day + idx) / 7.0)
            burst = 9.0 if (day + 3 * idx) % 41 in {0, 1, 2} else 0.0
            deterministic_noise = ((day * 17 + idx * 31) % 13 - 6) * 0.35
            rows.append(
                {
                    "lane_id": lane,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "loads": float(base + 0.02 * day + weekly + burst + deterministic_noise),
                }
            )
    return ForecastFrame.from_pandas(
        pd.DataFrame(rows),
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
    )


def _score_dataset(
    frame: ForecastFrame,
    *,
    horizon: int,
    folds: int,
) -> dict[str, Any]:
    data = frame.to_pandas()
    timestamps = data[frame.timestamp_col].drop_duplicates().sort_values().to_list()
    cutoffs = _cutoff_timestamps(timestamps, horizon=horizon, folds=folds)
    split_results = {}
    for split_idx, cutoff in enumerate(cutoffs, start=1):
        split_name = f"rolling_origin_{split_idx}"
        split_results[split_name] = _score_split(frame, cutoff=cutoff, horizon=horizon)
    return {
        "splits": split_results,
        "aggregate": _aggregate_dataset_scores(split_results),
    }


def _cutoff_timestamps(
    timestamps: list[pd.Timestamp],
    *,
    horizon: int,
    folds: int,
) -> list[pd.Timestamp]:
    required = horizon * folds + 1
    if len(timestamps) <= required:
        raise ValueError("not enough timestamps for requested horizon and folds")
    start = len(timestamps) - horizon * folds
    return [timestamps[start + idx * horizon] for idx in range(folds)]


def _score_split(frame: ForecastFrame, *, cutoff: pd.Timestamp, horizon: int) -> dict[str, Any]:
    data = frame.to_pandas()
    train = data[data[frame.timestamp_col] < cutoff]
    test = data[
        (data[frame.timestamp_col] >= cutoff)
        & (data[frame.timestamp_col] < cutoff + pd.Timedelta(days=horizon))
    ]
    if train.empty or test.empty:
        raise ValueError("benchmark split produced empty train or test data")
    train_frame = ForecastFrame.from_pandas(
        train,
        timestamp_col=frame.timestamp_col,
        target_col=frame.target_col,
        series_id_col=frame.series_id_col,
        freq=frame.freq,
    )
    actual = test.copy()
    actual["series_id"] = (
        "__single__" if frame.series_id_col is None else actual[frame.series_id_col]
    )
    actual = actual.rename(columns={frame.timestamp_col: "timestamp", frame.target_col: "actual"})
    actual["horizon"] = actual.groupby("series_id", sort=False).cumcount() + 1
    split_payload: dict[str, Any] = {
        "cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_max_timestamp": train[frame.timestamp_col].max().strftime("%Y-%m-%d"),
        "test_min_timestamp": test[frame.timestamp_col].min().strftime("%Y-%m-%d"),
        "models": {},
    }
    for name, model in _models().items():
        split_payload["models"][name] = _score_model(
            name,
            model,
            train_frame,
            actual,
            train[frame.target_col],
            horizon,
        )
    return split_payload


def _score_model(
    name: str,
    model: Any,
    train_frame: ForecastFrame,
    actual: pd.DataFrame,
    y_train: Any,
    horizon: int,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        fit_started = perf_counter()
        model.fit(train_frame)
        fit_seconds = perf_counter() - fit_started
        predict_started = perf_counter()
        forecast = _forecast_to_frame(model.predict(horizon))
        predict_seconds = perf_counter() - predict_started
        merged = actual[["series_id", "timestamp", "horizon", "actual"]].merge(
            forecast[["series_id", "timestamp", "horizon", "mean"]],
            on=["series_id", "timestamp", "horizon"],
            how="inner",
        )
        if len(merged) != len(actual):
            raise ValueError(
                f"model {name} produced {len(merged)} aligned rows for {len(actual)} actual rows"
            )
        metrics = ForecastMetricSet(seasonal_period=7).evaluate(
            merged["actual"],
            merged["mean"],
            horizon=merged["horizon"],
            series_id=merged["series_id"],
            y_train=y_train,
        )
        return {
            "status": "ok",
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": perf_counter() - started,
            "n_predictions": int(len(merged)),
            **{
                metric: float(metrics[metric])
                for metric in ("mae", "rmse", "mase", "wape", "smape", "bias")
                if metric in metrics and np.isfinite(metrics[metric])
            },
            "coverage_80": 0.0,
            "coverage_95": 0.0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "total_seconds": perf_counter() - started,
        }


def _models() -> dict[str, Any]:
    cartoboost = CartoBoostLagForecaster(
        lags=[1, 2, 7, 14, 28],
        rolling_windows=[7, 14, 28],
        calendar_features=True,
        trend_features=True,
        target_mode="delta_from_last",
        regressor_params={
            "n_estimators": 80,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "min_gain": 0.0,
            "splitters": ["axis"],
        },
    )
    return {
        "naive": NaiveForecaster(),
        "seasonal_naive": SeasonalNaiveForecaster(season_length=7),
        "theta": ThetaForecaster(season_length=7),
        "optimized_theta": OptimizedThetaForecaster(season_length=7),
        "cartoboost_lag": cartoboost,
        "weighted_ensemble": WeightedEnsembleForecaster(
            models={"theta": ThetaForecaster(season_length=7), "naive": NaiveForecaster()},
            weights={"theta": 0.7, "naive": 0.3},
        ),
    }


def _forecast_to_frame(result: Any) -> pd.DataFrame:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if "prediction" in frame.columns and "mean" not in frame.columns:
            frame = frame.rename(columns={"prediction": "mean"})
        return frame
    if hasattr(result, "predictions"):
        return pd.DataFrame(
            result.predictions(),
            columns=["series_id", "timestamp", "horizon", "model", "mean"],
        ).assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))
    raise TypeError("forecast result must expose to_pandas() or predictions()")


def _model_settings() -> dict[str, Any]:
    return {
        "naive": {},
        "seasonal_naive": {"season_length": 7},
        "theta": {"season_length": 7},
        "optimized_theta": {"season_length": 7, "default_grid": True},
        "cartoboost_lag": {
            "lags": [1, 2, 7, 14, 28],
            "rolling_windows": [7, 14, 28],
            "calendar_features": True,
            "trend_features": True,
            "regressor_params": {
                "n_estimators": 80,
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_samples_leaf": 5,
                "min_gain": 0.0,
                "splitters": ["axis"],
            },
        },
        "weighted_ensemble": {"weights": {"theta": 0.7, "naive": 0.3}},
    }


def _aggregate_dataset_scores(split_results: dict[str, Any]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for split in split_results.values():
        for model, metrics in split["models"].items():
            if metrics.get("status") == "ok":
                by_model.setdefault(model, []).append(metrics)
    aggregate = {}
    for model, rows in by_model.items():
        aggregate[model] = {
            metric: float(np.mean([row[metric] for row in rows if metric in row]))
            for metric in (
                "mae",
                "rmse",
                "mase",
                "wape",
                "smape",
                "bias",
                "fit_seconds",
                "predict_seconds",
                "total_seconds",
            )
            if any(metric in row for row in rows)
        }
    return aggregate


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        dataset: {
            model: metrics.get("rmse")
            for model, metrics in dataset_metrics["aggregate"].items()
            if "rmse" in metrics
        }
        for dataset, dataset_metrics in payload["metrics"].items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
