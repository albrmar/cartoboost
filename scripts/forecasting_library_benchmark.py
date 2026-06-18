#!/usr/bin/env python3
"""Benchmark CartoBoost against explicit forecasting libraries.

The fixture is synthetic but domain-shaped: daily pickup/dropoff lane demand with
zone IDs, route distance, airport-lane structure, borough codes, weekly effects,
and deterministic event spikes. The library baselines are functime and
StatsForecast models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from cartoboost import __version__
from cartoboost.regressor import CartoBoostRegressor

FEATURE_COLUMNS = [
    "loads_lag_1",
    "loads_lag_7",
    "loads_lag_14",
    "loads_lag_21",
    "loads_lag_28",
    "loads_roll_7",
    "loads_roll_14",
    "loads_roll_28",
    "date_dayofweek",
    "date_day",
    "date_month",
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
STATSFORECAST_MODELS = ["statsforecast_seasonal_naive", "statsforecast_autoets"]
FORECASTING_LIBRARY_MODELS = {
    "functime": FUNCTIME_MODELS,
    "statsforecast": STATSFORECAST_MODELS,
}
MODEL_LIBRARIES = {
    "cartoboost_lag": "cartoboost",
    **{model: "functime" for model in FUNCTIME_MODELS},
    **{model: "statsforecast" for model in STATSFORECAST_MODELS},
}
FORECASTING_LIBRARY_BASELINES = [*FUNCTIME_MODELS, *STATSFORECAST_MODELS]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CartoBoost global lag forecasts against functime and StatsForecast."
    )
    parser.add_argument("--source", choices=["polars", "duckdb"], default="polars")
    parser.add_argument("--output", default="artifacts/forecasting_library_benchmark_polars.json")
    parser.add_argument("--lanes", type=int, default=36)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmark_start = perf_counter()
    load_start = perf_counter()
    source = args.source
    table = load_fixture(source, lanes=args.lanes, days=args.days, seed=args.seed)
    load_seconds = perf_counter() - load_start
    metrics, quality, timing = score_models(table, horizon=args.horizon)
    total_seconds = perf_counter() - benchmark_start
    timing = {
        "total_seconds": total_seconds,
        "load_seconds": load_seconds,
        **timing,
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "benchmark": "geotemporal_lane_demand_forecasting_libraries",
        "fixture_source": source,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": FORECASTING_LIBRARY_MODELS,
        "model_libraries": MODEL_LIBRARIES,
        "dataset": {
            "series": args.lanes,
            "days": args.days,
            "horizon": args.horizon,
            "seed": args.seed,
            "domain": "daily NYC taxi-style pickup/dropoff lane demand",
            "static_covariates": STATIC_COVARIATES,
        },
        "models": ["cartoboost_lag", *FORECASTING_LIBRARY_BASELINES],
        "metrics": metrics,
        "quality": quality,
        "timing": timing,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"quality": quality, "timing": timing}, indent=2, sort_keys=True))
    return 0


def load_fixture(source: str, *, lanes: int, days: int, seed: int) -> Any:
    table = make_fixture(lanes=lanes, days=days, seed=seed)
    if source == "polars":
        return table
    if source == "duckdb":
        duckdb = require_duckdb()
        con = duckdb.connect(":memory:")
        try:
            con.register("lane_demand", table.to_arrow())
            return con.sql(
                """
                SELECT *
                FROM lane_demand
                ORDER BY lane_id, date
                """
            ).pl()
        finally:
            con.close()
    raise ValueError(f"unsupported fixture source {source!r}")


def make_fixture(*, lanes: int, days: int, seed: int) -> Any:
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
            airport_event = 4.0 if airport_lane and day % 28 in {5, 6, 7} else 0.0
            deterministic_noise = ((lane_idx * 17 + day * 13) % 11 - 5) * 0.12
            demand = max(
                0.0,
                base + lane_effect + weekly + slow_drift + airport_event + deterministic_noise,
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


def score_models(
    table: Any, *, horizon: int
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, Any]]:
    pl = require_polars()
    cutoff = table.select(pl.col("date").unique().sort()).to_series()[-horizon]
    train = table.filter(pl.col("date") < cutoff)
    test = table.filter(pl.col("date") >= cutoff)
    if train.is_empty() or test.is_empty():
        raise ValueError("benchmark split produced empty train or test data")

    actual = (
        test.sort(["lane_id", "date"])
        .with_columns((pl.int_range(pl.len()).over("lane_id") + 1).alias("horizon"))
        .select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").alias("timestamp"),
            "horizon",
            pl.col("loads").alias("actual"),
        )
    )
    cartoboost_predictions, cartoboost_timing = cartoboost_forecast(train, horizon)
    functime_predictions, functime_timing = functime_forecasts(train, horizon)
    statsforecast_predictions, statsforecast_timing = statsforecast_forecasts(train, horizon)
    predictions = cartoboost_predictions.join(
        functime_predictions,
        on=["series_id", "timestamp", "horizon"],
        how="inner",
    ).join(
        statsforecast_predictions,
        on=["series_id", "timestamp", "horizon"],
        how="inner",
    )
    scored = actual.join(predictions, on=["series_id", "timestamp", "horizon"], how="inner")
    if scored.height != actual.height:
        raise RuntimeError("forecast alignment dropped rows")

    metrics = {
        model: evaluate_metrics(scored, model, train)
        for model in ["cartoboost_lag", *FORECASTING_LIBRARY_BASELINES]
    }
    quality = quality_summary(metrics)
    timing = {
        "models": {
            "cartoboost_lag": cartoboost_timing,
            **functime_timing,
            **statsforecast_timing,
        }
    }
    return metrics, quality, timing


def quality_summary(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    best_library_model = min(FORECASTING_LIBRARY_BASELINES, key=lambda name: metrics[name]["rmse"])
    best_functime = min(FUNCTIME_MODELS, key=lambda name: metrics[name]["rmse"])
    best_statsforecast = min(STATSFORECAST_MODELS, key=lambda name: metrics[name]["rmse"])
    cartoboost_rmse = metrics["cartoboost_lag"]["rmse"]
    library_rmse = metrics[best_library_model]["rmse"]
    rmse_ranking = sorted(metrics, key=lambda name: metrics[name]["rmse"])
    mae_ranking = sorted(metrics, key=lambda name: metrics[name]["mae"])
    wape_ranking = sorted(metrics, key=lambda name: metrics[name]["wape"])
    return {
        "winner": "cartoboost_lag" if cartoboost_rmse < library_rmse else best_library_model,
        "comparison_libraries": list(FORECASTING_LIBRARY_MODELS),
        "forecasting_library_models": FORECASTING_LIBRARY_MODELS,
        "model_libraries": MODEL_LIBRARIES,
        "best_forecasting_library": MODEL_LIBRARIES[best_library_model],
        "best_forecasting_library_model": best_library_model,
        "best_forecasting_library_rmse": library_rmse,
        "best_forecasting_library_mae": metrics[best_library_model]["mae"],
        "best_forecasting_library_wape": metrics[best_library_model]["wape"],
        "best_functime_method": best_functime,
        "best_functime_rmse": metrics[best_functime]["rmse"],
        "best_statsforecast_method": best_statsforecast,
        "best_statsforecast_rmse": metrics[best_statsforecast]["rmse"],
        "rmse_ranking": rmse_ranking,
        "mae_ranking": mae_ranking,
        "wape_ranking": wape_ranking,
        "cartoboost_rmse": cartoboost_rmse,
        "rmse_delta_vs_best_forecasting_library": cartoboost_rmse - library_rmse,
        "rmse_ratio_vs_best_forecasting_library": cartoboost_rmse / library_rmse,
        "rmse_reduction_vs_best_forecasting_library": 1.0 - cartoboost_rmse / library_rmse,
        "cartoboost_mae": metrics["cartoboost_lag"]["mae"],
        "mae_delta_vs_best_forecasting_library": metrics["cartoboost_lag"]["mae"]
        - metrics[best_library_model]["mae"],
        "mae_reduction_vs_best_forecasting_library": 1.0
        - metrics["cartoboost_lag"]["mae"] / metrics[best_library_model]["mae"],
        "cartoboost_wape": metrics["cartoboost_lag"]["wape"],
        "wape_reduction_vs_best_forecasting_library": 1.0
        - metrics["cartoboost_lag"]["wape"] / metrics[best_library_model]["wape"],
    }


def cartoboost_forecast(train: Any, horizon: int) -> tuple[Any, dict[str, float]]:
    pl = require_polars()
    feature_start = perf_counter()
    feature_frame = build_history_features(train).drop_nulls(FEATURE_COLUMNS)
    x = feature_frame.select(FEATURE_COLUMNS).to_numpy()
    y = feature_frame.select("loads").to_numpy().ravel()
    feature_seconds = perf_counter() - feature_start
    model = CartoBoostRegressor(
        n_estimators=80,
        learning_rate=0.08,
        max_depth=5,
        min_samples_leaf=2,
        splitters=["axis_histogram:128", "periodic:7"],
    )
    fit_start = perf_counter()
    model.fit(x, y)
    fit_seconds = perf_counter() - fit_start

    predict_start = perf_counter()
    history = train.clone()
    forecast_frames = []
    for step in range(1, horizon + 1):
        future = next_future_rows(history, step)
        future_features = build_future_features(history, future).drop_nulls(FEATURE_COLUMNS)
        predictions = model.predict(future_features.select(FEATURE_COLUMNS).to_numpy())
        step_forecast = future_features.select(
            pl.col("lane_id").alias("series_id"),
            pl.col("date").alias("timestamp"),
            pl.lit(step).alias("horizon"),
        ).with_columns(pl.Series("cartoboost_lag", predictions))
        forecast_frames.append(step_forecast)
        predicted_future = future_features.with_columns(pl.Series("cartoboost_lag", predictions))
        history = pl.concat(
            [
                history,
                predicted_future.select(
                    "lane_id",
                    "date",
                    pl.col("cartoboost_lag").alias("loads"),
                    *STATIC_COVARIATES,
                ),
            ],
            how="vertical",
        )
    predict_seconds = perf_counter() - predict_start
    timing = {
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_predict_seconds": fit_seconds + predict_seconds,
        "total_seconds": feature_seconds + fit_seconds + predict_seconds,
    }
    return pl.concat(forecast_frames, how="vertical"), timing


def build_history_features(frame: Any) -> Any:
    pl = require_polars()
    return frame.sort(["lane_id", "date"]).with_columns(
        *[
            pl.col("loads").shift(lag).over("lane_id").alias(f"loads_lag_{lag}")
            for lag in [1, 7, 14, 21, 28]
        ],
        *[
            pl.col("loads")
            .shift(1)
            .rolling_mean(window)
            .over("lane_id")
            .alias(f"loads_roll_{window}")
            for window in [7, 14, 28]
        ],
        date_dayofweek=pl.col("date").dt.weekday().cast(pl.Float64),
        date_day=pl.col("date").dt.day().cast(pl.Float64),
        date_month=pl.col("date").dt.month().cast(pl.Float64),
    )


def next_future_rows(history: Any, step: int) -> Any:
    pl = require_polars()
    del step
    return (
        history.sort(["lane_id", "date"])
        .group_by("lane_id", maintain_order=True)
        .tail(1)
        .with_columns((pl.col("date") + pl.duration(days=1)).alias("date"))
        .select("lane_id", "date", *STATIC_COVARIATES)
    )


def build_future_features(history: Any, future: Any) -> Any:
    pl = require_polars()
    pieces = []
    for row in future.iter_rows(named=True):
        lane_history = history.filter(pl.col("lane_id") == row["lane_id"]).sort("date")
        values = dict(row)
        loads = lane_history["loads"].to_list()
        for lag in [1, 7, 14, 21, 28]:
            values[f"loads_lag_{lag}"] = float(loads[-lag]) if len(loads) >= lag else None
        for window in [7, 14, 28]:
            values[f"loads_roll_{window}"] = (
                float(np.mean(loads[-window:])) if len(loads) >= window else None
            )
        timestamp = values["date"]
        values["date_dayofweek"] = float(timestamp.weekday() + 1)
        values["date_day"] = float(timestamp.day)
        values["date_month"] = float(timestamp.month)
        pieces.append(values)
    return pl.DataFrame(pieces)


def functime_forecasts(train: Any, horizon: int) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    from functime.forecasting import lightgbm, ridge, snaive

    y = train.select(
        pl.col("lane_id").alias("entity"),
        pl.col("date").alias("time"),
        pl.col("loads").alias("target"),
    )
    model_specs = {
        "functime_snaive": snaive(freq="1d", sp=7),
        "functime_ridge": ridge(freq="1d", lags=28),
        "functime_lightgbm": lightgbm(
            freq="1d",
            lags=28,
            n_estimators=80,
            learning_rate=0.08,
            max_depth=5,
            min_child_samples=2,
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

    combined = forecasts[0]
    for frame in forecasts[1:]:
        combined = combined.join(frame, on=["series_id", "timestamp", "horizon"], how="inner")
    return combined, timings


def statsforecast_forecasts(train: Any, horizon: int) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, SeasonalNaive
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires statsforecast for the "
            "StatsForecast baselines; run `uv sync --group dev --group bench`."
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
        "statsforecast_seasonal_naive": SeasonalNaive(season_length=7),
        "statsforecast_autoets": AutoETS(season_length=7, model="AAA"),
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

    combined = forecasts[0]
    for frame in forecasts[1:]:
        combined = combined.join(frame, on=["series_id", "timestamp", "horizon"], how="inner")
    return combined, timings


def evaluate_metrics(scored: Any, prediction_col: str, train: Any) -> dict[str, float]:
    pl = require_polars()
    error_frame = scored.select(
        error=pl.col(prediction_col) - pl.col("actual"),
        abs_error=(pl.col(prediction_col) - pl.col("actual")).abs(),
        actual_abs=pl.col("actual").abs(),
        smape_den=(pl.col(prediction_col).abs() + pl.col("actual").abs()),
    )
    mae = float(error_frame.select(pl.col("abs_error").mean()).item())
    rmse = float(error_frame.select((pl.col("error").pow(2).mean()).sqrt()).item())
    wape = float(error_frame.select(pl.col("abs_error").sum() / pl.col("actual_abs").sum()).item())
    smape = float(
        error_frame.filter(pl.col("smape_den") > 0)
        .select((2.0 * pl.col("abs_error") / pl.col("smape_den")).mean())
        .item()
    )
    bias = float(error_frame.select(pl.col("error").mean()).item())
    train_scale = (
        train.sort(["lane_id", "date"])
        .with_columns((pl.col("loads") - pl.col("loads").shift(7).over("lane_id")).abs().alias("d"))
        .select(pl.col("d").mean())
        .item()
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "mase": mae / float(train_scale),
        "wape": wape,
        "smape": smape,
        "bias": bias,
    }


def require_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires polars; run "
            "`uv sync --group dev --group bench`."
        ) from exc
    return pl


def require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires duckdb for --source duckdb; run "
            "`uv sync --group dev --group bench`."
        ) from exc
    return duckdb


def require_pandas_for_benchmark() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires pandas through StatsForecast; run "
            "`uv sync --group dev --group bench`."
        ) from exc
    return pd


if __name__ == "__main__":
    raise SystemExit(main())
