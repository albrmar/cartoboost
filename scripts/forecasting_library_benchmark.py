#!/usr/bin/env python3
"""Benchmark CartoBoost forecasting against StatsForecast on geotemporal demand.

The fixture is synthetic but domain-shaped: daily pickup/dropoff lane demand with
zone IDs, route distance, airport-lane structure, borough codes, weekly effects,
and deterministic event spikes. The benchmark can source the same table through
Polars or DuckDB so the comparison is not a pandas-only workflow.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cartoboost import __version__
from cartoboost.forecasting import CartoBoostLagForecaster, ForecastFrame, ForecastMetricSet

STATIC_COVARIATES = [
    "pickup_zone",
    "dropoff_zone",
    "distance_miles",
    "airport_lane",
    "pickup_borough_code",
]
STATFORECAST_MODELS = ["SeasonalNaive", "AutoTheta", "AutoETS"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CartoBoost global lag forecasts against StatsForecast."
    )
    parser.add_argument("--backend", choices=["polars", "duckdb"], default="polars")
    parser.add_argument("--output", default="artifacts/forecasting_library_benchmark.json")
    parser.add_argument("--lanes", type=int, default=36)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    table = load_fixture(args.backend, lanes=args.lanes, days=args.days, seed=args.seed)
    metrics, summary = score_models(table, horizon=args.horizon)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "benchmark": "geotemporal_lane_demand_statsforecast",
        "backend": args.backend,
        "known_forecasting_library": "statsforecast",
        "dataset": {
            "series": args.lanes,
            "days": args.days,
            "horizon": args.horizon,
            "seed": args.seed,
            "domain": "daily NYC taxi-style pickup/dropoff lane demand",
            "static_covariates": STATIC_COVARIATES,
        },
        "models": ["cartoboost_lag", *STATFORECAST_MODELS],
        "metrics": metrics,
        "comparison": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def load_fixture(backend: str, *, lanes: int, days: int, seed: int) -> Any:
    table = make_fixture(lanes=lanes, days=days, seed=seed)
    if backend == "polars":
        return table
    if backend == "duckdb":
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
    raise ValueError(f"unsupported backend {backend!r}")


def make_fixture(*, lanes: int, days: int, seed: int) -> Any:
    pl = require_polars()
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
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
            timestamp = start + pd.Timedelta(days=day)
            weekly = [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 2.0][timestamp.dayofweek]
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
                    "date": timestamp.to_pydatetime(),
                    "loads": float(demand + lane_noise),
                    "pickup_zone": pickup_zone,
                    "dropoff_zone": dropoff_zone,
                    "distance_miles": float(distance),
                    "airport_lane": airport_lane,
                    "pickup_borough_code": pickup_borough_code,
                }
            )
    return pl.DataFrame(rows)


def score_models(table: Any, *, horizon: int) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    pl = require_polars()
    cutoff = table.select(pl.col("date").unique().sort()).to_series()[-horizon]
    train = table.filter(pl.col("date") < cutoff)
    test = table.filter(pl.col("date") >= cutoff)
    if train.is_empty() or test.is_empty():
        raise ValueError("benchmark split produced empty train or test data")

    actual = to_pandas_ns(test).rename(columns={"date": "timestamp", "loads": "actual"})
    actual["series_id"] = actual["lane_id"]
    actual = actual.sort_values(["series_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    actual["horizon"] = actual.groupby("series_id").cumcount() + 1

    merged = actual[["series_id", "timestamp", "horizon", "actual"]]
    merged = merged.merge(
        cartoboost_forecast(train, horizon),
        on=["series_id", "timestamp", "horizon"],
        how="inner",
    )
    merged = merged.merge(
        statsforecast_forecasts(train, horizon),
        on=["series_id", "timestamp", "horizon"],
        how="inner",
    )
    if len(merged) != len(actual):
        raise RuntimeError("forecast alignment dropped rows")

    train_pd = to_pandas_ns(train)
    metric_set = ForecastMetricSet(seasonal_period=7)
    metrics = {}
    for model_name in ["cartoboost_lag", *STATFORECAST_MODELS]:
        evaluated = metric_set.evaluate(
            merged["actual"],
            merged[model_name],
            horizon=merged["horizon"],
            series_id=merged["series_id"],
            y_train=train_pd["loads"],
        )
        metrics[model_name] = {
            metric: float(evaluated[metric])
            for metric in ("mae", "rmse", "mase", "wape", "smape", "bias")
            if metric in evaluated and np.isfinite(evaluated[metric])
        }

    best_known = min(STATFORECAST_MODELS, key=lambda name: metrics[name]["rmse"])
    cartoboost_rmse = metrics["cartoboost_lag"]["rmse"]
    known_rmse = metrics[best_known]["rmse"]
    summary = {
        "winner": "cartoboost_lag" if cartoboost_rmse < known_rmse else best_known,
        "best_known_method": best_known,
        "cartoboost_rmse": cartoboost_rmse,
        "best_known_rmse": known_rmse,
        "rmse_delta_vs_best_known": cartoboost_rmse - known_rmse,
        "rmse_ratio_vs_best_known": cartoboost_rmse / known_rmse,
        "cartoboost_mae": metrics["cartoboost_lag"]["mae"],
        "best_known_mae": metrics[best_known]["mae"],
        "mae_delta_vs_best_known": metrics["cartoboost_lag"]["mae"] - metrics[best_known]["mae"],
    }
    return metrics, summary


def cartoboost_forecast(train: Any, horizon: int) -> pd.DataFrame:
    train_pd = to_pandas_ns(train)
    frame = ForecastFrame.from_pandas(
        train_pd,
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
        static_covariates=STATIC_COVARIATES,
    )
    model = CartoBoostLagForecaster(
        lags=[1, 7, 14, 21, 28],
        rolling_windows=[7, 14, 28],
        calendar_features=True,
        regressor_params={
            "n_estimators": 80,
            "learning_rate": 0.08,
            "max_depth": 5,
            "min_samples_leaf": 2,
            "splitters": ["axis_histogram:128", "periodic:7"],
        },
    )
    model.fit(frame)
    forecast = model.predict(horizon).to_pandas()
    forecast = forecast.rename(columns={"mean": "cartoboost_lag"})
    return forecast[["series_id", "timestamp", "horizon", "cartoboost_lag"]]


def statsforecast_forecasts(train: Any, horizon: int) -> pd.DataFrame:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta, SeasonalNaive

    pl = require_polars()
    sf_train = train.select(
        pl.col("lane_id").alias("unique_id"),
        pl.col("date").alias("ds"),
        pl.col("loads").alias("y"),
    )
    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=7),
            AutoTheta(season_length=7),
            AutoETS(season_length=7),
        ],
        freq="1d",
        n_jobs=1,
    )
    forecast = sf.forecast(df=sf_train, h=horizon)
    forecast_pd = forecast.to_pandas().rename(columns={"unique_id": "series_id", "ds": "timestamp"})
    forecast_pd["timestamp"] = pd.to_datetime(forecast_pd["timestamp"]).astype("datetime64[ns]")
    forecast_pd["horizon"] = forecast_pd.groupby("series_id").cumcount() + 1
    return forecast_pd[["series_id", "timestamp", "horizon", *STATFORECAST_MODELS]]


def to_pandas_ns(frame: Any) -> pd.DataFrame:
    data = frame.to_pandas()
    data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
    return data


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
            "forecasting library benchmark requires duckdb for --backend duckdb; run "
            "`uv sync --group dev --group bench`."
        ) from exc
    return duckdb


if __name__ == "__main__":
    raise SystemExit(main())
