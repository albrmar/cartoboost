#!/usr/bin/env python3
"""Benchmark CartoBoost against explicit forecasting libraries.

The fixture is synthetic but domain-shaped: daily pickup/dropoff lane demand with
zone IDs, route distance, airport-lane structure, borough codes, weekly effects,
and deterministic event spikes. The library baselines are functime,
StatsForecast, and Prophet models. The same benchmark can also aggregate cached
NYC TLC taxi trip parquet files into real pickup/dropoff lane demand.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib  # noqa: I001
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001

from cartoboost import __version__
from cartoboost.regressor import CartoBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from scripts.run_nyc_taxi_quality_benchmarks import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    TLC_TRIP_RECORD_PAGE,
    ZoneContext,
    clean_tlc_frame,
    ensure_parquet_files,
    ensure_zone_lookup,
    parse_months,
)

FEATURE_COLUMNS = [
    "loads_lag_1",
    "loads_lag_7",
    "loads_lag_14",
    "loads_roll_7",
    "loads_roll_14",
    "date_dayofweek",
    "date_day",
    "date_month",
    "pickup_zone",
    "dropoff_zone",
    "distance_miles",
    "airport_lane",
    "pickup_borough_code",
]
LAGS = [1, 7, 14]
ROLLING_WINDOWS = [7, 14]
STATIC_COVARIATES = [
    "pickup_zone",
    "dropoff_zone",
    "distance_miles",
    "airport_lane",
    "pickup_borough_code",
]
FUNCTIME_MODELS = ["functime_snaive", "functime_ridge", "functime_lightgbm"]
STATSFORECAST_MODELS = ["statsforecast_seasonal_naive", "statsforecast_autoets"]
PROPHET_MODELS = ["prophet_additive"]
FORECASTING_LIBRARY_MODELS = {
    "functime": FUNCTIME_MODELS,
    "statsforecast": STATSFORECAST_MODELS,
    "prophet": PROPHET_MODELS,
}
MODEL_LIBRARIES = {
    "cartoboost_lag": "cartoboost",
    **{model: "functime" for model in FUNCTIME_MODELS},
    **{model: "statsforecast" for model in STATSFORECAST_MODELS},
    **{model: "prophet" for model in PROPHET_MODELS},
}
FORECASTING_LIBRARY_BASELINES = [*FUNCTIME_MODELS, *STATSFORECAST_MODELS, *PROPHET_MODELS]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CartoBoost global lag forecasts against functime, StatsForecast, and Prophet."
        )
    )
    parser.add_argument("--source", choices=["polars", "duckdb", "nyc-taxi"], default="polars")
    parser.add_argument("--output", default="artifacts/forecasting_library_benchmark_polars.json")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for forecast-vs-actual plots. Defaults to OUTPUT.parent / plots.",
    )
    parser.add_argument("--lanes", type=int, default=36)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1", help="Comma-separated TLC month numbers.")
    parser.add_argument("--taxi-type", default="yellow", choices=["yellow"])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    benchmark_start = perf_counter()
    load_start = perf_counter()
    source = args.source
    table, dataset = load_benchmark_table(args)
    load_seconds = perf_counter() - load_start
    metrics, quality, timing, plot_data = score_models(table, horizon=args.horizon)
    plot_dir = args.plot_dir or Path(args.output).parent / "plots"
    plot_paths = write_forecast_plots(plot_data, metrics, plot_dir, source)
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
        "dataset": dataset,
        "models": ["cartoboost_lag", *FORECASTING_LIBRARY_BASELINES],
        "metrics": metrics,
        "quality": quality,
        "timing": timing,
        "plots": plot_paths,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"quality": quality, "timing": timing}, indent=2, sort_keys=True))
    return 0


def load_benchmark_table(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.source == "nyc-taxi":
        return load_real_taxi_lane_demand(args)
    table = make_fixture(lanes=args.lanes, days=args.days, seed=args.seed)
    dataset = {
        "source": "synthetic_fixture",
        "series": args.lanes,
        "days": args.days,
        "horizon": args.horizon,
        "seed": args.seed,
        "domain": "daily NYC taxi-style pickup/dropoff lane demand",
        "static_covariates": STATIC_COVARIATES,
    }
    return load_fixture_source(args.source, table), dataset


def load_fixture_source(source: str, table: Any) -> Any:
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


def load_real_taxi_lane_demand(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    pandas = require_pandas_for_benchmark()
    pl = require_polars()
    months = parse_months(args.months)
    paths = ensure_parquet_files(
        taxi_type=args.taxi_type,
        year=args.year,
        months=months,
        cache_dir=args.cache_dir,
        no_download=args.no_download,
    )
    zone_lookup = ensure_zone_lookup(cache_dir=args.cache_dir, no_download=args.no_download)
    frames = [
        pandas.read_parquet(
            path,
            columns=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "passenger_count",
                "trip_distance",
                "fare_amount",
                "total_amount",
                "PULocationID",
                "DOLocationID",
            ],
        )
        for path in paths
    ]
    cleaned = clean_tlc_frame(pandas.concat(frames, ignore_index=True))
    cleaned["pickup_date"] = pandas.to_datetime(cleaned["tpep_pickup_datetime"]).dt.normalize()
    cleaned["lane_id"] = (
        "PU"
        + cleaned["PULocationID"].astype(int).astype(str)
        + "->DO"
        + cleaned["DOLocationID"].astype(int).astype(str)
    )
    top_lanes = (
        cleaned.groupby("lane_id")
        .size()
        .sort_values(ascending=False)
        .head(args.lanes)
        .index.tolist()
    )
    if not top_lanes:
        raise ValueError("real taxi aggregation produced no lanes")
    lane_frame = cleaned[cleaned["lane_id"].isin(top_lanes)].copy()
    grouped = (
        lane_frame.groupby(
            ["lane_id", "pickup_date", "PULocationID", "DOLocationID"], as_index=False
        )
        .agg(loads=("lane_id", "size"), distance_miles=("trip_distance", "mean"))
        .rename(columns={"pickup_date": "date"})
    )
    calendar = pandas.date_range(grouped["date"].min(), grouped["date"].max(), freq="D")
    completed = []
    for lane_id, lane in grouped.groupby("lane_id", sort=True):
        pickup_zone = int(lane["PULocationID"].iloc[0])
        dropoff_zone = int(lane["DOLocationID"].iloc[0])
        zone = zone_lookup.get(pickup_zone, ZoneContext(borough_code=7, service_zone_code=6))
        distance = float(lane["distance_miles"].mean())
        filled = pandas.DataFrame({"date": calendar})
        filled["lane_id"] = lane_id
        filled = filled.merge(lane[["date", "loads"]], on="date", how="left")
        filled["loads"] = filled["loads"].fillna(0.0).astype(float)
        filled["pickup_zone"] = pickup_zone
        filled["dropoff_zone"] = dropoff_zone
        filled["distance_miles"] = distance
        filled["airport_lane"] = float(zone.service_zone_code == 1)
        filled["pickup_borough_code"] = float(zone.borough_code)
        completed.append(filled)
    table = (
        pl.from_pandas(pandas.concat(completed, ignore_index=True))
        .with_columns(pl.col("date").cast(pl.Datetime("us")))
        .select(
            "lane_id",
            "date",
            "loads",
            *STATIC_COVARIATES,
        )
    )
    days = int(table.select(pl.col("date").n_unique()).item())
    min_train_days = max(LAGS) + 2
    if days - args.horizon < min_train_days:
        raise ValueError(
            f"real taxi benchmark needs at least {min_train_days + args.horizon} daily "
            f"periods for horizon={args.horizon}; got {days}. Reduce --horizon or add months."
        )
    dataset = {
        "source": "nyc_tlc_trip_records",
        "source_url": TLC_TRIP_RECORD_PAGE,
        "taxi_type": args.taxi_type,
        "year": args.year,
        "months": months,
        "series": len(top_lanes),
        "days": days,
        "horizon": args.horizon,
        "raw_rows": int(sum(len(frame) for frame in frames)),
        "clean_rows": int(len(cleaned)),
        "aggregated_rows": int(table.height),
        "domain": "real daily NYC TLC yellow taxi pickup/dropoff lane demand",
        "static_covariates": STATIC_COVARIATES,
    }
    return table, dataset


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
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, Any], Any]:
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
    prophet_predictions, prophet_timing = prophet_forecasts(train, horizon)
    predictions = (
        cartoboost_predictions.join(
            functime_predictions,
            on=["series_id", "timestamp", "horizon"],
            how="inner",
        )
        .join(
            statsforecast_predictions,
            on=["series_id", "timestamp", "horizon"],
            how="inner",
        )
        .join(
            prophet_predictions,
            on=["series_id", "timestamp", "horizon"],
            how="inner",
        )
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
            **prophet_timing,
        }
    }
    return metrics, quality, timing, scored


def quality_summary(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    best_library_model = min(FORECASTING_LIBRARY_BASELINES, key=lambda name: metrics[name]["rmse"])
    best_functime = min(FUNCTIME_MODELS, key=lambda name: metrics[name]["rmse"])
    best_statsforecast = min(STATSFORECAST_MODELS, key=lambda name: metrics[name]["rmse"])
    best_prophet = min(PROPHET_MODELS, key=lambda name: metrics[name]["rmse"])
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
        "best_prophet_method": best_prophet,
        "best_prophet_rmse": metrics[best_prophet]["rmse"],
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
    if feature_frame.is_empty():
        raise ValueError(
            "CartoBoost lag benchmark has no train rows after lag feature generation; "
            "use more history or a shorter horizon."
        )
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
        *[pl.col("loads").shift(lag).over("lane_id").alias(f"loads_lag_{lag}") for lag in LAGS],
        *[
            pl.col("loads")
            .shift(1)
            .rolling_mean(window)
            .over("lane_id")
            .alias(f"loads_roll_{window}")
            for window in ROLLING_WINDOWS
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
        for lag in LAGS:
            values[f"loads_lag_{lag}"] = float(loads[-lag]) if len(loads) >= lag else None
        for window in ROLLING_WINDOWS:
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
        "functime_ridge": ridge(freq="1d", lags=max(LAGS)),
        "functime_lightgbm": lightgbm(
            freq="1d",
            lags=max(LAGS),
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


def prophet_forecasts(train: Any, horizon: int) -> tuple[Any, dict[str, dict[str, float]]]:
    pl = require_polars()
    pd = require_pandas_for_benchmark()
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "forecasting library benchmark requires prophet for the Prophet baseline; run "
            "`uv sync --group dev --group bench`."
        ) from exc

    forecasts = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for lane_id in train.select("lane_id").unique().sort("lane_id").to_series().to_list():
        lane = (
            train.filter(pl.col("lane_id") == lane_id)
            .sort("date")
            .select(pl.col("date").alias("ds"), pl.col("loads").alias("y"))
            .to_pandas()
        )
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            uncertainty_samples=0,
        )
        fit_start = perf_counter()
        model.fit(lane)
        fit_seconds += perf_counter() - fit_start

        future = pd.DataFrame(
            {
                "ds": pd.date_range(
                    start=lane["ds"].max() + pd.Timedelta(days=1),
                    periods=horizon,
                    freq="D",
                )
            }
        )
        predict_start = perf_counter()
        forecast = model.predict(future)
        predict_seconds += perf_counter() - predict_start
        forecasts.append(
            pl.from_pandas(
                forecast[["ds", "yhat"]]
                .rename(columns={"ds": "timestamp", "yhat": "prophet_additive"})
                .assign(series_id=lane_id)
            )
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
            .sort("timestamp")
            .with_columns((pl.int_range(pl.len()) + 1).alias("horizon"))
            .select("series_id", "timestamp", "horizon", "prophet_additive")
        )

    return pl.concat(forecasts, how="vertical"), {
        "prophet_additive": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "fit_predict_seconds": fit_seconds + predict_seconds,
        }
    }


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


def write_forecast_plots(
    scored: Any, metrics: dict[str, dict[str, float]], output_dir: Path, source: str
) -> list[str]:
    pl = require_polars()
    output_dir.mkdir(parents=True, exist_ok=True)
    top_series = (
        scored.group_by("series_id")
        .agg(pl.col("actual").sum().alias("actual_sum"))
        .sort("actual_sum", descending=True)
        .head(4)
        .select("series_id")
        .to_series()
        .to_list()
    )
    paths = [
        plot_metric_bars(metrics, output_dir / f"{source}_tool_metric_comparison.png"),
        plot_horizon_errors(scored, output_dir / f"{source}_horizon_rmse_by_tool.png"),
        plot_forecast_lines(scored, top_series, output_dir / f"{source}_forecast_lines.png"),
        plot_forecast_scatter(scored, metrics, output_dir / f"{source}_actual_vs_predicted.png"),
    ]
    return [str(path.as_posix()) for path in paths]


def model_color(model: str) -> str:
    model_colors = {
        "cartoboost_lag": "#0f766e",
        "functime_snaive": "#6d28d9",
        "functime_ridge": "#a855f7",
        "functime_lightgbm": "#c084fc",
        "statsforecast_seasonal_naive": "#c2410c",
        "statsforecast_autoets": "#f97316",
        "prophet_additive": "#2563eb",
    }
    return model_colors[model]


def model_label(model: str) -> str:
    return f"{model.replace('_', ' ')} ({MODEL_LIBRARIES[model]})"


def plot_metric_bars(metrics: dict[str, dict[str, float]], output_path: Path) -> Path:
    models = sorted(metrics, key=lambda name: metrics[name]["rmse"])
    metric_names = ["rmse", "mae", "wape"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    y_positions = np.arange(len(models))
    for axis, metric in zip(axes, metric_names, strict=True):
        values = [metrics[model][metric] for model in models]
        bars = axis.barh(
            y_positions,
            values,
            color=[model_color(model) for model in models],
            edgecolor="#111827",
            linewidth=0.5,
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], fontsize=8, padding=2)
        axis.set_title(metric.upper())
        axis.set_yticks(y_positions, [model_label(model) for model in models])
        axis.invert_yaxis()
        axis.tick_params(axis="y", labelsize=9)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel(metric.upper())
    handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=library)
        for library, color in {
            "cartoboost": "#0f766e",
            "functime": "#7c3aed",
            "statsforecast": "#c2410c",
            "prophet": "#2563eb",
        }.items()
    ]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Forecasting tool quality on the same held-out lane days", y=1.06, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_horizon_errors(scored: Any, output_path: Path) -> Path:
    models = ["cartoboost_lag", *FORECASTING_LIBRARY_BASELINES]
    frame = scored.to_pandas()
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for model in models:
        horizon_rmse = (
            frame.assign(squared_error=(frame[model] - frame["actual"]) ** 2)
            .groupby("horizon", as_index=False)["squared_error"]
            .mean()
        )
        axis.plot(
            horizon_rmse["horizon"],
            np.sqrt(horizon_rmse["squared_error"]),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=model_color(model),
            alpha=0.95,
            label=model,
        )
    axis.set_title("RMSE by forecast horizon")
    axis.set_xlabel("days ahead")
    axis.set_ylabel("RMSE in daily trips")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_forecast_lines(scored: Any, series_ids: list[str], output_path: Path) -> Path:
    pl = require_polars()
    frame = scored.filter(pl.col("series_id").is_in(series_ids)).to_pandas()
    models = ["cartoboost_lag", "prophet_additive", "statsforecast_autoets", "functime_snaive"]
    fig, axes = plt.subplots(
        len(series_ids), 1, figsize=(11, max(3, 2.6 * len(series_ids))), sharex=True
    )
    if len(series_ids) == 1:
        axes = [axes]
    for axis, series_id in zip(axes, series_ids, strict=True):
        lane = frame[frame["series_id"] == series_id].sort_values("timestamp")
        axis.plot(lane["timestamp"], lane["actual"], color="#111827", linewidth=2.2, label="actual")
        for model in models:
            axis.plot(
                lane["timestamp"],
                lane[model],
                color=model_color(model),
                linewidth=1.4,
                label=model,
            )
        axis.set_title(series_id, fontsize=10)
        axis.set_ylabel("daily trips")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("forecast date")
    fig.suptitle("Forecast horizon: actual vs model predictions", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_forecast_scatter(
    scored: Any, metrics: dict[str, dict[str, float]], output_path: Path
) -> Path:
    frame = scored.to_pandas()
    models = sorted(metrics, key=lambda name: metrics[name]["rmse"])[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    actual_min = float(frame["actual"].min())
    actual_max = float(frame["actual"].max())
    for axis, model in zip(axes.ravel(), models, strict=True):
        axis.scatter(frame["actual"], frame[model], s=10, alpha=0.45, color=model_color(model))
        axis.plot([actual_min, actual_max], [actual_min, actual_max], color="#111827", linewidth=1)
        axis.set_title(f"{model} RMSE={metrics[model]['rmse']:.3f}", fontsize=10)
        axis.grid(alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("predicted daily trips")
    for axis in axes[-1, :]:
        axis.set_xlabel("actual daily trips")
    fig.suptitle("Forecast accuracy across held-out lane days", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


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
            "forecasting library benchmark requires pandas through StatsForecast and Prophet; run "
            "`uv sync --group dev --group bench`."
        ) from exc
    return pd


if __name__ == "__main__":
    raise SystemExit(main())
