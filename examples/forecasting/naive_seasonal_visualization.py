from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from cartoboost.forecasting import ForecastFrame, NaiveForecaster, SeasonalNaiveForecaster


def synthetic_zone_pickups(hours: int) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    rows: list[dict[str, Any]] = []
    zones = [
        ("PU132", "JFK Airport", 78.0, 0.14),
        ("PU236", "Upper East Side", 116.0, 0.07),
    ]
    hourly_profile = [
        -22.0,
        -28.0,
        -34.0,
        -38.0,
        -33.0,
        -18.0,
        14.0,
        36.0,
        42.0,
        27.0,
        10.0,
        4.0,
        7.0,
        9.0,
        5.0,
        3.0,
        18.0,
        31.0,
        35.0,
        24.0,
        12.0,
        2.0,
        -8.0,
        -16.0,
    ]

    for zone_id, zone_name, baseline, trend in zones:
        zone_shift = 6.0 if zone_id == "PU132" else -3.0
        for hour in range(hours):
            pickup_hour = start + timedelta(hours=hour)
            hour_of_day = pickup_hour.hour
            day = hour // 24
            smooth_variation = 2.2 * math.sin(hour / 3.5 + zone_shift)
            pickup_count = (
                baseline + hourly_profile[hour_of_day] + trend * hour + 1.6 * day + smooth_variation
            )
            rows.append(
                {
                    "PULocationID": zone_id,
                    "pickup_zone": zone_name,
                    "pickup_hour": pickup_hour,
                    "pickup_count": pickup_count,
                }
            )
    return pd.DataFrame(rows)


def forecast_panel(
    table: pd.DataFrame,
    model: NaiveForecaster | SeasonalNaiveForecaster,
    train_hours: int,
    horizon: int,
) -> pd.DataFrame:
    train = table.groupby("PULocationID", sort=False).head(train_hours).reset_index(drop=True)
    frame = ForecastFrame.from_pandas(
        train,
        timestamp_col="pickup_hour",
        target_col="pickup_count",
        series_id_col="PULocationID",
        freq="h",
    )
    model.fit(frame)
    rows = [
        {
            "PULocationID": series_id,
            "pickup_hour": pd.Timestamp(timestamp),
            "horizon": horizon_step,
            "model": model_name,
            "prediction": mean_value,
        }
        for series_id, timestamp, horizon_step, model_name, mean_value in model.predict(
            horizon
        ).predictions()
    ]
    return pd.DataFrame(rows)


def metric_summary(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    joined = actual.merge(
        forecast,
        on=["PULocationID", "pickup_hour"],
        validate="one_to_one",
    )
    errors = [row.prediction - row.pickup_count for row in joined.itertuples()]
    return {
        "mae": mean(abs(error) for error in errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
    }


def write_plot(
    table: pd.DataFrame,
    naive_forecast: pd.DataFrame,
    seasonal_forecast: pd.DataFrame,
    train_hours: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zone_id = "PU132"
    observed = table[table["PULocationID"] == zone_id]
    naive = naive_forecast[naive_forecast["PULocationID"] == zone_id]
    seasonal = seasonal_forecast[seasonal_forecast["PULocationID"] == zone_id]
    split_time = observed["pickup_hour"].iloc[train_hours - 1]

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(
        observed["pickup_hour"],
        observed["pickup_count"],
        label="observed pickups",
        color="#111827",
        linewidth=1.8,
    )
    axis.plot(
        naive["pickup_hour"],
        naive["prediction"],
        label="naive",
        color="#dc2626",
        linewidth=2.0,
    )
    axis.plot(
        seasonal["pickup_hour"],
        seasonal["prediction"],
        label="seasonal naive",
        color="#2563eb",
        linewidth=2.0,
    )
    axis.axvline(split_time, color="#6b7280", linestyle="--", linewidth=1)
    axis.set_title("JFK Pickup Demand: Naive vs Seasonal Naive")
    axis.set_xlabel("pickup hour")
    axis.set_ylabel("pickup count")
    axis.legend()
    axis.grid(alpha=0.22)
    fig.autofmt_xdate()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=120)
    parser.add_argument("--train-hours", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("target/examples/naive_seasonal_visualization.png"),
    )
    args = parser.parse_args()

    if args.hours <= args.train_hours:
        raise ValueError("--hours must be greater than --train-hours")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.train_hours < 24:
        raise ValueError("--train-hours must be at least 24 for season_length=24")
    if args.train_hours + args.horizon > args.hours:
        raise ValueError("--train-hours + --horizon must be <= --hours")

    table = synthetic_zone_pickups(args.hours)
    actual = table.groupby("PULocationID", sort=False).nth(
        list(range(args.train_hours, args.train_hours + args.horizon))
    )
    actual = actual.reset_index()

    naive_forecast = forecast_panel(
        table,
        NaiveForecaster(),
        train_hours=args.train_hours,
        horizon=args.horizon,
    )
    seasonal_forecast = forecast_panel(
        table,
        SeasonalNaiveForecaster(season_length=24),
        train_hours=args.train_hours,
        horizon=args.horizon,
    )
    write_plot(table, naive_forecast, seasonal_forecast, args.train_hours, args.output)

    naive_metrics = metric_summary(actual, naive_forecast)
    seasonal_metrics = metric_summary(actual, seasonal_forecast)
    payload = {
        "task": "example_taxi_zone_naive_seasonal_forecast",
        "rows": int(len(table)),
        "zones": int(table["PULocationID"].nunique()),
        "train_hours": int(args.train_hours),
        "horizon": int(args.horizon),
        "season_length": 24,
        "naive": naive_metrics,
        "seasonal_naive": seasonal_metrics,
        "rmse_improvement": naive_metrics["rmse"] - seasonal_metrics["rmse"],
        "plot": str(args.output),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
