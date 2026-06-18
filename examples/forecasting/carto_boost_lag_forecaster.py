from __future__ import annotations

import pandas as pd
from cartoboost.forecasting.global_models import CartoBoostLagForecaster
from cartoboost.forecasting.lag_features import (
    CalendarFeatureConfig,
    LagFeatureConfig,
    RollingFeatureConfig,
)


def build_pickup_demand() -> pd.DataFrame:
    rows = []
    for pickup_zone, base, capacity in [("zone_4", 42.0, 120.0), ("zone_79", 65.0, 180.0)]:
        for hour in range(36):
            timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour)
            commute_bump = 8.0 if timestamp.hour in {8, 17, 18} else 0.0
            weekend_discount = -5.0 if timestamp.dayofweek >= 5 else 0.0
            rows.append(
                {
                    "PULocationID": pickup_zone,
                    "pickup_hour": timestamp,
                    "pickup_trips": base + commute_bump + weekend_discount + 0.4 * hour,
                    "planned_drivers": 25.0 + (hour % 6),
                    "zone_capacity": capacity,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    training = build_pickup_demand()
    future = pd.DataFrame(
        {
            "PULocationID": ["zone_4", "zone_4", "zone_79", "zone_79"],
            "pickup_hour": pd.to_datetime(
                [
                    "2026-01-02 12:00",
                    "2026-01-02 13:00",
                    "2026-01-02 12:00",
                    "2026-01-02 13:00",
                ]
            ),
            "planned_drivers": [25.0, 26.0, 25.0, 26.0],
            "zone_capacity": [120.0, 120.0, 180.0, 180.0],
        }
    )

    forecaster = CartoBoostLagForecaster(
        time_col="pickup_hour",
        target_col="pickup_trips",
        panel_cols=["PULocationID"],
        lag_config=LagFeatureConfig(lags=[1, 2, 24]),
        rolling_config=RollingFeatureConfig(windows=[3, 6], aggregations=["mean"]),
        calendar_config=CalendarFeatureConfig(features=["hour", "dayofweek"]),
        static_cols=["zone_capacity"],
        known_future_cols=["planned_drivers"],
        regressor_params={
            "n_estimators": 30,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "splitters": ["axis"],
        },
    ).fit(training)

    result = forecaster.predict(future)
    print(result.frame[["PULocationID", "pickup_hour", "forecast"]].to_string(index=False))


if __name__ == "__main__":
    main()
