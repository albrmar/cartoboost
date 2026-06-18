from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from cartoboost.forecasting.global_models import CartoBoostLagForecaster  # noqa: E402
from cartoboost.forecasting.lag_features import (  # noqa: E402
    CalendarFeatureConfig,
    LagFeatureConfig,
    RollingFeatureConfig,
)


def _taxi_demand_frame() -> pd.DataFrame:
    rows = []
    for zone, base in [("A", 10.0), ("B", 40.0)]:
        for hour in range(8):
            rows.append(
                {
                    "PULocationID": zone,
                    "pickup_hour": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour),
                    "pickup_trips": base + hour * 2.0,
                    "planned_drivers": 20.0 + hour,
                    "zone_capacity": 100.0 if zone == "A" else 200.0,
                }
            )
    return pd.DataFrame(rows)


def test_cartoboost_lag_forecaster_fits_and_predicts_recursive_panel_forecasts() -> None:
    train = _taxi_demand_frame()
    future = pd.DataFrame(
        {
            "PULocationID": ["A", "A", "B", "B"],
            "pickup_hour": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 09:00",
                    "2026-01-01 08:00",
                    "2026-01-01 09:00",
                ]
            ),
            "planned_drivers": [28.0, 29.0, 28.0, 29.0],
            "zone_capacity": [100.0, 100.0, 200.0, 200.0],
        }
    )
    forecaster = CartoBoostLagForecaster(
        time_col="pickup_hour",
        target_col="pickup_trips",
        panel_cols=["PULocationID"],
        lag_config=LagFeatureConfig(lags=[1, 2]),
        rolling_config=RollingFeatureConfig(windows=[3], aggregations=["mean"]),
        calendar_config=CalendarFeatureConfig(features=["hour"]),
        static_cols=["zone_capacity"],
        known_future_cols=["planned_drivers"],
        regressor_params={
            "n_estimators": 8,
            "learning_rate": 0.3,
            "max_depth": 2,
            "min_samples_leaf": 1,
            "splitters": ["axis"],
        },
    ).fit(train)

    result = forecaster.predict(future)

    assert result.predictions.shape == (4,)
    assert list(result.frame["forecast"]) == pytest.approx(result.predictions)
    assert "pickup_trips_lag_1" in result.feature_names
    assert result.regressor_metadata["backend"] == "rust"
    assert np.all(np.isfinite(result.predictions))
    assert result.frame.loc[0, "forecast"] < result.frame.loc[2, "forecast"]


def test_cartoboost_lag_forecaster_rejects_historical_only_future_covariates() -> None:
    train = _taxi_demand_frame()
    future = pd.DataFrame(
        {
            "PULocationID": ["A"],
            "pickup_hour": [pd.Timestamp("2026-01-01 08:00")],
            "planned_drivers": [28.0],
            "observed_queue": [7.0],
        }
    )
    forecaster = CartoBoostLagForecaster(
        time_col="pickup_hour",
        target_col="pickup_trips",
        panel_cols=["PULocationID"],
        known_future_cols=["planned_drivers"],
        historical_covariate_cols=["observed_queue"],
        regressor_params={"n_estimators": 2, "min_samples_leaf": 1},
    ).fit(train)

    with pytest.raises(ValueError, match="historical-only"):
        forecaster.predict(future)
