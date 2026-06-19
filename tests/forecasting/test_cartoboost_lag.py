from __future__ import annotations

import pytest
from cartoboost.forecasting.global_models import CartoBoostLagForecaster
from cartoboost.forecasting.lag_features import (
    CalendarFeatureConfig,
    LagFeatureConfig,
    RollingFeatureConfig,
)

pd = pytest.importorskip("pandas")


def test_cartoboost_lag_converts_panel_and_delegates_to_native(install_fake_native):
    native = install_fake_native("CartoBoostLagForecaster")

    result = (
        CartoBoostLagForecaster(lags=[1], rolling_windows=[], calendar_features=False)
        .fit({"pickup_1": [10, 11, 12, 13], "pickup_2": [20, 22, 24, 26]})
        .predict(2)
    )

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {"lags": [1], "rolling_windows": [], "calendar_features": False},
    )
    assert native.calls[1][1].rows[:2] == [
        ("pickup_1", "1970-01-01T00:00:00", 10.0),
        ("pickup_1", "1970-01-02T00:00:00", 11.0),
    ]
    assert native.calls[2] == ("predict", (2,), {})


def test_cartoboost_lag_converts_supported_feature_configs(install_fake_native):
    native = install_fake_native("CartoBoostLagForecaster")

    CartoBoostLagForecaster(
        lag_config=LagFeatureConfig(
            lags=[1, 24],
            difference_lags=[24],
            rolling_trend_windows=[3],
        ),
        rolling_config=RollingFeatureConfig(windows=[3], aggregations=["mean"]),
        calendar_config=CalendarFeatureConfig(features=["dayofweek", "month", "day"]),
    ).fit({"pickup_1": [10, 11, 12, 13]})

    assert native.calls[0] == (
        "init",
        {
            "lags": [1, 24],
            "difference_lags": [24],
            "rolling_trend_windows": [3],
            "rolling_windows": [3],
            "calendar_features": True,
        },
    )


def test_cartoboost_lag_passes_supported_regressor_params(install_fake_native):
    native = install_fake_native("CartoBoostLagForecaster")

    CartoBoostLagForecaster(
        lags=[1, 7],
        rolling_windows=[7],
        difference_lags=[7],
        rolling_trend_windows=[7],
        calendar_features=True,
        trend_features=True,
        target_mode="delta_from_last",
        regressor_params={
            "n_estimators": 32,
            "learning_rate": 0.08,
            "max_depth": 3,
            "min_samples_leaf": 4,
            "min_gain": 0.0,
            "splitters": ["axis"],
        },
    ).fit({"pickup_1": [10, 11, 12, 13, 15, 18, 21, 24]})

    assert native.calls[0] == (
        "init",
        {
            "lags": [1, 7],
            "rolling_windows": [7],
            "difference_lags": [7],
            "rolling_trend_windows": [7],
            "calendar_features": True,
            "trend_features": True,
            "target_mode": "delta_from_last",
            "n_estimators": 32,
            "learning_rate": 0.08,
            "max_depth": 3,
            "min_samples_leaf": 4,
            "min_gain": 0.0,
            "splitters": ["axis"],
        },
    )


def test_cartoboost_lag_rejects_unsupported_regressor_params() -> None:
    with pytest.raises(ValueError, match="unsupported CartoBoostLagForecaster regressor_params"):
        CartoBoostLagForecaster(regressor_params={"subsample": 0.8})


def test_cartoboost_lag_rejects_unsupported_calendar_config() -> None:
    with pytest.raises(ValueError, match="unsupported: \\['hour'\\]"):
        CartoBoostLagForecaster(calendar_config=CalendarFeatureConfig(features=["hour"]))


def test_cartoboost_lag_dataframe_coercion_rejects_duplicate_panel_timestamps(
    install_fake_native,
):
    install_fake_native("CartoBoostLagForecaster")
    training = pd.DataFrame(
        {
            "PULocationID": ["zone_4", "zone_4"],
            "pickup_hour": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00"]),
            "pickup_trips": [42.0, 45.0],
        }
    )
    forecaster = CartoBoostLagForecaster(
        time_col="pickup_hour",
        target_col="pickup_trips",
        panel_cols=["PULocationID"],
    )

    with pytest.raises(ValueError, match="unique timestamps within each panel"):
        forecaster.fit(training)
