from __future__ import annotations

import pandas as pd
from cartoboost.forecasting import AutoForecaster, ForecastFrame


def test_auto_forecaster_delegates_to_native_auto_model(install_fake_native):
    native = install_fake_native("AutoForecastModel")
    frame = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "pickup_hour": pd.date_range("2026-01-01", periods=8, freq="D"),
                "pickup_trips": [10.0, 11.0, 13.0, 16.0, 18.0, 21.0, 23.0, 26.0],
            }
        ),
        timestamp_col="pickup_hour",
        target_col="pickup_trips",
        freq="D",
    )

    result = (
        AutoForecaster(
            season_length=7,
            objective="wape",
            validation_window=2,
            baseline_displacement_gain=0.04,
            hard_winner_relative_gain=0.06,
            min_blend_weight=0.2,
            max_blend_weight=0.8,
            max_direct_horizon=14,
            n_estimators=16,
        )
        .fit(frame)
        .predict(2)
    )

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "lags": [1, 2, 3, 7, 14, 28],
            "rolling_windows": [7, 14, 28],
            "calendar_features": True,
            "season_length": 7,
            "validation_window": 2,
            "objective": "wape",
            "baseline_displacement_gain": 0.04,
            "hard_winner_relative_gain": 0.06,
            "min_blend_weight": 0.2,
            "max_blend_weight": 0.8,
            "max_direct_horizon": 14,
            "n_estimators": 16,
        },
    )
    assert native.calls[1][0] == "fit"
    assert native.calls[2] == ("predict", (2,), {})
