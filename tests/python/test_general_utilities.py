import math

import pytest
from cartoboost import (
    arima_forecast,
    auto_arima_forecast,
    croston_forecast,
    ets_forecast,
    kalman_filter,
    local_level_kalman_filter,
    local_linear_trend_kalman_forecast,
    naive_forecast,
    ordinary_kriging_predict,
    sba_forecast,
    seasonal_naive_forecast,
    series_forecast,
    theta_forecast,
    tsb_forecast,
)


def assert_finite_forecast(values, horizon):
    assert len(values) == horizon
    assert all(math.isfinite(value) for value in values)


def test_general_series_forecast_wrappers_are_native_backed():
    assert naive_forecast([1.0, 2.0, 3.0], 2) == [3.0, 3.0]
    assert seasonal_naive_forecast([10.0, 20.0, 30.0, 40.0], 3, season_length=2) == [
        30.0,
        40.0,
        30.0,
    ]
    assert_finite_forecast(theta_forecast([10.0, 11.0, 13.0, 16.0], 2), 2)
    assert_finite_forecast(ets_forecast([10.0, 11.0, 13.0, 16.0], 2), 2)
    assert_finite_forecast(arima_forecast([10.0, 11.0, 13.0, 16.0, 20.0], 2, p=1, d=1), 2)
    assert_finite_forecast(auto_arima_forecast([10.0, 11.0, 13.0, 16.0, 20.0], 2), 2)
    assert_finite_forecast(
        local_linear_trend_kalman_forecast([10.0, 12.0, 14.0, 16.0], 2),
        2,
    )
    assert_finite_forecast(series_forecast("local_level_kalman", [10.0, 12.0, 14.0], 2), 2)


def test_general_intermittent_demand_utilities():
    values = [0.0, 0.0, 5.0, 0.0, 0.0, 7.0, 0.0]

    assert_finite_forecast(croston_forecast(values, 3), 3)
    assert_finite_forecast(sba_forecast(values, 3), 3)
    assert_finite_forecast(tsb_forecast(values, 3), 3)
    assert sba_forecast(values, 1)[0] < croston_forecast(values, 1)[0]

    with pytest.raises(ValueError, match="non-negative"):
        croston_forecast([1.0, -1.0], 1)


def test_general_local_level_kalman_filter():
    result = local_level_kalman_filter([12.0, 14.0, 16.0], horizon=2)

    assert set(result) == {"final_state", "estimates", "forecast"}
    assert len(result["estimates"]) == 2
    assert result["forecast"] == [result["final_state"]["level"]] * 2


def test_general_kalman_filter_tracks_trend_and_forecasts():
    result = kalman_filter(
        [12.0, 14.0, 16.0, 18.0],
        level_process_variance=0.01,
        trend_process_variance=0.001,
        observation_variance=0.1,
        horizon=2,
    )

    assert set(result) == {"final_state", "estimates", "forecast"}
    assert result["final_state"]["trend"] > 0.0
    assert len(result["estimates"]) == 3
    assert result["forecast"][1] > result["forecast"][0]


def test_general_kalman_filter_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least two observations"):
        kalman_filter([1.0])

    with pytest.raises(ValueError, match="must be positive and finite"):
        kalman_filter([1.0, 2.0], level_process_variance=0.0)


def test_general_ordinary_kriging_predicts_known_coordinate():
    predictions = ordinary_kriging_predict(
        observations=[(0.0, 0.0, 12.0), (10.0, 0.0, 42.0)],
        targets=[(0.0, 0.0), (10.0, 0.0)],
        range=1.0,
        nugget=1.0e-9,
    )

    assert len(predictions) == 2
    assert abs(predictions[0]["mean"] - 12.0) < 1.0e-4
    assert abs(sum(predictions[0]["weights"]) - 1.0) < 1.0e-8


def test_general_ordinary_kriging_rejects_bad_inputs():
    with pytest.raises(ValueError, match="observations must not be empty"):
        ordinary_kriging_predict([], [(0.0, 0.0)])

    with pytest.raises(ValueError, match="nugget"):
        ordinary_kriging_predict([(0.0, 0.0, 1.0)], [(0.0, 0.0)], nugget=-1.0)
