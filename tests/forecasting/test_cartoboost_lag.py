from cartoboost.forecasting.global_models import CartoBoostLagForecaster


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
