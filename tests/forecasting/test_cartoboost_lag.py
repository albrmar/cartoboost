from cartoboost.forecasting.global_models import CartoBoostLagForecaster


def test_cartoboost_lag_fit_predicts_with_rust_binding():
    result = (
        CartoBoostLagForecaster(lags=[1], rolling_windows=[], calendar_features=False)
        .fit({"pickup_1": [10, 11, 12, 13], "pickup_2": [20, 22, 24, 26]})
        .predict(2)
    )

    assert len(result.predictions()) == 4
    assert {row[3] for row in result.predictions()} == {"cartoboost_lag"}
