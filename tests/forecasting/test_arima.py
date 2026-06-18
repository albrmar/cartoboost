import pytest
from cartoboost.forecasting.local.arima import AutoARIMAForecaster


def test_auto_arima_rejects_python_fallback_policy():
    with pytest.raises(ValueError, match="error_policy='raise'"):
        AutoARIMAForecaster(error_policy="fallback")


def test_auto_arima_fit_predict_uses_rust_binding():
    model = AutoARIMAForecaster(max_p=2, max_d=1, max_q=1)

    model.fit([10.0, 11.0, 13.0, 16.0, 20.0])
    result = model.predict(2)

    assert [row[3] for row in result.predictions()] == ["auto_arima", "auto_arima"]
    assert [row[2] for row in result.predictions()] == [1, 2]
