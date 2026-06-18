import pytest
from cartoboost.forecasting.local import AutoARIMAForecaster


def test_auto_arima_rejects_python_fallback_policy():
    with pytest.raises(NotImplementedError, match="fallback policies"):
        AutoARIMAForecaster(error_policy="fallback")


def test_auto_arima_fit_requires_rust_binding():
    with pytest.raises(NotImplementedError, match="Rust binding.*AutoARIMAForecaster"):
        AutoARIMAForecaster().fit([1.0, 2.0, 3.0])
