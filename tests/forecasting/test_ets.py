import pytest
from cartoboost.forecasting.local import ETSForecaster


def test_ets_validates_parameters():
    with pytest.raises(ValueError, match="trend"):
        ETSForecaster(trend="bad")
    with pytest.raises(ValueError, match="seasonal_periods"):
        ETSForecaster(seasonal="add", seasonal_periods=1)


def test_ets_fit_requires_rust_binding():
    with pytest.raises(NotImplementedError, match="Rust binding.*ETSForecaster"):
        ETSForecaster().fit([1.0, 2.0, 3.0])
