import pytest
from cartoboost.forecasting.local.ets import ETSForecaster


def test_ets_validates_parameters():
    with pytest.raises(ValueError, match="trend"):
        ETSForecaster(trend="bad")
    with pytest.raises(ValueError, match="additive"):
        ETSForecaster(seasonal="mul", seasonal_periods=2)
    with pytest.raises(ValueError, match="seasonal_periods"):
        ETSForecaster(seasonal="add", seasonal_periods=1)
    with pytest.raises(ValueError, match="alpha"):
        ETSForecaster(alpha=0.0)
    with pytest.raises(ValueError, match="beta"):
        ETSForecaster(beta=1.2)
    with pytest.raises(ValueError, match="gamma"):
        ETSForecaster(seasonal="add", seasonal_periods=2, gamma=-0.1)


def test_ets_fit_predict_uses_rust_binding():
    model = ETSForecaster()

    model.fit([10.0, 12.0, 14.0, 16.0])
    result = model.predict(2)

    assert [row[3] for row in result.predictions()] == ["ets", "ets"]
    assert [row[2] for row in result.predictions()] == [1, 2]
