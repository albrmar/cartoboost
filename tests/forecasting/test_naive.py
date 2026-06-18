import pytest
from cartoboost.forecasting.local import NaiveForecaster


def test_naive_requires_fit_before_predict():
    with pytest.raises(RuntimeError, match="fitted"):
        NaiveForecaster().predict(2)


def test_naive_fit_predicts_with_rust_binding():
    result = NaiveForecaster().fit([1.0, 2.0, 4.0]).predict(2)

    assert result.columns() == ["series_id", "timestamp", "horizon", "model", "prediction"]
    assert [row[4] for row in result.predictions()] == [4.0, 4.0]
