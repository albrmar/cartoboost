import pytest
from cartoboost.forecasting.local import SeasonalNaiveForecaster


def test_seasonal_naive_validates_season_length():
    with pytest.raises(ValueError, match="positive integer"):
        SeasonalNaiveForecaster(season_length=0)


def test_seasonal_naive_fit_predicts_with_rust_binding():
    result = SeasonalNaiveForecaster(season_length=3).fit([10.0, 20.0, 30.0]).predict(4)

    assert [row[4] for row in result.predictions()] == [10.0, 20.0, 30.0, 10.0]
