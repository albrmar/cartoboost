import numpy as np
import pytest
from cartoboost.forecasting.local import SeasonalNaiveForecaster


def test_seasonal_naive_repeats_last_cycle():
    model = SeasonalNaiveForecaster(season_length=3).fit([10.0, 20.0, 30.0, 11.0, 21.0, 31.0])

    np.testing.assert_allclose(model.predict(5), [11.0, 21.0, 31.0, 11.0, 21.0])
    assert model.metadata_["season_length"] == 3


def test_seasonal_naive_validates_training_length():
    with pytest.raises(ValueError, match="season_length"):
        SeasonalNaiveForecaster(season_length=4).fit([1.0, 2.0, 3.0])
