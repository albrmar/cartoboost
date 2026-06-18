import numpy as np
import pytest
from cartoboost.forecasting.local import OptimizedThetaForecaster, ThetaForecaster


def test_theta_forecaster_is_deterministic_and_exposes_residuals():
    y = np.array([3.0, 4.0, 6.0, 9.0, 13.0, 18.0])
    model = ThetaForecaster(theta=2.0, alpha=0.4).fit(y)

    first = model.predict(4)
    second = model.predict(4)

    np.testing.assert_allclose(first, second)
    assert model.fitted_values_.shape == y.shape
    assert model.residuals_.shape == y.shape
    assert model.metadata_["theta"] == 2.0


def test_theta_supports_panel_additive_seasonality():
    y = np.array(
        [
            [10.0, 20.0],
            [14.0, 24.0],
            [11.0, 22.0],
            [15.0, 26.0],
            [12.0, 24.0],
            [16.0, 28.0],
        ]
    )
    model = ThetaForecaster(season_length=2, seasonality="additive").fit(y)

    forecast = model.predict(3)

    assert forecast.shape == (3, 2)
    assert model.seasonal_pattern_.shape == (2, 2)
    assert model.metadata_["seasonality"] == "additive"


def test_theta_multiplicative_requires_positive_values():
    with pytest.raises(ValueError, match="strictly positive"):
        ThetaForecaster(season_length=2, seasonality="multiplicative").fit([1.0, 2.0, 0.0, 3.0])


def test_optimized_theta_selects_from_grid_deterministically():
    y = np.array([1.0, 1.4, 2.1, 3.1, 4.2, 5.4])
    model = OptimizedThetaForecaster(theta_grid=(1.0, 2.0), alpha_grid=(0.2, 0.8)).fit(y)

    assert model.theta in {1.0, 2.0}
    assert model.alpha in {0.2, 0.8}
    assert len(model.validation_scores_) == 4
    assert model.metadata_["optimized"] is True
