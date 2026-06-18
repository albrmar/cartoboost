import pytest
from cartoboost.forecasting.local import OptimizedThetaForecaster, ThetaForecaster


def test_theta_validates_parameters():
    with pytest.raises(ValueError, match="theta must be positive"):
        ThetaForecaster(theta=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ThetaForecaster(alpha=2.0)


def test_theta_fit_predicts_with_rust_binding():
    result = ThetaForecaster(theta=2.0, alpha=0.4).fit([3.0, 4.0, 6.0, 9.0]).predict(2)

    assert len(result.predictions()) == 2
    assert {row[3] for row in result.predictions()} == {"theta"}
    assert all(row[4] > 0.0 for row in result.predictions())


def test_optimized_theta_fit_predicts_with_rust_binding():
    result = (
        OptimizedThetaForecaster(theta_grid=(1.0, 2.0), alpha_grid=(0.2, 0.8))
        .fit([1.0, 1.4, 2.1, 3.0])
        .predict(2)
    )

    assert len(result.predictions()) == 2
    assert {row[3] for row in result.predictions()} == {"optimized_theta"}
