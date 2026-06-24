import pytest
from cartoboost.forecasting.local import OptimizedThetaForecaster, ThetaForecaster


def test_theta_validates_parameters():
    with pytest.raises(ValueError, match="theta must be positive"):
        ThetaForecaster(theta=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ThetaForecaster(alpha=2.0)
    with pytest.raises(ValueError, match="season_length is required"):
        ThetaForecaster(seasonality="additive")
    with pytest.raises(ValueError, match="prediction_interval_levels"):
        ThetaForecaster(prediction_interval_levels=[1.2])


def test_optimized_theta_validates_parameters():
    with pytest.raises(ValueError, match="theta_grid"):
        OptimizedThetaForecaster(theta_grid=(0.0,))
    with pytest.raises(ValueError, match="alpha_grid"):
        OptimizedThetaForecaster(alpha_grid=(1.2,))
    with pytest.raises(ValueError, match="season_length is required"):
        OptimizedThetaForecaster(seasonality="multiplicative")


def test_theta_converts_series_and_delegates_to_native(install_fake_native):
    native = install_fake_native("ThetaForecaster")

    result = ThetaForecaster(theta=2.0, alpha=0.4).fit([3.0, 4.0, 6.0, 9.0]).predict(2)

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "theta": 2.0,
            "alpha": 0.4,
            "season_length": None,
            "seasonality": None,
            "prediction_interval_levels": (),
        },
    )
    assert native.calls[1][1].rows[-1] == ("__single__", "1970-01-04T00:00:00", 9.0)
    assert native.calls[2] == ("predict", (2,), {})


def test_optimized_theta_converts_series_and_delegates_to_native(install_fake_native):
    native = install_fake_native("OptimizedThetaForecaster")

    result = (
        OptimizedThetaForecaster(theta_grid=(1.0, 2.0), alpha_grid=(0.2, 0.8))
        .fit([1.0, 1.4, 2.1, 3.0])
        .predict(2)
    )

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "theta_grid": (1.0, 2.0),
            "alpha_grid": (0.2, 0.8),
            "season_length": None,
            "seasonality": None,
            "prediction_interval_levels": (),
        },
    )
    assert native.calls[1][1].rows[-1] == ("__single__", "1970-01-04T00:00:00", 3.0)
    assert native.calls[2] == ("predict", (2,), {})
