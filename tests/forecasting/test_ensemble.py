import pytest
from cartoboost.forecasting.ensemble import (
    BacktestWeightedEnsembleForecaster,
    WeightedEnsembleForecaster,
)


class ConstantModel:
    def predict(self, horizon, **kwargs):
        return [1.0] * horizon


def test_weighted_ensemble_requires_models():
    with pytest.raises(ValueError, match="requires at least one model"):
        WeightedEnsembleForecaster(models={})


def test_weighted_ensemble_fit_requires_rust_binding():
    with pytest.raises(NotImplementedError, match="Rust binding.*WeightedEnsembleForecaster"):
        WeightedEnsembleForecaster(models={"constant": ConstantModel()}).fit([1.0, 2.0])


def test_backtest_weighted_ensemble_fit_requires_rust_binding():
    with pytest.raises(
        NotImplementedError,
        match="Rust binding.*BacktestWeightedEnsembleForecaster",
    ):
        BacktestWeightedEnsembleForecaster(models={"constant": ConstantModel()}).fit([1.0, 2.0])
