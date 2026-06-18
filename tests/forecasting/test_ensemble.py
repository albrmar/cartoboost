import pytest
from cartoboost.forecasting.ensemble import (
    BacktestWeightedEnsembleForecaster,
    WeightedEnsembleForecaster,
)


class LastValueModel:
    def fit(self, y, **kwargs):
        self.y = y
        return self

    def predict(self, horizon, **kwargs):
        if isinstance(self.y, dict):
            return {key: [values[-1]] * horizon for key, values in self.y.items()}
        return [self.y[-1]] * horizon


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def fit(self, y, **kwargs):
        return self

    def predict(self, horizon, **kwargs):
        return [self.value] * horizon


class IntervalModel:
    def __init__(self, mean, lower, upper):
        self.mean = mean
        self.lower = lower
        self.upper = upper

    def predict(self, horizon, **kwargs):
        return {
            "mean": [self.mean] * horizon,
            "lower": [self.lower] * horizon,
            "upper": [self.upper] * horizon,
        }


def test_weighted_ensemble_uses_normalized_fixed_weights():
    model = WeightedEnsembleForecaster(
        models={"low": ConstantModel(10), "high": ConstantModel(20)},
        weights={"low": 1, "high": 3},
    )

    assert model.predict(2) == [17.5, 17.5]


def test_weighted_ensemble_supports_panel_forecasts_and_bounds():
    ensemble = WeightedEnsembleForecaster(
        models={"last": LastValueModel(), "zero": ConstantModel(0)},
        weights={"last": 1, "zero": 1},
        lower_bound=0,
        upper_bound=100,
    )
    ensemble.fit({"pickup_1": [4, 6], "pickup_2": [8, 10]})

    assert ensemble.predict(2) == {
        "pickup_1": [3.0, 3.0],
        "pickup_2": [5.0, 5.0],
    }


def test_weighted_ensemble_combines_explicit_intervals():
    ensemble = WeightedEnsembleForecaster(
        models={
            "a": IntervalModel(mean=10, lower=8, upper=12),
            "b": IntervalModel(mean=20, lower=18, upper=22),
        },
        weights={"a": 1, "b": 1},
        interval_level=0.8,
    )

    result = ensemble.predict(1)

    assert result["mean"] == [15.0]
    assert result["lower"] == [13.0]
    assert result["upper"] == [17.0]
    assert result["metadata"]["weights"] == {"a": 0.5, "b": 0.5}


def test_backtest_weighted_ensemble_learns_inverse_error_weights():
    ensemble = BacktestWeightedEnsembleForecaster(
        models={"last": LastValueModel(), "bad": ConstantModel(100)},
        backtest_horizon=1,
        min_train_size=3,
    )

    ensemble.fit([1, 2, 3, 4, 5, 6])

    assert ensemble.weights_["last"] > ensemble.weights_["bad"]
    assert ensemble.predict(2)[0] == pytest.approx(
        ensemble.weights_["last"] * 6 + ensemble.weights_["bad"] * 100
    )


def test_ensemble_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="same names"):
        WeightedEnsembleForecaster(
            models={"a": ConstantModel(1), "b": ConstantModel(2)},
            weights={"a": 1},
        )
