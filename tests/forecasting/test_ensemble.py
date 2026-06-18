import pytest
from cartoboost.forecasting.ensemble import (
    BacktestWeightedEnsembleForecaster,
    WeightedEnsembleForecaster,
)
from cartoboost.forecasting.local import NaiveForecaster, SeasonalNaiveForecaster


def test_weighted_ensemble_requires_models():
    with pytest.raises(ValueError, match="requires at least one model"):
        WeightedEnsembleForecaster(models={})


def test_weighted_ensemble_validates_weights_match_models():
    with pytest.raises(ValueError, match="weights must match models exactly"):
        WeightedEnsembleForecaster(
            models={"last": NaiveForecaster()},
            weights={"other": 1.0},
        )


def test_weighted_ensemble_rejects_unsupported_intervals():
    with pytest.raises(NotImplementedError, match="prediction intervals are not supported"):
        WeightedEnsembleForecaster(
            models={"last": NaiveForecaster()},
            interval_level=0.9,
        )


def test_weighted_ensemble_fits_native_members_for_single_series():
    model = WeightedEnsembleForecaster(
        models={
            "last": NaiveForecaster(),
            "seasonal": SeasonalNaiveForecaster(season_length=2),
        },
        weights={"last": 1.0, "seasonal": 3.0},
    )

    model.fit([10.0, 12.0, 14.0])
    result = model.predict(2)

    assert [row[-1] for row in result.predictions()] == [12.5, 14.0]
    assert model.get_metadata()["weights"] == {"last": 0.25, "seasonal": 0.75}


def test_weighted_ensemble_aligns_panel_series():
    model = WeightedEnsembleForecaster(
        models={
            "last": NaiveForecaster(),
            "seasonal": SeasonalNaiveForecaster(season_length=2),
        }
    )

    model.fit({"PU1->DO2": [10.0, 12.0, 14.0], "PU9->DO8": [30.0, 28.0, 26.0]})
    result = model.predict(1)

    assert result.predictions() == [
        ("PU1->DO2", "1970-01-04T00:00:00", 1, "weighted_ensemble", 13.0),
        ("PU9->DO8", "1970-01-04T00:00:00", 1, "weighted_ensemble", 27.0),
    ]


def test_backtest_weighted_ensemble_fit_requires_rust_binding():
    with pytest.raises(
        NotImplementedError,
        match="Rust binding.*BacktestWeightedEnsembleForecaster",
    ):
        BacktestWeightedEnsembleForecaster(models={"last": NaiveForecaster()}).fit([1.0, 2.0])
