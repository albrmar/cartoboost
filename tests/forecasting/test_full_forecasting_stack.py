from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from cartoboost.forecasting import (  # noqa: E402
    CartoBoostLagForecaster,
    ForecastArtifact,
    ForecastArtifactManifest,
    ForecastFrame,
    ForecastingConfig,
    RollingOriginBacktester,
    ThetaForecaster,
    WeightedEnsembleForecaster,
)


def _panel_frame() -> ForecastFrame:
    rows = []
    for lane, base in [("JFK->LGA", 20.0), ("LGA->EWR", 35.0)]:
        for day in range(36):
            rows.append(
                {
                    "lane_id": lane,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                    "loads": base + day * 0.5 + (day % 7) * 1.5,
                }
            )
    return ForecastFrame.from_pandas(
        pd.DataFrame(rows),
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
    )


def test_full_forecasting_stack_round_trip_and_leakage_boundaries(tmp_path) -> None:
    frame = _panel_frame()

    theta = ThetaForecaster(season_length=7, prediction_interval_levels=[0.8, 0.95])
    theta.fit(frame)
    theta_forecast = theta.predict(horizon=4)
    expected_cols = [
        "series_id",
        "timestamp",
        "horizon",
        "model",
        "mean",
        "lower_80",
        "upper_80",
        "lower_95",
        "upper_95",
    ]
    assert list(theta_forecast.to_pandas().columns) == expected_cols

    lag = CartoBoostLagForecaster(
        lags=[1, 7],
        rolling_windows=[7],
        calendar_features=True,
        regressor_params={
            "n_estimators": 6,
            "learning_rate": 0.2,
            "max_depth": 2,
            "min_samples_leaf": 1,
            "splitters": ["axis"],
        },
        prediction_interval_levels=[0.8, 0.95],
    )
    lag.fit(frame)
    lag_forecast = lag.predict(horizon=4)
    assert list(lag_forecast.to_pandas().columns) == expected_cols

    backtester = RollingOriginBacktester(horizon=4, min_train_size=21, step_size=7)
    backtest = backtester.evaluate(ThetaForecaster(season_length=7), frame)
    assert backtest.folds
    for fold in backtest.folds:
        assert fold.fold.train_end < fold.fold.validation_start

    ensemble = WeightedEnsembleForecaster(
        models={
            "theta": ThetaForecaster(season_length=7, prediction_interval_levels=[0.8]),
            "lag": CartoBoostLagForecaster(
                lags=[1, 7],
                rolling_windows=[7],
                regressor_params={
                    "n_estimators": 4,
                    "max_depth": 2,
                    "min_samples_leaf": 1,
                    "splitters": ["axis"],
                },
                prediction_interval_levels=[0.8],
            ),
        },
        weights={"theta": 0.6, "lag": 0.4},
    )
    ensemble.fit(frame)
    ensemble_forecast = ensemble.predict(4)
    table = ensemble_forecast.to_pandas()
    assert list(table[["series_id", "timestamp", "horizon", "model", "mean"]].columns) == [
        "series_id",
        "timestamp",
        "horizon",
        "model",
        "mean",
    ]
    assert len(table) == 8

    manifest = ForecastArtifactManifest(
        model_name="weighted_ensemble",
        horizon=4,
        columns=tuple(table.columns),
        forecast_path="forecast.csv",
        forecast_format="csv",
        freq="D",
        target_column="loads",
        time_column="date",
        panel_columns=("lane_id",),
        backtest_metrics=backtest.metrics,
        ensemble_metadata=ensemble.metadata_,
    )
    artifact = ForecastArtifact(table.to_dict(orient="records"), manifest)
    artifact.save(tmp_path)
    loaded = ForecastArtifact.load(tmp_path)
    assert loaded.manifest.model_name == "weighted_ensemble"
    assert len(loaded.forecast) == len(table)


def test_documented_toml_shape_parses() -> None:
    config = ForecastingConfig.from_toml(
        """
        horizon = 14
        freq = "D"
        target_column = "loads"
        time_column = "date"
        panel_columns = ["lane_id"]

        [[models]]
        name = "theta"

        [models.params]
        season_length = 7
        prediction_interval_levels = [0.8, 0.95]
        """
    )

    assert config.horizon == 14
    assert config.models[0].name == "theta"
