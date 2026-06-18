from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cartoboost.forecasting import (
    ExpandingWindowSplitter,
    ForecastMetricSet,
    RollingOriginBacktester,
)


class MeanFareModel:
    fit_count = 0

    def __init__(self) -> None:
        self.mean_ = None

    def fit(self, rows, targets):
        type(self).fit_count += 1
        self.seen_max_timestamp_ = rows["timestamp"].max()
        self.mean_ = float(np.mean(targets))
        return self

    def predict(self, rows):
        if rows["timestamp"].min() <= self.seen_max_timestamp_:
            raise AssertionError("validation leaked into training")
        return np.full(len(rows), self.mean_)


class BadHorizonModel(MeanFareModel):
    def predict(self, rows):
        return np.asarray([1.0])


def taxi_trips() -> pd.DataFrame:
    rows = []
    for pickup in ["pickup_1", "pickup_2"]:
        for hour in range(6):
            rows.append(
                {
                    "series_id": pickup,
                    "timestamp": hour,
                    "trip_distance": float(hour + 1),
                    "fare": float(10 + hour),
                }
            )
    return pd.DataFrame(rows)


def test_backtester_fits_fresh_model_per_fold_and_returns_structured_results() -> None:
    MeanFareModel.fit_count = 0
    splitter = ExpandingWindowSplitter(
        horizon=2,
        step=2,
        min_train_size=3,
        timestamp_col="timestamp",
        series_id_col="series_id",
    )
    backtester = RollingOriginBacktester(
        splitter=splitter,
        metric_set=ForecastMetricSet(),
        target_col="fare",
        timestamp_col="timestamp",
        series_id_col="series_id",
        feature_cols=["timestamp", "trip_distance", "series_id"],
    )

    result = backtester.run(MeanFareModel(), taxi_trips())

    assert MeanFareModel.fit_count == len(result.folds)
    assert result.metrics["mae"] > 0
    as_json = result.to_json()
    assert as_json["folds"][0]["fold_id"] == "fold_0000"
    assert {"actual", "prediction", "series_id", "timestamp", "horizon"} <= set(
        as_json["folds"][0]["predictions"][0]
    )
    assert len(result.to_pandas()) == sum(len(fold.predictions) for fold in result.folds)


def test_backtester_rejects_models_that_do_not_predict_exact_validation_shape() -> None:
    splitter = ExpandingWindowSplitter(horizon=2, min_train_size=3, timestamp_col="timestamp")
    backtester = RollingOriginBacktester(
        splitter=splitter,
        target_col="fare",
        timestamp_col="timestamp",
        series_id_col="series_id",
        feature_cols=["timestamp", "trip_distance"],
    )

    with pytest.raises(ValueError, match="exact validation horizon"):
        backtester.run(BadHorizonModel(), taxi_trips())
