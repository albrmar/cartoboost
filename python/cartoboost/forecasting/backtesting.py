"""Rolling-origin backtesting orchestration."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metrics import ForecastMetricSet
from .schema import ForecastFrame
from .splitters import ForecastFold, RollingOriginSplitter


@dataclass
class BacktestFoldResult:
    fold: ForecastFold
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold.fold_id,
            "metrics": _jsonable(self.metrics),
            "predictions": _jsonable(self.predictions),
            "metadata": _jsonable(
                {
                    **self.fold.metadata,
                    "train_start": self.fold.train_start,
                    "train_end": self.fold.train_end,
                    "validation_start": self.fold.validation_start,
                    "validation_end": self.fold.validation_end,
                    "horizon": self.fold.horizon,
                    "step": self.fold.step,
                }
            ),
        }


@dataclass
class BacktestResult:
    folds: list[BacktestFoldResult] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, float]:
        if not self.folds:
            return {}
        keys = {
            key
            for fold in self.folds
            for key, value in fold.metrics.items()
            if isinstance(value, int | float) and np.isfinite(value)
        }
        return {
            key: float(np.mean([fold.metrics[key] for fold in self.folds if key in fold.metrics]))
            for key in sorted(keys)
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "metrics": _jsonable(self.metrics),
            "folds": [fold.to_json() for fold in self.folds],
        }

    def to_pandas(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_pandas requires pandas; install pandas to use this helper"
            ) from exc
        rows = [row for fold in self.folds for row in fold.predictions]
        return pd.DataFrame(rows)


class RollingOriginBacktester:
    """Fit a fresh model per fold and score exact-horizon predictions."""

    def __init__(
        self,
        *,
        splitter: RollingOriginSplitter | None = None,
        horizon: int | None = None,
        min_train_size: int = 1,
        step_size: int = 1,
        max_train_size: int | None = None,
        metric_set: ForecastMetricSet | None = None,
        target_col: str = "target",
        timestamp_col: str = "timestamp",
        series_id_col: str | None = "series_id",
        feature_cols: list[str] | None = None,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        if splitter is None:
            if horizon is None:
                raise ValueError("either splitter or horizon is required")
            splitter = RollingOriginSplitter(
                horizon=horizon,
                step=step_size,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                timestamp_col=timestamp_col,
                series_id_col=series_id_col,
            )
        self.splitter = splitter
        self.metric_set = metric_set or ForecastMetricSet()
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.series_id_col = series_id_col
        self.feature_cols = feature_cols
        self.model_factory = model_factory

    def evaluate(self, model: Any, frame: ForecastFrame) -> BacktestResult:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("evaluate requires a ForecastFrame")
        data = frame.to_pandas()
        splitter = RollingOriginSplitter(
            horizon=self.splitter.horizon,
            step=self.splitter.step,
            min_train_size=self.splitter.min_train_size,
            max_train_size=self.splitter.max_train_size,
            n_splits=self.splitter.n_splits,
            timestamp_col=frame.timestamp_col,
            series_id_col=frame.series_id_col,
            window=self.splitter.window,
        )
        folds: list[BacktestFoldResult] = []
        for fold in splitter.split(data):
            train = _take(data, fold.train_indices)
            validation = _take(data, fold.validation_indices)
            train_frame = ForecastFrame.from_pandas(
                train,
                timestamp_col=frame.timestamp_col,
                target_col=frame.target_col,
                series_id_col=frame.series_id_col,
                freq=frame.freq,
                static_covariates=frame.static_covariates,
                known_future_covariates=frame.known_future_covariates,
                historical_covariates=frame.historical_covariates,
                allow_irregular=frame.allow_irregular,
            )
            fitted = self._new_model(model)
            fitted.fit(train_frame)
            forecast = fitted.predict(fold.horizon)
            forecast_data = forecast.to_pandas().rename(columns={"mean": "prediction"})
            actual = validation.copy()
            if frame.series_id_col is None:
                actual["series_id"] = "__single__"
            else:
                actual["series_id"] = actual[frame.series_id_col]
            actual = actual.rename(
                columns={frame.timestamp_col: "timestamp", frame.target_col: "actual"}
            )
            actual["horizon"] = _horizon_numbers(validation, frame.timestamp_col)
            merged = actual[["series_id", "timestamp", "horizon", "actual"]].merge(
                forecast_data[["series_id", "timestamp", "horizon", "prediction"]],
                on=["series_id", "timestamp", "horizon"],
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(actual):
                raise ValueError("forecast rows did not align to validation rows")
            metrics = self.metric_set.evaluate(
                merged["actual"],
                merged["prediction"],
                horizon=merged["horizon"],
                series_id=merged["series_id"],
                y_train=train[frame.target_col],
            )
            rows = []
            for row in merged.to_dict(orient="records"):
                rows.append({"fold_id": fold.fold_id, **row})
            folds.append(BacktestFoldResult(fold=fold, metrics=metrics, predictions=rows))
        return BacktestResult(folds=folds)

    def run(self, model: Any, data: Any) -> BacktestResult:
        folds: list[BacktestFoldResult] = []
        for fold in self.splitter.split(data):
            train = _take(data, fold.train_indices)
            validation = _take(data, fold.validation_indices)
            y_train = np.asarray(_column(train, self.target_col), dtype=float)
            y_validation = np.asarray(_column(validation, self.target_col), dtype=float)
            fitted = self._new_model(model)
            fitted.fit(_features(train, self.target_col, self.feature_cols), y_train)
            predictions = np.asarray(
                fitted.predict(_features(validation, self.target_col, self.feature_cols)),
                dtype=float,
            )
            if predictions.shape != y_validation.shape:
                raise ValueError("model predictions must match the exact validation horizon shape")

            horizon = _horizon_numbers(validation, self.timestamp_col)
            series = _optional_column(validation, self.series_id_col)
            metrics = self.metric_set.evaluate(
                y_validation,
                predictions,
                horizon=horizon,
                series_id=series,
                y_train=y_train,
            )
            rows = _prediction_rows(
                fold,
                validation,
                y_validation,
                predictions,
                horizon,
                self.timestamp_col,
                self.series_id_col,
            )
            folds.append(BacktestFoldResult(fold=fold, metrics=metrics, predictions=rows))
        return BacktestResult(folds=folds)

    def _new_model(self, model: Any) -> Any:
        if self.model_factory is not None:
            return self.model_factory()
        try:
            from sklearn.base import clone

            return clone(model)
        except Exception:
            return deepcopy(model)


def _features(data: Any, target_col: str, feature_cols: list[str] | None) -> Any:
    if feature_cols is not None:
        return data[feature_cols]
    if hasattr(data, "drop"):
        return data.drop(columns=[target_col])
    arr = np.asarray(data)
    return np.delete(arr, -1, axis=1)


def _take(data: Any, indices: np.ndarray) -> Any:
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[indices]


def _column(data: Any, name: str) -> Any:
    try:
        return data[name]
    except Exception as exc:
        raise ValueError(f"data must contain column {name!r}") from exc


def _optional_column(data: Any, name: str | None) -> Any | None:
    if name is None:
        return None
    try:
        return data[name]
    except Exception:
        return None


def _horizon_numbers(validation: Any, timestamp_col: str) -> np.ndarray:
    timestamps = np.asarray(_column(validation, timestamp_col))
    unique = np.unique(timestamps)
    mapping = {value: i + 1 for i, value in enumerate(unique)}
    return np.asarray([mapping[value] for value in timestamps], dtype=int)


def _prediction_rows(
    fold: ForecastFold,
    validation: Any,
    actual: np.ndarray,
    prediction: np.ndarray,
    horizon: np.ndarray,
    timestamp_col: str,
    series_id_col: str | None,
) -> list[dict[str, Any]]:
    timestamps = np.asarray(_column(validation, timestamp_col))
    series = _optional_column(validation, series_id_col)
    series_values = np.asarray(series) if series is not None else np.array([None] * actual.size)
    return [
        {
            "fold_id": fold.fold_id,
            "series_id": _jsonable(series_values[i]),
            "timestamp": _jsonable(timestamps[i]),
            "horizon": int(horizon[i]),
            "actual": float(actual[i]),
            "prediction": float(prediction[i]),
        }
        for i in range(actual.size)
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value
