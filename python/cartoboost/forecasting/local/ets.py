from __future__ import annotations

from typing import Any

import numpy as np

from .naive import (
    ForecastResult,
    NaiveForecaster,
    _as_series_matrix,
    _future_timestamps,
    _restore_shape,
    _validate_horizon,
    _with_residual_intervals,
)


class ETSForecaster(NaiveForecaster):
    """Exponential smoothing forecaster backed by statsmodels."""

    def __init__(
        self,
        *,
        trend: str | None = None,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        damped_trend: bool = False,
    ) -> None:
        if trend not in {None, "add", "mul", "additive", "multiplicative"}:
            raise ValueError("trend must be None, 'add', 'mul', 'additive', or 'multiplicative'")
        if seasonal not in {None, "add", "mul", "additive", "multiplicative"}:
            raise ValueError("seasonal must be None, 'add', 'mul', 'additive', or 'multiplicative'")
        if seasonal is not None and (seasonal_periods is None or int(seasonal_periods) <= 1):
            raise ValueError("seasonal_periods must be greater than 1 when seasonal is set")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = None if seasonal_periods is None else int(seasonal_periods)
        self.damped_trend = bool(damped_trend)
        self.is_fitted_ = False

    def fit(
        self,
        y: Any,
        *,
        timestamps: Any | None = None,
        series_ids: list[Any] | None = None,
    ) -> ETSForecaster:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError as exc:
            raise ImportError(
                "ETSForecaster requires statsmodels. Install it with `pip install statsmodels`."
            ) from exc

        matrix, fitted_timestamps, fitted_series_ids, single_series = _as_series_matrix(
            y, timestamps=timestamps, series_ids=series_ids
        )
        models = []
        fitted_columns = []
        residual_columns = []
        for col in range(matrix.shape[1]):
            model = ExponentialSmoothing(
                matrix[:, col],
                trend=_statsmodels_component(self.trend),
                seasonal=_statsmodels_component(self.seasonal),
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
                initialization_method="estimated",
            ).fit(optimized=True)
            fitted = np.asarray(model.fittedvalues, dtype=float)
            models.append(model)
            fitted_columns.append(fitted)
            residual_columns.append(matrix[:, col] - fitted)

        fitted_matrix = np.column_stack(fitted_columns)
        residual_matrix = np.column_stack(residual_columns)
        self.training_values_ = matrix
        self.models_ = models
        self.fitted_values_ = _restore_shape(fitted_matrix, single_series)
        self.residuals_ = _restore_shape(residual_matrix, single_series)
        self.timestamps_ = fitted_timestamps
        self.series_ids_ = fitted_series_ids
        self.single_series_ = single_series
        self.metadata_ = {
            "model": self.__class__.__name__,
            "n_obs": int(matrix.shape[0]),
            "n_series": int(matrix.shape[1]),
            "series_ids": list(fitted_series_ids),
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "damped_trend": self.damped_trend,
            "interval_method": "residual_normal_fallback",
            "has_timestamps": fitted_timestamps is not None,
        }
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        *,
        return_interval: bool = False,
        level: float = 0.95,
    ) -> np.ndarray | ForecastResult:
        self._check_is_fitted()
        horizon = _validate_horizon(horizon)
        mean = np.column_stack(
            [np.asarray(model.forecast(horizon), dtype=float) for model in self.models_]
        )
        restored = _restore_shape(mean, self.single_series_)
        if not return_interval:
            return restored
        lower, upper = _with_residual_intervals(mean, _matrix_residuals(self.residuals_), level)
        metadata = dict(self.metadata_)
        metadata["interval_level"] = float(level)
        return ForecastResult(
            mean=restored,
            lower=_restore_shape(lower, self.single_series_),
            upper=_restore_shape(upper, self.single_series_),
            timestamps=_future_timestamps(self.timestamps_, horizon),
            metadata=metadata,
        )


def _statsmodels_component(value: str | None) -> str | None:
    if value == "additive":
        return "add"
    if value == "multiplicative":
        return "mul"
    return value


def _matrix_residuals(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr
