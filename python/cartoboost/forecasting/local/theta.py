from __future__ import annotations

from typing import Any

import numpy as np

from ..schema import ForecastFrame
from .naive import (
    NaiveForecaster,
    _as_series_matrix,
    _matrix_from_frame,
    _restore_shape,
    _table_forecast_result,
    _validate_horizon,
)


class ThetaForecaster(NaiveForecaster):
    """Deterministic theta-method forecaster with optional seasonal adjustment."""

    def __init__(
        self,
        *,
        theta: float = 2.0,
        alpha: float = 0.2,
        season_length: int | None = None,
        seasonality: str | None = None,
        seasonal_mode: str | None = None,
        prediction_interval_levels: list[float] | tuple[float, ...] = (),
    ) -> None:
        if seasonal_mode is not None:
            seasonality = seasonal_mode
        theta = float(theta)
        alpha = float(alpha)
        if theta <= 0:
            raise ValueError("theta must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if seasonality not in {None, "additive", "multiplicative"}:
            raise ValueError("seasonality must be None, 'additive', or 'multiplicative'")
        if season_length is not None and int(season_length) <= 1:
            raise ValueError("season_length must be greater than 1 when provided")
        super().__init__(prediction_interval_levels=prediction_interval_levels)
        self.theta = theta
        self.alpha = alpha
        self.season_length = None if season_length is None else int(season_length)
        self.seasonality = seasonality

    def fit(
        self,
        y: Any,
        *,
        timestamps: Any | None = None,
        series_ids: list[Any] | None = None,
    ) -> ThetaForecaster:
        if isinstance(y, ForecastFrame):
            matrix, fitted_timestamps, fitted_series_ids, single_series = _matrix_from_frame(y)
            self.forecast_frame_metadata_ = y.to_metadata()
            self.forecast_freq_ = y.freq
            self.forecast_timestamp_col_ = y.timestamp_col
            self.forecast_target_col_ = y.target_col
            self.forecast_series_id_col_ = y.series_id_col
        else:
            matrix, fitted_timestamps, fitted_series_ids, single_series = _as_series_matrix(
                y, timestamps=timestamps, series_ids=series_ids
            )
            self.forecast_frame_metadata_ = None
            self.forecast_freq_ = None
            self.forecast_timestamp_col_ = None
            self.forecast_target_col_ = None
            self.forecast_series_id_col_ = None
        if self.seasonality and self.season_length is None:
            raise ValueError("season_length is required when seasonality is enabled")
        if self.seasonality and matrix.shape[0] < self.season_length * 2:
            raise ValueError("seasonal theta requires at least two full seasonal cycles")
        if self.seasonality == "multiplicative" and np.any(matrix <= 0):
            raise ValueError("multiplicative seasonality requires strictly positive y")

        adjusted, seasonal_pattern = self._deseasonalize(matrix)
        self.seasonal_pattern_ = seasonal_pattern
        components = [
            _fit_theta_series(adjusted[:, i], self.theta, self.alpha)
            for i in range(adjusted.shape[1])
        ]
        fitted_adjusted = np.column_stack([component["fitted"] for component in components])
        fitted = self._reseasonalize(fitted_adjusted, np.arange(matrix.shape[0]))
        residuals = matrix - fitted

        self.training_values_ = matrix
        self.adjusted_values_ = adjusted
        self.components_ = components
        self.fitted_values_ = _restore_shape(fitted, single_series)
        self.residuals_ = _restore_shape(residuals, single_series)
        self.timestamps_ = fitted_timestamps
        self.series_ids_ = fitted_series_ids
        self.single_series_ = single_series
        self.metadata_ = {
            "model": self.__class__.__name__,
            "n_obs": int(matrix.shape[0]),
            "n_series": int(matrix.shape[1]),
            "series_ids": list(fitted_series_ids),
            "theta": self.theta,
            "alpha": self.alpha,
            "season_length": self.season_length,
            "seasonality": self.seasonality,
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
        future_frame: ForecastFrame | None = None,
    ):
        self._check_is_fitted()
        horizon = _validate_horizon(horizon)
        forecasts = np.column_stack(
            [_forecast_theta_component(component, horizon) for component in self.components_]
        )
        start = self.training_values_.shape[0]
        mean = self._reseasonalize(forecasts, np.arange(start, start + horizon))
        if self.forecast_frame_metadata_ is not None:
            return _table_forecast_result(
                self,
                mean,
                horizon,
                levels=self.prediction_interval_levels,
                future_frame=future_frame,
            )
        return self._format_prediction(mean, return_interval=return_interval, level=level)

    def _deseasonalize(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.seasonality is None:
            return matrix.copy(), None
        pattern = np.zeros((self.season_length, matrix.shape[1]), dtype=float)
        for idx in range(self.season_length):
            seasonal_slice = matrix[idx :: self.season_length]
            if self.seasonality == "additive":
                pattern[idx] = np.mean(seasonal_slice, axis=0)
            else:
                pattern[idx] = np.mean(seasonal_slice, axis=0)
        if self.seasonality == "additive":
            pattern = pattern - np.mean(pattern, axis=0, keepdims=True)
            return matrix - pattern[np.arange(matrix.shape[0]) % self.season_length], pattern

        series_mean = np.mean(matrix, axis=0, keepdims=True)
        pattern = pattern / series_mean
        pattern_mean = np.mean(pattern, axis=0, keepdims=True)
        pattern = np.where(pattern_mean == 0, 1.0, pattern / pattern_mean)
        return matrix / pattern[np.arange(matrix.shape[0]) % self.season_length], pattern

    def _reseasonalize(self, values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if self.seasonality is None:
            return values
        pattern = self.seasonal_pattern_[positions % self.season_length]
        if self.seasonality == "additive":
            return values + pattern
        return values * pattern


class OptimizedThetaForecaster(ThetaForecaster):
    """Theta forecaster with deterministic in-sample grid validation."""

    def __init__(
        self,
        *,
        theta_grid: list[float] | tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
        alpha_grid: list[float] | tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8),
        season_length: int | None = None,
        seasonality: str | None = None,
        prediction_interval_levels: list[float] | tuple[float, ...] = (),
    ) -> None:
        if not theta_grid:
            raise ValueError("theta_grid must not be empty")
        if not alpha_grid:
            raise ValueError("alpha_grid must not be empty")
        self.theta_grid = tuple(float(v) for v in theta_grid)
        self.alpha_grid = tuple(float(v) for v in alpha_grid)
        self.validation_scores_: list[dict[str, float]] = []
        super().__init__(
            theta=self.theta_grid[0],
            alpha=self.alpha_grid[0],
            season_length=season_length,
            seasonality=seasonality,
            prediction_interval_levels=prediction_interval_levels,
        )

    def fit(self, y: Any, *, timestamps: Any | None = None, series_ids: list[Any] | None = None):
        if isinstance(y, ForecastFrame):
            matrix, _, _, _ = _matrix_from_frame(y)
        else:
            matrix, _, _, _ = _as_series_matrix(y, timestamps=timestamps, series_ids=series_ids)
        best_score = float("inf")
        best_theta = self.theta_grid[0]
        best_alpha = self.alpha_grid[0]
        scores: list[dict[str, float]] = []
        for theta in self.theta_grid:
            for alpha in self.alpha_grid:
                candidate = ThetaForecaster(
                    theta=theta,
                    alpha=alpha,
                    season_length=self.season_length,
                    seasonality=self.seasonality,
                )
                candidate.fit(matrix)
                residuals = np.asarray(candidate.residuals_)
                if residuals.ndim == 1:
                    eval_residuals = residuals[1:]
                else:
                    eval_residuals = residuals[1:, :]
                score = float(np.mean(eval_residuals**2))
                scores.append({"theta": theta, "alpha": alpha, "mse": score})
                if (score, theta, alpha) < (best_score, best_theta, best_alpha):
                    best_score = score
                    best_theta = theta
                    best_alpha = alpha
        self.theta = best_theta
        self.alpha = best_alpha
        self.validation_scores_ = scores
        super().fit(y, timestamps=timestamps, series_ids=series_ids)
        self.metadata_["optimized"] = True
        self.metadata_["validation_mse"] = best_score
        self.metadata_["theta_grid"] = list(self.theta_grid)
        self.metadata_["alpha_grid"] = list(self.alpha_grid)
        return self


def _fit_theta_series(y: np.ndarray, theta: float, alpha: float) -> dict[str, Any]:
    n = y.shape[0]
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    trend = intercept + slope * x
    ses = np.empty(n, dtype=float)
    ses[0] = y[0]
    for i in range(1, n):
        ses[i] = alpha * y[i - 1] + (1.0 - alpha) * ses[i - 1]
    fitted = 0.5 * (ses + trend)
    return {
        "last_level": float(alpha * y[-1] + (1.0 - alpha) * ses[-1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "fitted": fitted,
        "theta": float(theta),
    }


def _forecast_theta_component(component: dict[str, Any], horizon: int) -> np.ndarray:
    steps = np.arange(1, horizon + 1, dtype=float)
    drift = (1.0 - 1.0 / component["theta"]) * component["slope"] * steps
    return component["last_level"] + drift
