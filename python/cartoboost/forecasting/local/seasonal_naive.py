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


class SeasonalNaiveForecaster(NaiveForecaster):
    """Repeats the last observed seasonal cycle."""

    def __init__(
        self,
        season_length: int,
        *,
        prediction_interval_levels: list[float] | tuple[float, ...] = (),
    ) -> None:
        season_length = int(season_length)
        if season_length <= 0:
            raise ValueError("season_length must be a positive integer")
        super().__init__(prediction_interval_levels=prediction_interval_levels)
        self.season_length = season_length

    def fit(
        self,
        y: Any,
        *,
        timestamps: Any | None = None,
        series_ids: list[Any] | None = None,
    ) -> SeasonalNaiveForecaster:
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
        if matrix.shape[0] < self.season_length:
            raise ValueError("y must contain at least season_length observations")
        fitted = matrix.copy()
        fitted[: self.season_length] = matrix[: self.season_length]
        fitted[self.season_length :] = matrix[: -self.season_length]
        residuals = matrix - fitted
        self.training_values_ = matrix
        self.seasonal_values_ = matrix[-self.season_length :].copy()
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
            "season_length": self.season_length,
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
        indices = np.arange(horizon) % self.season_length
        mean = self.seasonal_values_[indices]
        if self.forecast_frame_metadata_ is not None:
            return _table_forecast_result(
                self,
                mean,
                horizon,
                levels=self.prediction_interval_levels,
                future_frame=future_frame,
            )
        return self._format_prediction(mean, return_interval=return_interval, level=level)
