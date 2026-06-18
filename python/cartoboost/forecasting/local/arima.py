from __future__ import annotations

from typing import Any

import numpy as np

from .naive import (
    ForecastResult,
    NaiveForecaster,
    _as_series_matrix,
    _restore_shape,
    _validate_horizon,
)


class AutoARIMAForecaster(NaiveForecaster):
    """Auto-ARIMA forecaster backed by pmdarima with an explicit fallback policy."""

    def __init__(
        self,
        *,
        seasonal: bool = False,
        m: int = 1,
        error_policy: str = "raise",
        **kwargs: Any,
    ):
        if error_policy not in {"raise", "fallback"}:
            raise ValueError("error_policy must be 'raise' or 'fallback'")
        if int(m) <= 0:
            raise ValueError("m must be a positive integer")
        self.seasonal = bool(seasonal)
        self.m = int(m)
        self.error_policy = error_policy
        self.auto_arima_kwargs = dict(kwargs)
        self.is_fitted_ = False

    def fit(
        self,
        y: Any,
        *,
        timestamps: Any | None = None,
        series_ids: list[Any] | None = None,
    ) -> AutoARIMAForecaster:
        matrix, fitted_timestamps, fitted_series_ids, single_series = _as_series_matrix(
            y, timestamps=timestamps, series_ids=series_ids
        )
        try:
            from pmdarima import auto_arima
        except ImportError as exc:
            if self.error_policy == "raise":
                raise ImportError(
                    "AutoARIMAForecaster requires pmdarima. Install it with `pip install pmdarima`."
                ) from exc
            return self._fit_fallback(
                matrix, fitted_timestamps, fitted_series_ids, single_series, str(exc)
            )

        models = []
        fitted_columns = []
        residual_columns = []
        try:
            for col in range(matrix.shape[1]):
                model = auto_arima(
                    matrix[:, col],
                    seasonal=self.seasonal,
                    m=self.m,
                    suppress_warnings=True,
                    error_action="raise",
                    **self.auto_arima_kwargs,
                )
                fitted = np.asarray(model.predict_in_sample(), dtype=float)
                models.append(model)
                fitted_columns.append(fitted)
                residual_columns.append(matrix[:, col] - fitted)
        except Exception as exc:
            if self.error_policy == "raise":
                raise
            return self._fit_fallback(
                matrix, fitted_timestamps, fitted_series_ids, single_series, str(exc)
            )

        self.training_values_ = matrix
        self.models_ = models
        self.fallback_model_ = None
        self.fitted_values_ = _restore_shape(np.column_stack(fitted_columns), single_series)
        self.residuals_ = _restore_shape(np.column_stack(residual_columns), single_series)
        self.timestamps_ = fitted_timestamps
        self.series_ids_ = fitted_series_ids
        self.single_series_ = single_series
        self.metadata_ = {
            "model": self.__class__.__name__,
            "backend": "pmdarima",
            "n_obs": int(matrix.shape[0]),
            "n_series": int(matrix.shape[1]),
            "series_ids": list(fitted_series_ids),
            "seasonal": self.seasonal,
            "m": self.m,
            "error_policy": self.error_policy,
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
        if self.fallback_model_ is not None:
            result = self.fallback_model_.predict(
                horizon, return_interval=return_interval, level=level
            )
            if return_interval:
                result.metadata.update(self.metadata_)
            return result

        mean = np.column_stack(
            [np.asarray(model.predict(n_periods=horizon), dtype=float) for model in self.models_]
        )
        return self._format_prediction(mean, return_interval=return_interval, level=level)

    def _fit_fallback(
        self,
        matrix: np.ndarray,
        timestamps: np.ndarray | None,
        series_ids: list[Any],
        single_series: bool,
        reason: str,
    ) -> AutoARIMAForecaster:
        fallback = NaiveForecaster().fit(matrix, timestamps=timestamps, series_ids=series_ids)
        fallback.single_series_ = single_series
        self.training_values_ = matrix
        self.models_ = []
        self.fallback_model_ = fallback
        self.fitted_values_ = _restore_shape(fallback.fitted_values_, single_series)
        self.residuals_ = _restore_shape(fallback.residuals_, single_series)
        self.timestamps_ = timestamps
        self.series_ids_ = series_ids
        self.single_series_ = single_series
        self.metadata_ = {
            "model": self.__class__.__name__,
            "backend": "naive_fallback",
            "fallback_reason": reason,
            "n_obs": int(matrix.shape[0]),
            "n_series": int(matrix.shape[1]),
            "series_ids": list(series_ids),
            "seasonal": self.seasonal,
            "m": self.m,
            "error_policy": self.error_policy,
            "has_timestamps": timestamps is not None,
        }
        self.is_fitted_ = True
        return self
