from __future__ import annotations

from typing import Any

from .frequency import validate_horizon
from .schema import ForecastFrame


class BaseForecaster:
    """Base guardrails for forecasting estimators."""

    is_fitted_: bool = False

    @staticmethod
    def validate_horizon(horizon: int) -> int:
        return validate_horizon(horizon)

    def _mark_fitted(self) -> None:
        self.is_fitted_ = True

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise ValueError("forecaster is not fitted")

    def fit(self, frame: ForecastFrame, *_args: Any, **_kwargs: Any) -> BaseForecaster:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("fit requires a ForecastFrame")
        self._mark_fitted()
        return self

    def predict(self, horizon: int, *_args: Any, **_kwargs: Any) -> Any:
        self._check_is_fitted()
        return self.validate_horizon(horizon)


class SingleSeriesForecasterMixin:
    """Mixin for estimators that only accept one time series."""

    def _validate_single_series_frame(self, frame: ForecastFrame) -> ForecastFrame:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("expected a ForecastFrame")
        if frame.is_panel:
            raise ValueError("single-series forecasters require data without series_id_col")
        return frame


class PanelForecasterMixin:
    """Mixin for estimators that require isolated panel series."""

    def _validate_panel_frame(self, frame: ForecastFrame) -> ForecastFrame:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("expected a ForecastFrame")
        if not frame.is_panel:
            raise ValueError("panel forecasters require series_id_col")
        if not frame.series_ids:
            raise ValueError("panel forecasters require at least one series")
        return frame
