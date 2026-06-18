from __future__ import annotations

from .._native_wrappers import ForecastResult, NativeForecastWrapper


class NaiveForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust naive forecasting binding."""

    native_class_name = "NaiveForecaster"

    def __init__(
        self,
        *,
        prediction_interval_levels: list[float] | tuple[float, ...] = (),
    ) -> None:
        super().__init__(
            prediction_interval_levels=tuple(float(level) for level in prediction_interval_levels)
        )


__all__ = ["ForecastResult", "NaiveForecaster"]
