from __future__ import annotations

from .._native_wrappers import ForecastResult, NativeForecastWrapper


def _prediction_interval_levels(
    levels: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    converted = tuple(float(level) for level in levels)
    if any(not 0.0 < level < 1.0 for level in converted):
        raise ValueError("prediction_interval_levels must be between 0 and 1")
    return converted


class NaiveForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust naive forecasting binding."""

    native_class_name = "NaiveForecaster"

    def __init__(
        self,
        *,
        prediction_interval_levels: list[float] | tuple[float, ...] = (),
    ) -> None:
        super().__init__(
            prediction_interval_levels=_prediction_interval_levels(prediction_interval_levels)
        )


__all__ = ["ForecastResult", "NaiveForecaster"]
