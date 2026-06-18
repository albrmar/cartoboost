from __future__ import annotations

from .._native_wrappers import NativeForecastWrapper
from .naive import _prediction_interval_levels


class SeasonalNaiveForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust seasonal naive forecasting binding."""

    native_class_name = "SeasonalNaiveForecaster"

    def __init__(
        self,
        season_length: int,
        *,
        prediction_interval_levels: list[float] | tuple[float, ...] = (),
    ) -> None:
        season_length = int(season_length)
        if season_length <= 0:
            raise ValueError("season_length must be a positive integer")
        super().__init__(
            season_length=season_length,
            prediction_interval_levels=_prediction_interval_levels(prediction_interval_levels),
        )
        self.season_length = season_length


__all__ = ["SeasonalNaiveForecaster"]
