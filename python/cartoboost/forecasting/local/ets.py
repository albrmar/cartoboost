from __future__ import annotations

from .._native_wrappers import NativeForecastWrapper


class ETSForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust ETS forecasting binding."""

    native_class_name = "ETSForecaster"

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
        super().__init__(
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=None if seasonal_periods is None else int(seasonal_periods),
            damped_trend=bool(damped_trend),
        )
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = None if seasonal_periods is None else int(seasonal_periods)
        self.damped_trend = bool(damped_trend)


__all__ = ["ETSForecaster"]
