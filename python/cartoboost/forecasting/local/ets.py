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
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float | None = None,
    ) -> None:
        if trend not in {None, "add", "additive"}:
            raise ValueError("Rust ETS currently supports trend=None or additive trend only")
        if seasonal not in {None, "add", "additive"}:
            raise ValueError(
                "Rust ETS currently supports seasonal=None or additive seasonality only"
            )
        if damped_trend:
            raise ValueError("Rust ETS currently does not support damped_trend")
        if seasonal is not None and (seasonal_periods is None or int(seasonal_periods) <= 1):
            raise ValueError("seasonal_periods must be greater than 1 when seasonal is set")
        if seasonal is None and gamma is not None:
            raise ValueError("gamma requires additive seasonality")
        super().__init__(
            alpha=float(alpha),
            beta=float(beta),
            gamma=None if gamma is None else float(gamma),
            season_length=None if seasonal_periods is None else int(seasonal_periods),
        )
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = None if seasonal_periods is None else int(seasonal_periods)
        self.damped_trend = bool(damped_trend)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = None if gamma is None else float(gamma)


__all__ = ["ETSForecaster"]
