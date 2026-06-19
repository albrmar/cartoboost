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
        alpha = float(alpha)
        beta = float(beta)
        gamma = None if gamma is None else float(gamma)
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if gamma is not None and not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            season_length=None if seasonal_periods is None else int(seasonal_periods),
        )
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = None if seasonal_periods is None else int(seasonal_periods)
        self.damped_trend = bool(damped_trend)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fitted_values(self, series_id: str = "__single__") -> list[float]:
        self._check_is_fitted()
        return list(self._native_model.fitted_values(series_id))

    def residuals(self, series_id: str = "__single__") -> list[float]:
        self._check_is_fitted()
        return list(self._native_model.residuals(series_id))

    def levels(self, series_id: str = "__single__") -> list[float]:
        self._check_is_fitted()
        return list(self._native_model.level_values(series_id))

    def trends(self, series_id: str = "__single__") -> list[float]:
        self._check_is_fitted()
        return list(self._native_model.trend_values(series_id))

    def seasonal_components(self, series_id: str = "__single__") -> list[float]:
        self._check_is_fitted()
        return list(self._native_model.seasonal_values(series_id))


__all__ = ["ETSForecaster"]
