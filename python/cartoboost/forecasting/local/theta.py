from __future__ import annotations

from .._native_wrappers import NativeForecastWrapper
from .naive import _prediction_interval_levels


class ThetaForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust theta forecasting binding."""

    native_class_name = "ThetaForecaster"

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
        if seasonality is not None and season_length is None:
            raise ValueError("season_length is required when seasonality is set")
        if season_length is not None and int(season_length) <= 1:
            raise ValueError("season_length must be greater than 1 when provided")
        super().__init__(
            theta=theta,
            alpha=alpha,
            season_length=None if season_length is None else int(season_length),
            seasonality=seasonality,
            prediction_interval_levels=_prediction_interval_levels(prediction_interval_levels),
        )
        self.theta = theta
        self.alpha = alpha
        self.season_length = None if season_length is None else int(season_length)
        self.seasonality = seasonality


class OptimizedThetaForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust optimized theta forecasting binding."""

    native_class_name = "OptimizedThetaForecaster"

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
        theta_grid = tuple(float(value) for value in theta_grid)
        alpha_grid = tuple(float(value) for value in alpha_grid)
        if any(theta <= 0 for theta in theta_grid):
            raise ValueError("theta_grid values must be positive")
        if any(not 0 < alpha <= 1 for alpha in alpha_grid):
            raise ValueError("alpha_grid values must be in (0, 1]")
        if seasonality not in {None, "additive", "multiplicative"}:
            raise ValueError("seasonality must be None, 'additive', or 'multiplicative'")
        if seasonality is not None and season_length is None:
            raise ValueError("season_length is required when seasonality is set")
        if season_length is not None and int(season_length) <= 1:
            raise ValueError("season_length must be greater than 1 when provided")
        super().__init__(
            theta_grid=theta_grid,
            alpha_grid=alpha_grid,
            season_length=None if season_length is None else int(season_length),
            seasonality=seasonality,
            prediction_interval_levels=_prediction_interval_levels(prediction_interval_levels),
        )
        self.theta_grid = theta_grid
        self.alpha_grid = alpha_grid
        self.season_length = None if season_length is None else int(season_length)
        self.seasonality = seasonality


__all__ = ["OptimizedThetaForecaster", "ThetaForecaster"]
