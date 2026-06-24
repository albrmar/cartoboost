from __future__ import annotations

from typing import Any

from .._native_wrappers import NativeForecastWrapper


class KalmanForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust local linear Kalman forecasting binding."""

    native_class_name = "KalmanForecaster"

    def __init__(
        self,
        level_process_variance: float = 0.05,
        trend_process_variance: float = 0.005,
        observation_variance: float = 1.0,
        **params: Any,
    ) -> None:
        native_params = {
            "level_process_variance": float(level_process_variance),
            "trend_process_variance": float(trend_process_variance),
            "observation_variance": float(observation_variance),
        }
        super().__init__(**native_params)
        self.level_process_variance = float(level_process_variance)
        self.trend_process_variance = float(trend_process_variance)
        self.observation_variance = float(observation_variance)
        for key, value in params.items():
            setattr(self, key, value)


class AutoKalmanForecaster(NativeForecastWrapper):
    """Self-tuning Rust local-linear Kalman forecaster.

    Candidate variance triples are scored on each series tail, and the selected
    shared parameters are refit on the full training frame.
    """

    native_class_name = "AutoKalmanForecaster"

    def __init__(
        self,
        level_process_variance_grid: list[float] | tuple[float, ...] | None = None,
        trend_process_variance_grid: list[float] | tuple[float, ...] | None = None,
        observation_variance_grid: list[float] | tuple[float, ...] | None = None,
        validation_window: int | None = None,
        **params: Any,
    ) -> None:
        native_params: dict[str, Any] = {
            "level_process_variance_grid": _optional_float_list(level_process_variance_grid),
            "trend_process_variance_grid": _optional_float_list(trend_process_variance_grid),
            "observation_variance_grid": _optional_float_list(observation_variance_grid),
            "validation_window": validation_window,
        }
        super().__init__(**native_params)
        self.level_process_variance_grid = native_params["level_process_variance_grid"]
        self.trend_process_variance_grid = native_params["trend_process_variance_grid"]
        self.observation_variance_grid = native_params["observation_variance_grid"]
        self.validation_window = validation_window
        for key, value in params.items():
            setattr(self, key, value)


class LocalLevelKalmanForecaster(NativeForecastWrapper):
    """Rust local-level Kalman forecaster for noisy level-only series."""

    native_class_name = "LocalLevelKalmanForecaster"

    def __init__(
        self,
        level_process_variance: float = 0.05,
        observation_variance: float = 1.0,
        **params: Any,
    ) -> None:
        native_params = {
            "level_process_variance": float(level_process_variance),
            "observation_variance": float(observation_variance),
        }
        super().__init__(**native_params)
        self.level_process_variance = float(level_process_variance)
        self.observation_variance = float(observation_variance)
        for key, value in params.items():
            setattr(self, key, value)


class AutoLocalLevelKalmanForecaster(NativeForecastWrapper):
    """Self-tuning Rust local-level Kalman forecaster."""

    native_class_name = "AutoLocalLevelKalmanForecaster"

    def __init__(
        self,
        level_process_variance_grid: list[float] | tuple[float, ...] | None = None,
        observation_variance_grid: list[float] | tuple[float, ...] | None = None,
        validation_window: int | None = None,
        **params: Any,
    ) -> None:
        native_params: dict[str, Any] = {
            "level_process_variance_grid": _optional_float_list(level_process_variance_grid),
            "observation_variance_grid": _optional_float_list(observation_variance_grid),
            "validation_window": validation_window,
        }
        super().__init__(**native_params)
        self.level_process_variance_grid = native_params["level_process_variance_grid"]
        self.observation_variance_grid = native_params["observation_variance_grid"]
        self.validation_window = validation_window
        for key, value in params.items():
            setattr(self, key, value)


def _optional_float_list(values: list[float] | tuple[float, ...] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in values]


__all__ = [
    "AutoKalmanForecaster",
    "AutoLocalLevelKalmanForecaster",
    "KalmanForecaster",
    "LocalLevelKalmanForecaster",
]
