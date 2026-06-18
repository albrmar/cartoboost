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


__all__ = ["KalmanForecaster"]
