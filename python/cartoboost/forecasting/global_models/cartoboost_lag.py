from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._native_wrappers import NativeForecastWrapper


@dataclass(frozen=True)
class ForecastResult:
    """Thin result container for native CartoBoost lag forecast outputs."""

    frame: Any
    predictions: np.ndarray
    feature_names: list[str]
    regressor_metadata: dict[str, Any]


class CartoBoostLagForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust CartoBoost lag forecasting binding."""

    native_class_name = "CartoBoostLagForecaster"

    def __init__(self, **params: Any) -> None:
        native_params = {
            key: value
            for key, value in params.items()
            if key
            in {
                "lags",
                "rolling_windows",
                "calendar_features",
                "recursive",
                "prediction_interval_levels",
            }
        }
        super().__init__(**native_params)
        for key, value in params.items():
            setattr(self, key, value)


__all__ = ["CartoBoostLagForecaster", "ForecastResult"]
