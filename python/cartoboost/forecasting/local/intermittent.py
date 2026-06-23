from __future__ import annotations

import math

from .._native_wrappers import NativeForecastWrapper


class CrostonForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust Croston intermittent-demand forecaster."""

    native_class_name = "CrostonForecaster"

    def __init__(self, *, alpha: float = 0.2) -> None:
        alpha = float(alpha)
        _validate_unit_interval("alpha", alpha)
        super().__init__(alpha=alpha)
        self.alpha = alpha


class SbaForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust SBA intermittent-demand forecaster."""

    native_class_name = "SbaForecaster"

    def __init__(self, *, alpha: float = 0.2) -> None:
        alpha = float(alpha)
        _validate_unit_interval("alpha", alpha)
        super().__init__(alpha=alpha)
        self.alpha = alpha


class TsbForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust TSB intermittent-demand forecaster."""

    native_class_name = "TsbForecaster"

    def __init__(self, *, alpha: float = 0.2, beta: float = 0.2) -> None:
        alpha = float(alpha)
        beta = float(beta)
        _validate_unit_interval("alpha", alpha)
        _validate_unit_interval("beta", beta)
        super().__init__(alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in (0, 1]")


__all__ = ["CrostonForecaster", "SbaForecaster", "TsbForecaster"]
