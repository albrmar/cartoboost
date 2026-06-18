from __future__ import annotations

from typing import Any

from .._native_wrappers import NativeForecastWrapper


class AutoARIMAForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust auto-ARIMA forecasting binding."""

    native_class_name = "AutoARIMAForecaster"

    def __init__(
        self,
        *,
        seasonal: bool = False,
        m: int = 1,
        error_policy: str = "raise",
        **kwargs: Any,
    ) -> None:
        if error_policy != "raise":
            raise NotImplementedError(
                "AutoARIMAForecaster fallback policies are not available in Python; "
                "Rust binding support is required."
            )
        if int(m) <= 0:
            raise ValueError("m must be a positive integer")
        super().__init__(
            seasonal=bool(seasonal),
            m=int(m),
            error_policy=error_policy,
            **kwargs,
        )
        self.seasonal = bool(seasonal)
        self.m = int(m)
        self.error_policy = error_policy


__all__ = ["AutoARIMAForecaster"]
