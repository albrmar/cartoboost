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
        max_p: int = 3,
        max_d: int = 1,
        max_q: int = 2,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"unknown AutoARIMAForecaster parameters: {unknown}")
        if error_policy != "raise":
            raise ValueError("AutoARIMAForecaster supports error_policy='raise' only")
        if int(m) <= 0:
            raise ValueError("m must be a positive integer")
        super().__init__(
            seasonal=bool(seasonal),
            m=int(m),
            error_policy=error_policy,
            max_p=int(max_p),
            max_d=int(max_d),
            max_q=int(max_q),
        )
        self.seasonal = bool(seasonal)
        self.m = int(m)
        self.error_policy = error_policy
        self.max_p = int(max_p)
        self.max_d = int(max_d)
        self.max_q = int(max_q)


__all__ = ["AutoARIMAForecaster"]
