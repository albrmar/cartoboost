from __future__ import annotations

from typing import Any

from .._native_wrappers import NativeForecastWrapper


class ArimaForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust ARIMA(p,d,q) forecasting binding."""

    native_class_name = "ArimaForecaster"

    def __init__(
        self,
        *,
        p: int = 1,
        d: int = 0,
        q: int = 0,
    ) -> None:
        p = int(p)
        d = int(d)
        q = int(q)
        if p < 0 or d < 0 or q < 0:
            raise ValueError("ARIMA order parameters p, d, and q must be nonnegative")
        if p > 8:
            raise ValueError("p must be <= 8")
        if d > 2:
            raise ValueError("d must be <= 2")
        if q > 8:
            raise ValueError("q must be <= 8")
        super().__init__(p=p, d=d, q=q)
        self.p = p
        self.d = d
        self.q = q


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
        if seasonal:
            raise ValueError("AutoARIMAForecaster Rust binding currently supports seasonal=False")
        if int(m) <= 0:
            raise ValueError("m must be a positive integer")
        max_p = int(max_p)
        max_d = int(max_d)
        max_q = int(max_q)
        if max_p < 0 or max_d < 0 or max_q < 0:
            raise ValueError("max_p, max_d, and max_q must be nonnegative")
        if max_p > 8:
            raise ValueError("max_p must be <= 8")
        if max_d > 2:
            raise ValueError("max_d must be <= 2")
        if max_q > 8:
            raise ValueError("max_q must be <= 8")
        super().__init__(
            seasonal=bool(seasonal),
            m=int(m),
            error_policy=error_policy,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
        )
        self.seasonal = bool(seasonal)
        self.m = int(m)
        self.error_policy = error_policy
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q


__all__ = ["ArimaForecaster", "AutoARIMAForecaster"]
