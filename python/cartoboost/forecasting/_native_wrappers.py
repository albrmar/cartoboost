from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForecastResult:
    """Thin result container for native forecasting outputs."""

    mean: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.mean, dtype=dtype)


class NativeForecastWrapper:
    """Base class for Python forecasting wrappers over Rust/PyO3 implementations."""

    native_class_name: str

    def __init__(self, **params: Any) -> None:
        self._params = dict(params)
        self._native_model: Any | None = None
        self.is_fitted_ = False

    def fit(self, *args: Any, **kwargs: Any) -> NativeForecastWrapper:
        native_model = self._new_native_model()
        fit = getattr(native_model, "fit", None)
        if fit is None:
            raise NotImplementedError(
                f"Rust binding {self.native_class_name!r} does not expose fit()."
            )
        native_args = self._coerce_fit_args(args)
        result = fit(*native_args, **kwargs)
        self._native_model = native_model if result is None else result
        self.is_fitted_ = True
        return self

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        self._check_is_fitted()
        predict = getattr(self._native_model, "predict", None)
        if predict is None:
            raise NotImplementedError(
                f"Rust binding {self.native_class_name!r} does not expose predict()."
            )
        return predict(*args, **kwargs)

    def forecast(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)

    def predict_interval(self, *args: Any, **kwargs: Any) -> Any:
        self._check_is_fitted()
        method = getattr(self._native_model, "predict_interval", None)
        if method is None:
            raise NotImplementedError(
                f"Rust binding {self.native_class_name!r} does not expose predict_interval()."
            )
        return method(*args, **kwargs)

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def get_metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        method = getattr(self._native_model, "get_metadata", None)
        if method is not None:
            return dict(method())
        metadata = getattr(self._native_model, "metadata_", None)
        return {} if metadata is None else dict(metadata)

    @property
    def metadata_(self) -> dict[str, Any]:
        return self.get_metadata()

    def _new_native_model(self) -> Any:
        native_class = _native_class(self.native_class_name)
        if native_class is None:
            raise NotImplementedError(
                f"Rust binding for {self.__class__.__name__} is not available: "
                f"cartoboost._native.{self.native_class_name} is missing."
            )
        return native_class(**self._params)

    def _coerce_fit_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            return args
        first = args[0]
        native_frame = getattr(first, "_native_frame", None)
        if native_frame is not None:
            return (native_frame, *args[1:])
        if _is_native_forecast_frame(first):
            return args
        return (_native_frame_from_values(first), *args[1:])

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self._native_model is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before predict")

    def __getattr__(self, name: str) -> Any:
        native_model = self.__dict__.get("_native_model")
        if native_model is not None and hasattr(native_model, name):
            return getattr(native_model, name)
        raise AttributeError(name)


def _native_class(name: str) -> Any | None:
    try:
        from cartoboost import _native
    except ImportError:
        return None
    return getattr(_native, name, None)


def _is_native_forecast_frame(value: Any) -> bool:
    return value.__class__.__name__ == "ForecastFrame" and value.__class__.__module__.endswith(
        "._native"
    )


def _native_frame_from_values(values: Any) -> Any:
    native_frame_class = _native_class("ForecastFrame")
    if native_frame_class is None:
        raise NotImplementedError("Rust binding for ForecastFrame is not available.")
    if isinstance(values, dict):
        rows = []
        lengths = {len(series_values) for series_values in values.values()}
        if len(lengths) != 1:
            raise ValueError("all panel series must have the same length")
        for series_id, series_values in values.items():
            for idx, value in enumerate(series_values):
                rows.append(
                    (
                        str(series_id),
                        (datetime(1970, 1, 1) + timedelta(days=idx)).strftime("%Y-%m-%dT%H:%M:%S"),
                        float(value),
                    )
                )
        return native_frame_class(rows, "D")
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        rows = [
            (
                "__single__",
                (datetime(1970, 1, 1) + timedelta(days=idx)).strftime("%Y-%m-%dT%H:%M:%S"),
                float(value),
            )
            for idx, value in enumerate(arr)
        ]
    elif arr.ndim == 2:
        rows = []
        for idx in range(arr.shape[0]):
            timestamp = (datetime(1970, 1, 1) + timedelta(days=idx)).strftime("%Y-%m-%dT%H:%M:%S")
            for series_idx in range(arr.shape[1]):
                rows.append((str(series_idx), timestamp, float(arr[idx, series_idx])))
    else:
        raise ValueError("forecast training values must be a 1D series or 2D panel")
    return native_frame_class(rows, "D")
