from __future__ import annotations

from typing import Any

from ._native_wrappers import NativeForecastWrapper


class NBeatsForecaster(NativeForecastWrapper):
    """Thin Python wrapper for the Rust N-BEATS style forecasting expert."""

    native_class_name = "NBeatsForecaster"

    def __init__(
        self,
        *,
        input_size: int = 8,
        hidden_size: int = 16,
        epochs: int = 80,
        learning_rate: float = 0.01,
        **metadata: Any,
    ) -> None:
        _validate_common(input_size, hidden_size, epochs, learning_rate)
        super().__init__(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            metadata=dict(metadata),
        )

    def _new_native_model(self) -> Any:
        try:
            from cartoboost import _native
        except ImportError as exc:
            raise NotImplementedError(
                "Rust binding for NBeatsForecaster is not available."
            ) from exc
        native_class = getattr(_native, self.native_class_name, None)
        if native_class is None:
            raise NotImplementedError("Rust binding for NBeatsForecaster is not available.")
        return native_class(
            input_size=self._params["input_size"],
            hidden_size=self._params["hidden_size"],
            epochs=self._params["epochs"],
            learning_rate=self._params["learning_rate"],
        )


class NHiTSForecaster(NativeForecastWrapper):
    """Thin Python wrapper for the Rust N-HiTS style forecasting expert."""

    native_class_name = "NHiTSForecaster"

    def __init__(
        self,
        *,
        input_size: int = 12,
        hidden_size: int = 16,
        epochs: int = 80,
        learning_rate: float = 0.01,
        pooling_size: int = 2,
        **metadata: Any,
    ) -> None:
        _validate_common(input_size, hidden_size, epochs, learning_rate)
        if pooling_size < 1 or pooling_size > input_size:
            raise ValueError("pooling_size must be between 1 and input_size")
        super().__init__(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            pooling_size=int(pooling_size),
            metadata=dict(metadata),
        )

    def _new_native_model(self) -> Any:
        try:
            from cartoboost import _native
        except ImportError as exc:
            raise NotImplementedError("Rust binding for NHiTSForecaster is not available.") from exc
        native_class = getattr(_native, self.native_class_name, None)
        if native_class is None:
            raise NotImplementedError("Rust binding for NHiTSForecaster is not available.")
        return native_class(
            input_size=self._params["input_size"],
            hidden_size=self._params["hidden_size"],
            epochs=self._params["epochs"],
            learning_rate=self._params["learning_rate"],
            pooling_size=self._params["pooling_size"],
        )


NHITSForecaster = NHiTSForecaster
NBEATSForecaster = NBeatsForecaster

__all__ = ["NBeatsForecaster", "NBEATSForecaster", "NHiTSForecaster", "NHITSForecaster"]


def _validate_common(
    input_size: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
) -> None:
    if input_size < 1:
        raise ValueError("input_size must be a positive integer")
    if hidden_size < 1:
        raise ValueError("hidden_size must be a positive integer")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
