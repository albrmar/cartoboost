from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ._native_wrappers import NativeForecastWrapper


class WeightedEnsembleForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust weighted ensemble forecasting binding."""

    native_class_name = "WeightedEnsembleForecaster"

    def __init__(
        self,
        models: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
        interval_level: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not models:
            raise ValueError("WeightedEnsembleForecaster requires at least one model")
        if interval_level is not None or lower_bound is not None or upper_bound is not None:
            raise NotImplementedError(
                "WeightedEnsembleForecaster prediction intervals are not supported yet"
            )
        weights = _normalize_weights(models, weights)
        super().__init__(
            models=dict(models),
            weights=weights,
            interval_level=interval_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata={} if metadata is None else dict(metadata),
        )
        self.models = dict(models)
        self.weights = weights

    def _new_native_model(self) -> Any:
        try:
            from cartoboost import _native
        except ImportError as exc:
            raise NotImplementedError(
                "Rust binding for WeightedEnsembleForecaster is not available."
            ) from exc
        native_ensemble_class = getattr(_native, "WeightedEnsembleForecaster", None)
        if native_ensemble_class is None:
            raise NotImplementedError(
                "Rust binding for WeightedEnsembleForecaster is not available."
            )

        native_members = []
        for name, model in self.models.items():
            native_model = _native_model_from_wrapper(model)
            native_members.append((name, native_model, self.weights[name]))
        return native_ensemble_class(native_members)

    def get_metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        metadata_json = getattr(self._native_model, "metadata_json", None)
        if metadata_json is None:
            return {}
        return dict(json.loads(metadata_json()))


class BacktestWeightedEnsembleForecaster(WeightedEnsembleForecaster):
    """Thin wrapper for the Rust backtest-weighted ensemble forecasting binding."""

    native_class_name = "BacktestWeightedEnsembleForecaster"

    def __init__(
        self,
        models: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
        interval_level: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        backtest_horizon: int = 1,
        min_train_size: int | None = None,
        step_size: int = 1,
        error_floor: float = 1e-9,
    ) -> None:
        if backtest_horizon < 1:
            raise ValueError("backtest_horizon must be a positive integer")
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        super().__init__(
            models=models,
            weights=weights,
            interval_level=interval_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata=metadata,
        )
        self._params.update(
            {
                "backtest_horizon": int(backtest_horizon),
                "min_train_size": min_train_size,
                "step_size": int(step_size),
                "error_floor": float(error_floor),
            }
        )
        self.backtest_horizon = int(backtest_horizon)
        self.min_train_size = min_train_size
        self.step_size = int(step_size)
        self.error_floor = float(error_floor)

    def _new_native_model(self) -> Any:
        raise NotImplementedError(
            "Rust binding for BacktestWeightedEnsembleForecaster is not available."
        )


__all__ = ["BacktestWeightedEnsembleForecaster", "WeightedEnsembleForecaster"]


def _normalize_weights(
    models: Mapping[str, Any],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        return {name: 1.0 for name in models}
    missing = set(models) - set(weights)
    extra = set(weights) - set(models)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing weights for {sorted(missing)}")
        if extra:
            details.append(f"unknown weights for {sorted(extra)}")
        raise ValueError("weights must match models exactly: " + "; ".join(details))
    return {name: float(weight) for name, weight in weights.items()}


def _native_model_from_wrapper(model: Any) -> Any:
    if model.__class__.__module__.endswith("._native"):
        return model
    new_native_model = getattr(model, "_new_native_model", None)
    if new_native_model is None:
        raise TypeError(
            "WeightedEnsembleForecaster only supports native CartoBoost forecasting wrappers"
        )
    return new_native_model()
