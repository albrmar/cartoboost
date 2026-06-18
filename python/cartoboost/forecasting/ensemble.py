from __future__ import annotations

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
        super().__init__(
            models=dict(models),
            weights=None if weights is None else dict(weights),
            interval_level=interval_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata={} if metadata is None else dict(metadata),
        )
        self.models = dict(models)
        self.weights = None if weights is None else dict(weights)


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


__all__ = ["BacktestWeightedEnsembleForecaster", "WeightedEnsembleForecaster"]
