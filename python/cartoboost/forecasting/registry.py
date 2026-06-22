"""Forecasting model registry and lightweight default model specs."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._native_wrappers import NativeForecastWrapper

Factory = Callable[..., Any]


@dataclass(frozen=True)
class ForecastModelSpec:
    """Description and constructor for a forecasting model family."""

    name: str
    factory: Factory | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    optional_dependencies: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ForecastModelSpec name must be non-empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "optional_dependencies", tuple(self.optional_dependencies))

    def validate_dependencies(self) -> None:
        """Raise a clear install hint when an optional backend is unavailable."""

        missing = [
            dep for dep in self.optional_dependencies if importlib.util.find_spec(dep) is None
        ]
        if missing:
            deps = ", ".join(missing)
            raise ImportError(
                f"Forecast model '{self.name}' requires optional package(s): {deps}. "
                "Install the matching CartoBoost forecasting extra or register a custom factory."
            )

    def create(self, **overrides: Any) -> Any:
        """Validate dependencies and construct the configured model."""

        self.validate_dependencies()
        if self.factory is None:
            raise ValueError(f"Forecast model '{self.name}' has no registered factory")
        params = {**self.params, **overrides}
        return self.factory(**params)


class ForecastRegistry:
    """Duplicate-safe registry for forecast model specifications."""

    def __init__(self, specs: Iterable[ForecastModelSpec] | None = None) -> None:
        self._specs: dict[str, ForecastModelSpec] = {}
        for spec in specs or ():
            self.register(spec)

    def register(self, spec: ForecastModelSpec, *, override: bool = False) -> ForecastModelSpec:
        if spec.name in self._specs and not override:
            raise ValueError(
                f"Forecast model '{spec.name}' is already registered; "
                "pass override=True to replace it"
            )
        self._specs[spec.name] = spec
        return spec

    def unregister(self, name: str) -> ForecastModelSpec:
        try:
            return self._specs.pop(name)
        except KeyError as exc:
            raise KeyError(f"Forecast model '{name}' is not registered") from exc

    def get(self, name: str) -> ForecastModelSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            known = ", ".join(self.names())
            raise KeyError(
                f"Forecast model '{name}' is not registered. Known models: {known}"
            ) from exc

    def create(self, name: str, **overrides: Any) -> Any:
        return self.get(name).create(**overrides)

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[ForecastModelSpec, ...]:
        return tuple(self._specs.values())

    def as_dict(self) -> dict[str, ForecastModelSpec]:
        return dict(self._specs)

    @classmethod
    def defaults(cls) -> ForecastRegistry:
        registry = cls()
        for spec in default_model_specs():
            registry.register(spec)
        return registry


class RegisteredNativeForecaster(NativeForecastWrapper):
    """Configurable native forecasting wrapper used by registry-only model specs."""

    def __init__(self, *, native_class_name: str, **params: Any) -> None:
        self.native_class_name = native_class_name
        super().__init__(**params)


def default_model_specs() -> tuple[ForecastModelSpec, ...]:
    from .ensemble import BottomUpReconciler, MinTraceReconciler
    from .global_models import CartoBoostLagForecaster
    from .local import (
        AutoARIMAForecaster,
        AutoKalmanForecaster,
        AutoLocalLevelKalmanForecaster,
        ETSForecaster,
        KalmanForecaster,
        LocalLevelKalmanForecaster,
        NaiveForecaster,
        OptimizedThetaForecaster,
        PiecewiseLinearSeasonalForecaster,
        SeasonalNaiveForecaster,
        ThetaForecaster,
    )

    return (
        ForecastModelSpec("naive", factory=NaiveForecaster),
        ForecastModelSpec("seasonal_naive", factory=SeasonalNaiveForecaster),
        ForecastModelSpec("theta", factory=ThetaForecaster),
        ForecastModelSpec("optimized_theta", factory=OptimizedThetaForecaster),
        ForecastModelSpec("piecewise_linear_seasonal", factory=PiecewiseLinearSeasonalForecaster),
        ForecastModelSpec("ets", factory=ETSForecaster),
        ForecastModelSpec("auto_arima", factory=AutoARIMAForecaster),
        ForecastModelSpec("kalman", factory=KalmanForecaster),
        ForecastModelSpec("local_level_kalman", factory=LocalLevelKalmanForecaster),
        ForecastModelSpec("auto_kalman", factory=AutoKalmanForecaster),
        ForecastModelSpec("auto_local_level_kalman", factory=AutoLocalLevelKalmanForecaster),
        ForecastModelSpec("cartoboost_lag", factory=CartoBoostLagForecaster),
        ForecastModelSpec(
            "local_linear_trend_kalman",
            factory=_native_factory("LocalLinearTrendKalmanForecaster"),
        ),
        ForecastModelSpec(
            "unobserved_components",
            factory=_native_factory("UnobservedComponentsForecaster"),
        ),
        ForecastModelSpec("sarimax", factory=_native_factory("SarimaxForecaster")),
        ForecastModelSpec(
            "dynamic_regression",
            factory=_native_factory("DynamicRegressionForecaster"),
        ),
        ForecastModelSpec("croston", factory=_native_factory("CrostonForecaster")),
        ForecastModelSpec("sba", factory=_native_factory("SbaForecaster")),
        ForecastModelSpec("tsb", factory=_native_factory("TsbForecaster")),
        ForecastModelSpec("mstl_ets", factory=_native_factory("MstlEtsForecaster")),
        ForecastModelSpec("stl_arima", factory=_native_factory("StlArimaForecaster")),
        ForecastModelSpec(
            "quantile_carto_boost_lag",
            factory=_native_factory("QuantileCartoBoostLagForecaster"),
        ),
        ForecastModelSpec(
            "conformal_forecaster",
            factory=_native_factory("ConformalForecaster"),
        ),
        ForecastModelSpec("bottom_up_reconciler", factory=BottomUpReconciler),
        ForecastModelSpec("min_trace_reconciler", factory=MinTraceReconciler),
        ForecastModelSpec(
            "foundation_model_adapter_optional",
            factory=_native_factory("FoundationModelAdapterForecaster"),
            metadata={"optional": True},
        ),
    )


def _native_factory(native_class_name: str) -> Factory:
    def factory(**params: Any) -> RegisteredNativeForecaster:
        return RegisteredNativeForecaster(native_class_name=native_class_name, **params)

    return factory
