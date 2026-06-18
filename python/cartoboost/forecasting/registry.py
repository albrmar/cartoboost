"""Forecasting model registry and lightweight default model specs."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

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


def default_model_specs() -> tuple[ForecastModelSpec, ...]:
    from .global_models import CartoBoostLagForecaster
    from .local import (
        AutoARIMAForecaster,
        ETSForecaster,
        OptimizedThetaForecaster,
        ThetaForecaster,
    )

    return (
        ForecastModelSpec("naive", factory=_NaiveForecaster),
        ForecastModelSpec("seasonal_naive", factory=_SeasonalNaiveForecaster),
        ForecastModelSpec("theta", factory=ThetaForecaster),
        ForecastModelSpec("optimized_theta", factory=OptimizedThetaForecaster),
        ForecastModelSpec("ets", factory=ETSForecaster),
        ForecastModelSpec("auto_arima", factory=AutoARIMAForecaster),
        ForecastModelSpec("cartoboost_lag", factory=CartoBoostLagForecaster),
        ForecastModelSpec(
            "weighted_ensemble",
            factory=_weighted_ensemble_factory,
            metadata={"kind": "ensemble"},
        ),
    )


class _NaiveForecaster:
    def __init__(self, *, drift: bool = False) -> None:
        self.drift = drift
        self._series: list[float] | dict[Any, list[float]] | None = None

    def fit(self, y: Any, **_: Any) -> _NaiveForecaster:
        self._series = _copy_series(y)
        return self

    def predict(self, horizon: int, **_: Any) -> list[float] | dict[Any, list[float]]:
        _validate_horizon(horizon)
        if self._series is None:
            raise ValueError("Naive forecaster must be fit before predict")
        if isinstance(self._series, dict):
            return {key: self._predict_one(values, horizon) for key, values in self._series.items()}
        return self._predict_one(self._series, horizon)

    def _predict_one(self, values: list[float], horizon: int) -> list[float]:
        if not values:
            raise ValueError("Naive forecaster requires at least one observation")
        last = values[-1]
        if not self.drift or len(values) < 2:
            return [last] * horizon
        step = (values[-1] - values[0]) / (len(values) - 1)
        return [last + step * (idx + 1) for idx in range(horizon)]


class _SeasonalNaiveForecaster:
    def __init__(self, *, season_length: int = 1) -> None:
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = int(season_length)
        self._series: list[float] | dict[Any, list[float]] | None = None

    def fit(self, y: Any, **_: Any) -> _SeasonalNaiveForecaster:
        self._series = _copy_series(y)
        return self

    def predict(self, horizon: int, **_: Any) -> list[float] | dict[Any, list[float]]:
        _validate_horizon(horizon)
        if self._series is None:
            raise ValueError("Seasonal naive forecaster must be fit before predict")
        if isinstance(self._series, dict):
            return {key: self._predict_one(values, horizon) for key, values in self._series.items()}
        return self._predict_one(self._series, horizon)

    def _predict_one(self, values: list[float], horizon: int) -> list[float]:
        if not values:
            raise ValueError("Seasonal naive forecaster requires at least one observation")
        season = values[-self.season_length :]
        return [season[idx % len(season)] for idx in range(horizon)]


def _weighted_ensemble_factory(**params: Any) -> Any:
    from .ensemble import WeightedEnsembleForecaster

    return WeightedEnsembleForecaster(**params)


def _missing_optional_factory(name: str) -> Factory:
    def factory(**_: Any) -> Any:
        raise ImportError(
            f"Forecast model '{name}' needs a concrete backend adapter. "
            "Register a ForecastModelSpec with a custom factory to use it."
        )

    return factory


def _copy_series(y: Any) -> list[float] | dict[Any, list[float]]:
    if isinstance(y, Mapping):
        copied = {key: _float_list(values) for key, values in y.items()}
        if not copied:
            raise ValueError("panel series must contain at least one panel")
        return copied
    return _float_list(y)


def _float_list(values: Any) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("series must contain at least one observation")
    return result


def _validate_horizon(horizon: int) -> None:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
