import importlib.util

import pytest
from cartoboost.forecasting.registry import ForecastModelSpec, ForecastRegistry


def test_default_registry_contains_forecasting_v1_models():
    registry = ForecastRegistry.defaults()

    assert registry.names() == (
        "naive",
        "seasonal_naive",
        "theta",
        "optimized_theta",
        "ets",
        "auto_arima",
        "kalman",
        "kriging",
        "cartoboost_lag",
    )


def test_registry_prevents_duplicate_names_unless_override():
    registry = ForecastRegistry()
    registry.register(ForecastModelSpec("taxi_demand", factory=lambda: "first"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(ForecastModelSpec("taxi_demand", factory=lambda: "second"))

    registry.register(ForecastModelSpec("taxi_demand", factory=lambda: "second"), override=True)

    assert registry.create("taxi_demand") == "second"


def test_model_spec_reports_missing_optional_dependencies(monkeypatch):
    def missing(name):
        if name == "missing_backend":
            return None
        return importlib.util.find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", missing)
    spec = ForecastModelSpec(
        "taxi_theta",
        factory=lambda: object(),
        optional_dependencies=("missing_backend",),
    )

    with pytest.raises(ImportError, match="missing_backend"):
        spec.create()


def test_default_registry_models_are_native_wrappers(install_fake_native):
    native = install_fake_native("NaiveForecaster")
    registry = ForecastRegistry.defaults()

    naive = registry.create("naive")

    assert naive.fit([1.0, 2.0, 3.0]).predict(1) == {"args": (1,), "kwargs": {}}
    assert native.calls[1][1].rows[-1] == ("__single__", "1970-01-03T00:00:00", 3.0)
