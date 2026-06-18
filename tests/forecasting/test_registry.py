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
        "cartoboost_lag",
        "weighted_ensemble",
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


def test_default_naive_forecaster_is_native_wrapper():
    registry = ForecastRegistry.defaults()

    naive = registry.create("naive")
    seasonal = registry.create("seasonal_naive", season_length=2)

    assert [row[4] for row in naive.fit([1.0, 2.0, 3.0]).predict(1).predictions()] == [3.0]
    assert [
        row[4]
        for row in seasonal.fit({"pickup_1": [10, 11, 12], "pickup_2": [20, 21, 22]})
        .predict(1)
        .predictions()
    ] == [11.0, 21.0]
