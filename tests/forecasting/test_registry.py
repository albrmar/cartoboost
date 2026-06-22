import importlib.util

import pytest
from cartoboost.forecasting.local import PiecewiseLinearSeasonalForecaster
from cartoboost.forecasting.registry import ForecastModelSpec, ForecastRegistry


def test_default_registry_contains_forecasting_v1_models():
    registry = ForecastRegistry.defaults()

    assert registry.names() == (
        "naive",
        "seasonal_naive",
        "theta",
        "optimized_theta",
        "piecewise_linear_seasonal",
        "ets",
        "auto_arima",
        "kalman",
        "local_level_kalman",
        "auto_kalman",
        "auto_local_level_kalman",
        "cartoboost_lag",
        "local_linear_trend_kalman",
        "unobserved_components",
        "sarimax",
        "dynamic_regression",
        "croston",
        "sba",
        "tsb",
        "mstl_ets",
        "stl_arima",
        "quantile_carto_boost_lag",
        "conformal_forecaster",
        "bottom_up_reconciler",
        "min_trace_reconciler",
        "foundation_model_adapter_optional",
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


def test_registry_placeholder_models_delegate_to_named_native_binding(install_fake_native):
    native = install_fake_native("LocalLinearTrendKalmanForecaster")
    registry = ForecastRegistry.defaults()

    model = registry.create("local_linear_trend_kalman", process_variance=0.1)

    assert model.native_class_name == "LocalLinearTrendKalmanForecaster"
    assert model.fit([1.0, 2.0]).predict(1) == {"args": (1,), "kwargs": {}}
    assert native.calls[0] == ("init", {"process_variance": 0.1})


def test_piecewise_linear_seasonal_wrapper_normalizes_native_overrides(install_fake_native):
    native = install_fake_native("PiecewiseLinearSeasonalForecaster")
    model = PiecewiseLinearSeasonalForecaster(
        growth="flat",
        component_mode="multiplicative",
        changepoints=2,
        changepoint_range=0.9,
        changepoint_timestamps=("2026-01-08T00:00:00",),
        custom_seasonalities=[
            {
                "name": "biweekly_pickup_cycle",
                "periodDays": 14,
                "fourierOrder": 2,
                "mode": "additive",
                "conditionName": "rush_hour",
                "l2Regularization": 0.05,
            }
        ],
        events=[("airport_surge", "2026-02-01T00:00:00", -1, 1)],
        event_mode="additive",
        extra_regressors=("airport_queue",),
        regressor_modes={"airport_queue": "additive"},
        extra_regressor_monotonic_constraints={"airport_queue": 1},
        future_regressors={"airport_queue": (1, 0)},
        future_regressors_by_series={"PULocationID=132": {"airport_queue": (0, 1)}},
        trend_adjustments={"2": 1.1},
        trend_adjustments_by_series={"PULocationID=132": {"1": 1.2}},
        prediction_interval_levels=(0.8,),
        quantile_levels=(0.25, 0.75),
        uncertainty_samples=16,
        trend_uncertainty_policy="normal",
        fit_loss="huber",
        huber_delta=1.25,
        irls_iterations=4,
    ).fit({"PULocationID=132": [10.0, 12.0, 14.0]})

    model.predict(
        2,
        future_regressors={"airport_queue": [0, 1]},
        future_regressors_by_series={"PULocationID=132": {"airport_queue": [1, 0]}},
        prediction_interval_levels=[0.9],
        uncertainty_samples=8,
        trend_adjustments={1: 1.05},
        trend_adjustments_by_series={"PULocationID=132": {2: 1.15}},
    )
    model.components(
        2,
        future_regressors={"airport_queue": [1, 0]},
        trend_adjustments={2: 1.1},
    )
    model.samples(
        2,
        future_regressors_by_series={"PULocationID=132": {"airport_queue": [1, 1]}},
        uncertainty_samples=4,
    )
    model.quantiles(2, [0.1, 0.9], future_regressors={"airport_queue": [0, 0]})
    restored = PiecewiseLinearSeasonalForecaster.from_json(model.to_json())

    assert restored.is_fitted_
    assert native.calls[0] == (
        "init",
        {
            "growth": "flat",
            "component_mode": "multiplicative",
            "changepoints": 2,
            "changepoint_range": 0.9,
            "changepoint_timestamps": ["2026-01-08T00:00:00"],
            "yearly_fourier_order": 0,
            "weekly_fourier_order": 3,
            "daily_fourier_order": 0,
            "auto_yearly_seasonality": True,
            "auto_weekly_seasonality": True,
            "auto_daily_seasonality": True,
            "custom_seasonalities": [
                ("biweekly_pickup_cycle", 14.0, 2, "additive", "rush_hour", 0.05)
            ],
            "changepoint_l2_regularization": 0.05,
            "changepoint_l1_regularization": 0.0,
            "seasonality_l2_regularization": 0.01,
            "yearly_l2_regularization": None,
            "weekly_l2_regularization": None,
            "daily_l2_regularization": None,
            "event_l2_regularization": 0.01,
            "regressor_l2_regularization": 0.01,
            "event_l2_regularization_by_name": {},
            "regressor_l2_regularization_by_name": {},
            "events": [("airport_surge", "2026-02-01T00:00:00", -1, 1)],
            "event_mode": "additive",
            "extra_regressors": ["airport_queue"],
            "regressor_modes": {"airport_queue": "additive"},
            "extra_regressor_monotonic_constraints": {"airport_queue": 1},
            "regressor_standardization": "auto",
            "future_regressors": {"airport_queue": [1.0, 0.0]},
            "future_regressors_by_series": {"PULocationID=132": {"airport_queue": [0.0, 1.0]}},
            "trend_adjustments": {2: 1.1},
            "trend_adjustments_by_series": {"PULocationID=132": {1: 1.2}},
            "residual_shock_window": 0,
            "residual_shock_scale": 0.0,
            "residual_shock_decay": 1.0,
            "prediction_interval_levels": (0.8,),
            "quantile_levels": (0.25, 0.75),
            "uncertainty_samples": 16,
            "trend_uncertainty_policy": "normal",
            "trend_uncertainty_scale": 1.0,
            "coefficient_uncertainty_scale": 1.0,
            "uncertainty_seed": 14172834030107287843,
            "cap": None,
            "floor": 0.0,
            "cap_regressor": None,
            "floor_regressor": None,
            "fit_loss": "huber",
            "huber_delta": 1.25,
            "irls_iterations": 4,
        },
    )
    assert native.calls[1][1].rows[-1] == ("PULocationID=132", "1970-01-03T00:00:00", 14.0)
    assert native.calls[2] == (
        "predict",
        (
            2,
            {"airport_queue": [0.0, 1.0]},
            {"PULocationID=132": {"airport_queue": [1.0, 0.0]}},
            (0.9,),
            8,
            {1: 1.05},
            {"PULocationID=132": {2: 1.15}},
        ),
        {},
    )
    assert native.calls[3] == (
        "components_json",
        (2, {"airport_queue": [1.0, 0.0]}, None, {2: 1.1}, None),
        {},
    )
    assert native.calls[4] == (
        "samples_json",
        (2, None, {"PULocationID=132": {"airport_queue": [1.0, 1.0]}}, 4, None, None),
        {},
    )
    assert native.calls[5] == (
        "quantiles_json",
        (2, (0.1, 0.9), {"airport_queue": [0.0, 0.0]}, None, None, None, None),
        {},
    )
    assert native.calls[-3][0] == "to_json"
    assert native.calls[-2][0] == "from_json"
    assert native.calls[-1][0] == "init"
