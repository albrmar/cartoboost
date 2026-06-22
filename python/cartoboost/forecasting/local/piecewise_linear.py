from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .._native_wrappers import NativeForecastWrapper, _native_class
from .naive import _prediction_interval_levels

EventSpec = Mapping[str, Any] | tuple[str, str] | tuple[str, str, int | None, int | None]
SeasonalitySpec = (
    Mapping[str, Any]
    | tuple[str, float, int]
    | tuple[str, float, int, str | None]
    | tuple[str, float, int, str | None, str | None]
    | tuple[str, float, int, str | None, str | None, float | None]
)


class PiecewiseLinearSeasonalForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust piecewise-linear seasonal forecasting binding."""

    native_class_name = "PiecewiseLinearSeasonalForecaster"

    def __init__(
        self,
        *,
        growth: str = "linear",
        component_mode: str = "additive",
        changepoints: int = 12,
        changepoint_range: float = 0.8,
        changepoint_timestamps: Sequence[str] = (),
        yearly_fourier_order: int = 0,
        weekly_fourier_order: int = 3,
        daily_fourier_order: int = 0,
        auto_yearly_seasonality: bool = True,
        auto_weekly_seasonality: bool = True,
        auto_daily_seasonality: bool = True,
        custom_seasonalities: Sequence[SeasonalitySpec] = (),
        changepoint_l2_regularization: float = 0.05,
        changepoint_l1_regularization: float = 0.0,
        seasonality_l2_regularization: float = 0.01,
        yearly_l2_regularization: float | None = None,
        weekly_l2_regularization: float | None = None,
        daily_l2_regularization: float | None = None,
        event_l2_regularization: float = 0.01,
        regressor_l2_regularization: float = 0.01,
        event_l2_regularization_by_name: Mapping[str, float] | None = None,
        regressor_l2_regularization_by_name: Mapping[str, float] | None = None,
        events: Sequence[EventSpec] = (),
        event_mode: str | None = None,
        extra_regressors: Sequence[str] = (),
        regressor_modes: Mapping[str, str] | None = None,
        extra_regressor_monotonic_constraints: Mapping[str, int] | None = None,
        regressor_standardization: str = "auto",
        future_regressors: Mapping[str, Sequence[float]] | None = None,
        future_regressors_by_series: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        trend_adjustments: Mapping[int, float] | None = None,
        trend_adjustments_by_series: Mapping[str, Mapping[int, float]] | None = None,
        residual_shock_window: int = 0,
        residual_shock_scale: float = 0.0,
        residual_shock_decay: float = 1.0,
        prediction_interval_levels: list[float] | tuple[float, ...] = (),
        quantile_levels: list[float] | tuple[float, ...] = (),
        uncertainty_samples: int = 0,
        trend_uncertainty_policy: str = "laplace",
        trend_uncertainty_scale: float = 1.0,
        coefficient_uncertainty_scale: float = 1.0,
        uncertainty_seed: int = 0xC4B0_0575_A11C_E123,
        cap: float | None = None,
        floor: float = 0.0,
        cap_regressor: str | None = None,
        floor_regressor: str | None = None,
        fit_loss: str = "squared",
        huber_delta: float = 1.345,
        irls_iterations: int = 5,
    ) -> None:
        growth = _validate_choice("growth", growth, {"linear", "flat", "logistic"})
        component_mode = _validate_choice(
            "component_mode", component_mode, {"additive", "multiplicative"}
        )
        fit_loss = _validate_choice("fit_loss", fit_loss, {"squared", "huber"})
        regressor_standardization = _validate_choice(
            "regressor_standardization", regressor_standardization, {"auto", "none"}
        )
        if int(changepoints) < 0:
            raise ValueError("changepoints must be nonnegative")
        changepoint_range = float(changepoint_range)
        if not 0.0 < changepoint_range <= 1.0:
            raise ValueError("changepoint_range must be in (0, 1]")
        orders = {
            "yearly_fourier_order": yearly_fourier_order,
            "weekly_fourier_order": weekly_fourier_order,
            "daily_fourier_order": daily_fourier_order,
        }
        for name, value in orders.items():
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if int(uncertainty_samples) < 0:
            raise ValueError("uncertainty_samples must be nonnegative")
        if int(irls_iterations) < 0:
            raise ValueError("irls_iterations must be nonnegative")
        huber_delta = float(huber_delta)
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        trend_uncertainty_policy = _validate_choice(
            "trend_uncertainty_policy", trend_uncertainty_policy, {"laplace", "normal"}
        )
        trend_uncertainty_scale = float(trend_uncertainty_scale)
        if trend_uncertainty_scale < 0.0:
            raise ValueError("trend_uncertainty_scale must be nonnegative")
        coefficient_uncertainty_scale = float(coefficient_uncertainty_scale)
        if coefficient_uncertainty_scale < 0.0:
            raise ValueError("coefficient_uncertainty_scale must be nonnegative")
        residual_shock_window = int(residual_shock_window)
        if residual_shock_window < 0:
            raise ValueError("residual_shock_window must be nonnegative")
        residual_shock_scale = float(residual_shock_scale)
        if residual_shock_scale < 0.0:
            raise ValueError("residual_shock_scale must be nonnegative")
        residual_shock_decay = float(residual_shock_decay)
        if residual_shock_decay < 0.0 or residual_shock_decay > 1.0:
            raise ValueError("residual_shock_decay must be in [0, 1]")
        super().__init__(
            growth=growth,
            component_mode=component_mode,
            changepoints=int(changepoints),
            changepoint_range=changepoint_range,
            changepoint_timestamps=[str(timestamp) for timestamp in changepoint_timestamps],
            yearly_fourier_order=int(yearly_fourier_order),
            weekly_fourier_order=int(weekly_fourier_order),
            daily_fourier_order=int(daily_fourier_order),
            auto_yearly_seasonality=bool(auto_yearly_seasonality),
            auto_weekly_seasonality=bool(auto_weekly_seasonality),
            auto_daily_seasonality=bool(auto_daily_seasonality),
            custom_seasonalities=_seasonality_tuples(custom_seasonalities),
            changepoint_l2_regularization=float(changepoint_l2_regularization),
            changepoint_l1_regularization=float(changepoint_l1_regularization),
            seasonality_l2_regularization=float(seasonality_l2_regularization),
            yearly_l2_regularization=None
            if yearly_l2_regularization is None
            else float(yearly_l2_regularization),
            weekly_l2_regularization=None
            if weekly_l2_regularization is None
            else float(weekly_l2_regularization),
            daily_l2_regularization=None
            if daily_l2_regularization is None
            else float(daily_l2_regularization),
            event_l2_regularization=float(event_l2_regularization),
            regressor_l2_regularization=float(regressor_l2_regularization),
            event_l2_regularization_by_name=_float_mapping(event_l2_regularization_by_name),
            regressor_l2_regularization_by_name=_float_mapping(regressor_l2_regularization_by_name),
            events=_event_tuples(events),
            event_mode=None
            if event_mode is None
            else _validate_choice("event_mode", event_mode, {"additive", "multiplicative"}),
            extra_regressors=[str(name) for name in extra_regressors],
            regressor_modes=_regressor_mode_values(regressor_modes),
            extra_regressor_monotonic_constraints=_monotonic_constraint_values(
                extra_regressor_monotonic_constraints
            ),
            regressor_standardization=regressor_standardization,
            future_regressors=_future_regressor_values(future_regressors),
            future_regressors_by_series=_future_regressor_values_by_series(
                future_regressors_by_series
            ),
            trend_adjustments=_trend_adjustment_values(trend_adjustments),
            trend_adjustments_by_series=_trend_adjustment_values_by_series(
                trend_adjustments_by_series
            ),
            residual_shock_window=residual_shock_window,
            residual_shock_scale=residual_shock_scale,
            residual_shock_decay=residual_shock_decay,
            prediction_interval_levels=_prediction_interval_levels(prediction_interval_levels),
            quantile_levels=_prediction_interval_levels(quantile_levels),
            uncertainty_samples=int(uncertainty_samples),
            trend_uncertainty_policy=trend_uncertainty_policy,
            trend_uncertainty_scale=trend_uncertainty_scale,
            coefficient_uncertainty_scale=coefficient_uncertainty_scale,
            uncertainty_seed=int(uncertainty_seed),
            cap=None if cap is None else float(cap),
            floor=float(floor),
            cap_regressor=None if cap_regressor is None else str(cap_regressor),
            floor_regressor=None if floor_regressor is None else str(floor_regressor),
            fit_loss=fit_loss,
            huber_delta=huber_delta,
            irls_iterations=int(irls_iterations),
        )

    def to_json(self) -> str:
        self._check_is_fitted()
        return self._native_model.to_json()

    def predict(
        self,
        horizon: int,
        *,
        future_regressors: Mapping[str, Sequence[float]] | None = None,
        future_regressors_by_series: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        prediction_interval_levels: list[float] | tuple[float, ...] | None = None,
        uncertainty_samples: int | None = None,
        trend_adjustments: Mapping[int, float] | None = None,
        trend_adjustments_by_series: Mapping[str, Mapping[int, float]] | None = None,
    ) -> Any:
        self._check_is_fitted()
        return self._native_model.predict(
            int(horizon),
            _future_regressor_values(future_regressors) if future_regressors is not None else None,
            _future_regressor_values_by_series(future_regressors_by_series)
            if future_regressors_by_series is not None
            else None,
            _prediction_interval_levels(prediction_interval_levels)
            if prediction_interval_levels is not None
            else None,
            None if uncertainty_samples is None else int(uncertainty_samples),
            _trend_adjustment_values(trend_adjustments) if trend_adjustments is not None else None,
            _trend_adjustment_values_by_series(trend_adjustments_by_series)
            if trend_adjustments_by_series is not None
            else None,
        )

    def components(
        self,
        horizon: int,
        *,
        future_regressors: Mapping[str, Sequence[float]] | None = None,
        future_regressors_by_series: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        trend_adjustments: Mapping[int, float] | None = None,
        trend_adjustments_by_series: Mapping[str, Mapping[int, float]] | None = None,
    ) -> dict[str, Any]:
        self._check_is_fitted()
        return dict(
            json.loads(
                self._native_model.components_json(
                    int(horizon),
                    _future_regressor_values(future_regressors)
                    if future_regressors is not None
                    else None,
                    _future_regressor_values_by_series(future_regressors_by_series)
                    if future_regressors_by_series is not None
                    else None,
                    _trend_adjustment_values(trend_adjustments)
                    if trend_adjustments is not None
                    else None,
                    _trend_adjustment_values_by_series(trend_adjustments_by_series)
                    if trend_adjustments_by_series is not None
                    else None,
                )
            )
        )

    def components_json(
        self,
        horizon: int,
        *,
        future_regressors: Mapping[str, Sequence[float]] | None = None,
        future_regressors_by_series: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        trend_adjustments: Mapping[int, float] | None = None,
        trend_adjustments_by_series: Mapping[str, Mapping[int, float]] | None = None,
    ) -> str:
        self._check_is_fitted()
        return str(
            self._native_model.components_json(
                int(horizon),
                _future_regressor_values(future_regressors)
                if future_regressors is not None
                else None,
                _future_regressor_values_by_series(future_regressors_by_series)
                if future_regressors_by_series is not None
                else None,
                _trend_adjustment_values(trend_adjustments)
                if trend_adjustments is not None
                else None,
                _trend_adjustment_values_by_series(trend_adjustments_by_series)
                if trend_adjustments_by_series is not None
                else None,
            )
        )

    def samples(
        self,
        horizon: int,
        *,
        future_regressors: Mapping[str, Sequence[float]] | None = None,
        future_regressors_by_series: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        uncertainty_samples: int | None = None,
        trend_adjustments: Mapping[int, float] | None = None,
        trend_adjustments_by_series: Mapping[str, Mapping[int, float]] | None = None,
    ) -> dict[str, Any]:
        self._check_is_fitted()
        return dict(
            json.loads(
                self._native_model.samples_json(
                    int(horizon),
                    _future_regressor_values(future_regressors)
                    if future_regressors is not None
                    else None,
                    _future_regressor_values_by_series(future_regressors_by_series)
                    if future_regressors_by_series is not None
                    else None,
                    None if uncertainty_samples is None else int(uncertainty_samples),
                    _trend_adjustment_values(trend_adjustments)
                    if trend_adjustments is not None
                    else None,
                    _trend_adjustment_values_by_series(trend_adjustments_by_series)
                    if trend_adjustments_by_series is not None
                    else None,
                )
            )
        )

    def samples_json(
        self,
        horizon: int,
        *,
        future_regressors: Mapping[str, Sequence[float]] | None = None,
        future_regressors_by_series: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        uncertainty_samples: int | None = None,
        trend_adjustments: Mapping[int, float] | None = None,
        trend_adjustments_by_series: Mapping[str, Mapping[int, float]] | None = None,
    ) -> str:
        self._check_is_fitted()
        return str(
            self._native_model.samples_json(
                int(horizon),
                _future_regressor_values(future_regressors)
                if future_regressors is not None
                else None,
                _future_regressor_values_by_series(future_regressors_by_series)
                if future_regressors_by_series is not None
                else None,
                None if uncertainty_samples is None else int(uncertainty_samples),
                _trend_adjustment_values(trend_adjustments)
                if trend_adjustments is not None
                else None,
                _trend_adjustment_values_by_series(trend_adjustments_by_series)
                if trend_adjustments_by_series is not None
                else None,
            )
        )

    def quantiles(
        self,
        horizon: int,
        quantile_levels: list[float] | tuple[float, ...] | None = None,
        *,
        future_regressors: Mapping[str, Sequence[float]] | None = None,
        future_regressors_by_series: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        uncertainty_samples: int | None = None,
        trend_adjustments: Mapping[int, float] | None = None,
        trend_adjustments_by_series: Mapping[str, Mapping[int, float]] | None = None,
    ) -> dict[str, Any]:
        self._check_is_fitted()
        levels = None if quantile_levels is None else _prediction_interval_levels(quantile_levels)
        return dict(
            json.loads(
                self._native_model.quantiles_json(
                    int(horizon),
                    levels,
                    _future_regressor_values(future_regressors)
                    if future_regressors is not None
                    else None,
                    _future_regressor_values_by_series(future_regressors_by_series)
                    if future_regressors_by_series is not None
                    else None,
                    None if uncertainty_samples is None else int(uncertainty_samples),
                    _trend_adjustment_values(trend_adjustments)
                    if trend_adjustments is not None
                    else None,
                    _trend_adjustment_values_by_series(trend_adjustments_by_series)
                    if trend_adjustments_by_series is not None
                    else None,
                )
            )
        )

    def quantiles_json(
        self,
        horizon: int,
        quantile_levels: list[float] | tuple[float, ...] | None = None,
        *,
        future_regressors: Mapping[str, Sequence[float]] | None = None,
        future_regressors_by_series: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        uncertainty_samples: int | None = None,
        trend_adjustments: Mapping[int, float] | None = None,
        trend_adjustments_by_series: Mapping[str, Mapping[int, float]] | None = None,
    ) -> str:
        self._check_is_fitted()
        levels = None if quantile_levels is None else _prediction_interval_levels(quantile_levels)
        return str(
            self._native_model.quantiles_json(
                int(horizon),
                levels,
                _future_regressor_values(future_regressors)
                if future_regressors is not None
                else None,
                _future_regressor_values_by_series(future_regressors_by_series)
                if future_regressors_by_series is not None
                else None,
                None if uncertainty_samples is None else int(uncertainty_samples),
                _trend_adjustment_values(trend_adjustments)
                if trend_adjustments is not None
                else None,
                _trend_adjustment_values_by_series(trend_adjustments_by_series)
                if trend_adjustments_by_series is not None
                else None,
            )
        )

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_json() + "\n")
        return path

    @classmethod
    def from_json(cls, value: str) -> PiecewiseLinearSeasonalForecaster:
        native_class = _native_class(cls.native_class_name)
        if native_class is None:
            raise NotImplementedError(
                "Rust binding for PiecewiseLinearSeasonalForecaster is not available."
            )
        model = cls()
        model._native_model = native_class.from_json(value)
        model.is_fitted_ = True
        return model

    @classmethod
    def load_json(cls, path: str | Path) -> PiecewiseLinearSeasonalForecaster:
        return cls.from_json(Path(path).read_text())


def _validate_choice(name: str, value: str, choices: set[str]) -> str:
    value = str(value).strip().lower().replace("-", "_")
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {expected}")
    return value


def _event_tuples(events: Sequence[EventSpec]) -> list[tuple[str, str, int | None, int | None]]:
    normalized: list[tuple[str, str, int | None, int | None]] = []
    for event in events:
        if isinstance(event, Mapping):
            name = event.get("name")
            timestamp = event.get("timestamp")
            lower_window = event.get("lower_window", event.get("lowerWindow", 0))
            upper_window = event.get("upper_window", event.get("upperWindow", 0))
        else:
            if len(event) == 2:
                name, timestamp = event
                lower_window = 0
                upper_window = 0
            elif len(event) == 4:
                name, timestamp, lower_window, upper_window = event
            else:
                raise ValueError("events must be mappings or 2-/4-item tuples")
        if name is None or timestamp is None:
            raise ValueError("events require name and timestamp")
        normalized.append(
            (
                str(name),
                str(timestamp),
                None if lower_window is None else int(lower_window),
                None if upper_window is None else int(upper_window),
            )
        )
    return normalized


def _seasonality_tuples(
    seasonalities: Sequence[SeasonalitySpec],
) -> list[tuple[str, float, int, str | None, str | None, float | None]]:
    normalized: list[tuple[str, float, int, str | None, str | None, float | None]] = []
    for seasonality in seasonalities:
        if isinstance(seasonality, Mapping):
            name = seasonality.get("name")
            period_days = seasonality.get("period_days", seasonality.get("periodDays"))
            fourier_order = seasonality.get("fourier_order", seasonality.get("fourierOrder"))
            mode = seasonality.get("mode")
            condition_name = seasonality.get("condition_name", seasonality.get("conditionName"))
            l2_regularization = seasonality.get(
                "l2_regularization", seasonality.get("l2Regularization")
            )
        else:
            if len(seasonality) == 3:
                name, period_days, fourier_order = seasonality
                mode = None
                condition_name = None
                l2_regularization = None
            elif len(seasonality) == 4:
                name, period_days, fourier_order, mode = seasonality
                condition_name = None
                l2_regularization = None
            elif len(seasonality) == 5:
                name, period_days, fourier_order, mode, condition_name = seasonality
                l2_regularization = None
            elif len(seasonality) == 6:
                name, period_days, fourier_order, mode, condition_name, l2_regularization = (
                    seasonality
                )
            else:
                raise ValueError("custom_seasonalities must be mappings or 3-/4-/5-/6-item tuples")
        if name is None or period_days is None or fourier_order is None:
            raise ValueError("custom_seasonalities require name, period_days, and fourier_order")
        period_days = float(period_days)
        fourier_order = int(fourier_order)
        if period_days <= 0.0:
            raise ValueError("custom seasonality period_days must be positive")
        if fourier_order <= 0:
            raise ValueError("custom seasonality fourier_order must be positive")
        if l2_regularization is not None and float(l2_regularization) < 0.0:
            raise ValueError("custom seasonality l2_regularization must be nonnegative")
        normalized.append(
            (
                str(name),
                period_days,
                fourier_order,
                None
                if mode is None
                else _validate_choice(
                    "custom seasonality mode", mode, {"additive", "multiplicative"}
                ),
                None if condition_name is None else str(condition_name),
                None if l2_regularization is None else float(l2_regularization),
            )
        )
    return normalized


def _regressor_mode_values(regressor_modes: Mapping[str, str] | None) -> dict[str, str]:
    return {
        str(name): _validate_choice("regressor mode", mode, {"additive", "multiplicative"})
        for name, mode in (regressor_modes or {}).items()
    }


def _monotonic_constraint_values(constraints: Mapping[str, int] | None) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for name, value in (constraints or {}).items():
        direction = int(value)
        if direction not in {-1, 0, 1}:
            raise ValueError("monotonic constraint values must be -1, 0, or 1")
        normalized[str(name)] = direction
    return normalized


def _future_regressor_values(
    future_regressors: Mapping[str, Sequence[float]] | None,
) -> dict[str, list[float]]:
    return {
        str(name): [float(value) for value in values]
        for name, values in (future_regressors or {}).items()
    }


def _future_regressor_values_by_series(
    future_regressors: Mapping[str, Mapping[str, Sequence[float]]] | None,
) -> dict[str, dict[str, list[float]]]:
    return {
        str(series_id): _future_regressor_values(values_by_name)
        for series_id, values_by_name in (future_regressors or {}).items()
    }


def _trend_adjustment_values(values: Mapping[int, float] | None) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for horizon, multiplier in (values or {}).items():
        horizon = int(horizon)
        multiplier = float(multiplier)
        if horizon <= 0:
            raise ValueError("trend_adjustments horizon keys must be positive")
        if multiplier <= 0.0:
            raise ValueError("trend_adjustments multipliers must be positive")
        normalized[horizon] = multiplier
    return normalized


def _trend_adjustment_values_by_series(
    values: Mapping[str, Mapping[int, float]] | None,
) -> dict[str, dict[int, float]]:
    return {
        str(series_id): _trend_adjustment_values(adjustments)
        for series_id, adjustments in (values or {}).items()
    }


def _float_mapping(values: Mapping[str, float] | None) -> dict[str, float]:
    return {str(name): float(value) for name, value in (values or {}).items()}


__all__ = ["PiecewiseLinearSeasonalForecaster"]
