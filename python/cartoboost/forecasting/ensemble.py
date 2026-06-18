"""Weighted ensemble forecasters for single-series and panel forecasts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import sqrt
from statistics import NormalDist
from typing import Any

from .schema import ForecastFrame
from .schema import ForecastResult as TableForecastResult

Forecast = list[float] | dict[Any, list[float]]
ForecastResult = Forecast | dict[str, Any]


@dataclass
class WeightedEnsembleForecaster:
    """Combine forecasters with fixed, normalized weights."""

    models: Mapping[str, Any]
    weights: Mapping[str, float] | None = None
    interval_level: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("WeightedEnsembleForecaster requires at least one model")
        self.models = dict(self.models)
        self.weights_ = _normalize_weights(self.models, self.weights)
        self.metadata_ = {
            "ensemble": "weighted",
            "weights": dict(self.weights_),
            **dict(self.metadata),
        }
        self.residual_scale_: float | dict[Any, float] | None = None

    def fit(self, y: Any, **fit_params: Any) -> WeightedEnsembleForecaster:
        self.frame_mode_ = isinstance(y, ForecastFrame)
        for model in self.models.values():
            fit = getattr(model, "fit", None)
            if fit is not None:
                fit(y, **fit_params)
        if self.frame_mode_:
            self.residual_scale_ = None
            return self
        self.residual_scale_ = self._estimate_residual_scale(y)
        return self

    def predict(
        self,
        horizon: int,
        *,
        return_intervals: bool | None = None,
        **predict_params: Any,
    ) -> ForecastResult:
        _validate_horizon(horizon)
        if getattr(self, "frame_mode_", False):
            return self._predict_frame_mode(horizon, **predict_params)
        member_forecasts = {
            name: _model_forecast(model, horizon, **predict_params)
            for name, model in self.models.items()
        }
        means = _weighted_average(member_forecasts, self.weights_, field_name="mean")
        means = _apply_bounds(means, self.lower_bound, self.upper_bound)
        include_intervals = (
            self.interval_level is not None if return_intervals is None else return_intervals
        )
        if not include_intervals:
            return means
        lower, upper = self._interval_bounds(means, member_forecasts)
        return {
            "mean": means,
            "lower": _apply_bounds(lower, self.lower_bound, self.upper_bound),
            "upper": _apply_bounds(upper, self.lower_bound, self.upper_bound),
            "metadata": dict(self.metadata_),
        }

    def _predict_frame_mode(self, horizon: int, **predict_params: Any) -> TableForecastResult:
        member_tables = {}
        for name, model in self.models.items():
            forecast = model.predict(horizon, **predict_params)
            if not hasattr(forecast, "to_pandas"):
                raise TypeError("frame-mode ensembles require component ForecastResult outputs")
            table = forecast.to_pandas()
            member_tables[name] = table
        first = next(iter(member_tables.values()))
        key_cols = ["series_id", "timestamp", "horizon"]
        output = first[key_cols].copy()
        output["model"] = self.__class__.__name__
        output["mean"] = 0.0
        interval_cols = sorted(
            {
                col
                for table in member_tables.values()
                for col in table.columns
                if col.startswith("lower_") or col.startswith("upper_")
            }
        )
        for col in interval_cols:
            output[col] = 0.0
        for name, table in member_tables.items():
            aligned = output[key_cols].merge(
                table,
                on=key_cols,
                how="left",
                validate="one_to_one",
            )
            if aligned["mean"].isna().any():
                raise ValueError("component forecast rows are not aligned")
            weight = self.weights_[name]
            output["mean"] += weight * aligned["mean"].astype(float)
            for col in interval_cols:
                if col in aligned:
                    output[col] += weight * aligned[col].astype(float)
                else:
                    output[col] += weight * aligned["mean"].astype(float)
        output = output.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
        return TableForecastResult(
            data=output,
            timestamp_col="timestamp",
            prediction_col="mean",
            series_id_col="series_id",
        )

    def forecast(self, horizon: int, **predict_params: Any) -> ForecastResult:
        return self.predict(horizon, **predict_params)

    def _estimate_residual_scale(self, y: Any) -> float | dict[Any, float] | None:
        series = _coerce_series(y)
        try:
            fitted = self.predict(_series_length(series), return_intervals=False)
        except Exception:
            return None
        if isinstance(series, dict):
            if not isinstance(fitted, dict):
                return None
            return {
                key: _rmse(_align_tail(values, fitted[key]))
                for key, values in series.items()
                if key in fitted
            }
        if isinstance(fitted, dict):
            return None
        return _rmse(_align_tail(series, fitted))

    def _interval_bounds(
        self,
        means: Forecast,
        member_forecasts: Mapping[str, dict[str, Forecast]],
    ) -> tuple[Forecast, Forecast]:
        explicit_lower = _weighted_average_or_none(member_forecasts, self.weights_, "lower")
        explicit_upper = _weighted_average_or_none(member_forecasts, self.weights_, "upper")
        if explicit_lower is not None and explicit_upper is not None:
            return explicit_lower, explicit_upper
        level = self.interval_level if self.interval_level is not None else 0.8
        if not 0.0 < level < 1.0:
            raise ValueError("interval_level must be between 0 and 1")
        z_value = NormalDist().inv_cdf(0.5 + level / 2.0)
        spread = _ensemble_spread(member_forecasts, self.weights_, means)
        scale = _combine_scale(spread, self.residual_scale_, means)
        return _offset_forecast(means, scale, -z_value), _offset_forecast(means, scale, z_value)


@dataclass
class BacktestWeightedEnsembleForecaster(WeightedEnsembleForecaster):
    """Learn ensemble weights from rolling-origin inverse-error backtests."""

    backtest_horizon: int = 1
    min_train_size: int | None = None
    step_size: int = 1
    error_floor: float = 1e-9

    def fit(self, y: Any, **fit_params: Any) -> BacktestWeightedEnsembleForecaster:
        series = _coerce_series(y)
        _validate_horizon(self.backtest_horizon)
        if self.step_size < 1:
            raise ValueError("step_size must be >= 1")
        min_train_size = self.min_train_size or max(2, self.backtest_horizon)
        errors = {name: [] for name in self.models}
        for train, actual in _rolling_splits(
            series, min_train_size, self.backtest_horizon, self.step_size
        ):
            for name, model in self.models.items():
                candidate = _clone_model(model)
                fit = getattr(candidate, "fit", None)
                if fit is not None:
                    fit(train, **fit_params)
                forecast = _model_forecast(candidate, self.backtest_horizon)
                errors[name].append(_mae_between(actual, forecast["mean"]))
        mean_errors = {
            name: sum(values) / len(values) if values else float("inf")
            for name, values in errors.items()
        }
        if not any(value < float("inf") for value in mean_errors.values()):
            raise ValueError("not enough observations to run ensemble backtest weighting")
        inverse = {
            name: 1.0 / max(error, self.error_floor)
            for name, error in mean_errors.items()
            if error < float("inf")
        }
        self.weights_ = _normalize_weight_values(inverse)
        self.metadata_ = {
            **dict(self.metadata_),
            "ensemble": "backtest_weighted",
            "weights": dict(self.weights_),
            "backtest_errors": mean_errors,
            "backtest_horizon": self.backtest_horizon,
            "min_train_size": min_train_size,
            "step_size": self.step_size,
        }
        for model in self.models.values():
            fit = getattr(model, "fit", None)
            if fit is not None:
                fit(series, **fit_params)
        self.residual_scale_ = self._estimate_residual_scale(series)
        return self


def _model_forecast(model: Any, horizon: int, **params: Any) -> dict[str, Forecast]:
    method = getattr(model, "predict", None) or getattr(model, "forecast", None)
    if method is None:
        raise TypeError(f"model {model!r} does not provide predict(horizon) or forecast(horizon)")
    raw = method(horizon, **params)
    if isinstance(raw, Mapping) and "mean" in raw:
        result = {"mean": _coerce_forecast(raw["mean"])}
        if "lower" in raw:
            result["lower"] = _coerce_forecast(raw["lower"])
        if "upper" in raw:
            result["upper"] = _coerce_forecast(raw["upper"])
        return result
    return {"mean": _coerce_forecast(raw)}


def _weighted_average(
    member_forecasts: Mapping[str, dict[str, Forecast]],
    weights: Mapping[str, float],
    *,
    field_name: str,
) -> Forecast:
    values = {name: forecast[field_name] for name, forecast in member_forecasts.items()}
    panel_values = [value for value in values.values() if isinstance(value, dict)]
    if panel_values:
        panel_keys = set(panel_values[0])
        for value in values.values():
            if isinstance(value, dict) and set(value) != panel_keys:
                raise ValueError("all panel forecasts must contain the same panel keys")
        return {
            key: _weighted_vector(
                {
                    name: forecast[key] if isinstance(forecast, dict) else forecast
                    for name, forecast in values.items()
                },
                weights,
            )
            for key in panel_values[0]
        }
    return _weighted_vector(values, weights)


def _weighted_average_or_none(
    member_forecasts: Mapping[str, dict[str, Forecast]],
    weights: Mapping[str, float],
    field_name: str,
) -> Forecast | None:
    if not all(field_name in forecast for forecast in member_forecasts.values()):
        return None
    return _weighted_average(member_forecasts, weights, field_name=field_name)


def _weighted_vector(
    values: Mapping[str, list[float]], weights: Mapping[str, float]
) -> list[float]:
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1:
        raise ValueError("all member forecasts must have the same horizon")
    horizon = lengths.pop()
    return [sum(weights[name] * values[name][idx] for name in values) for idx in range(horizon)]


def _normalize_weights(
    models: Mapping[str, Any],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        return {name: 1.0 / len(models) for name in models}
    if set(weights) != set(models):
        raise ValueError("weights must contain exactly the same names as models")
    return _normalize_weight_values(weights)


def _normalize_weight_values(weights: Mapping[str, float]) -> dict[str, float]:
    cleaned = {name: float(weight) for name, weight in weights.items()}
    if any(weight < 0.0 for weight in cleaned.values()):
        raise ValueError("weights must be non-negative")
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    return {name: weight / total for name, weight in cleaned.items()}


def _coerce_forecast(values: Any) -> Forecast:
    if isinstance(values, Mapping):
        result = {key: [float(item) for item in forecast] for key, forecast in values.items()}
        if not result:
            raise ValueError("panel forecast must contain at least one panel")
        return result
    return [float(item) for item in values]


def _coerce_series(values: Any) -> Forecast:
    forecast = _coerce_forecast(values)
    if isinstance(forecast, list) and not forecast:
        raise ValueError("series must contain at least one observation")
    return forecast


def _apply_bounds(values: Forecast, lower: float | None, upper: float | None) -> Forecast:
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower_bound must be <= upper_bound")

    def clip(value: float) -> float:
        if lower is not None:
            value = max(float(lower), value)
        if upper is not None:
            value = min(float(upper), value)
        return value

    if isinstance(values, dict):
        return {key: [clip(value) for value in forecast] for key, forecast in values.items()}
    return [clip(value) for value in values]


def _ensemble_spread(
    member_forecasts: Mapping[str, dict[str, Forecast]],
    weights: Mapping[str, float],
    means: Forecast,
) -> Forecast:
    members = {name: forecast["mean"] for name, forecast in member_forecasts.items()}
    if isinstance(means, dict):
        return {
            key: [
                sqrt(
                    sum(
                        weights[name]
                        * (
                            (
                                members[name][key][idx]
                                if isinstance(members[name], dict)
                                else members[name][idx]
                            )
                            - means[key][idx]
                        )
                        ** 2
                        for name in members
                    )
                )
                for idx in range(len(means[key]))
            ]
            for key in means
        }
    return [
        sqrt(sum(weights[name] * (members[name][idx] - means[idx]) ** 2 for name in members))
        for idx in range(len(means))
    ]


def _combine_scale(
    spread: Forecast, residual_scale: float | dict[Any, float] | None, means: Forecast
) -> Forecast:
    if isinstance(means, dict):
        residuals = residual_scale if isinstance(residual_scale, dict) else {}
        return {
            key: [
                sqrt(value * value + float(residuals.get(key, 0.0)) ** 2) for value in spread[key]
            ]
            for key in means
        }
    residual = float(residual_scale or 0.0)
    return [sqrt(value * value + residual * residual) for value in spread]


def _offset_forecast(values: Forecast, scale: Forecast, multiplier: float) -> Forecast:
    if isinstance(values, dict):
        return {
            key: [value + multiplier * scale[key][idx] for idx, value in enumerate(forecast)]
            for key, forecast in values.items()
        }
    return [value + multiplier * scale[idx] for idx, value in enumerate(values)]


def _align_tail(actual: list[float], fitted: list[float]) -> list[float]:
    size = min(len(actual), len(fitted))
    return [actual[-size + idx] - fitted[-size + idx] for idx in range(size)] if size else []


def _rmse(residuals: list[float]) -> float:
    if not residuals:
        return 0.0
    return sqrt(sum(value * value for value in residuals) / len(residuals))


def _series_length(series: Forecast) -> int:
    if isinstance(series, dict):
        return min(len(values) for values in series.values())
    return len(series)


def _rolling_splits(
    series: Forecast,
    min_train_size: int,
    horizon: int,
    step_size: int,
) -> list[tuple[Forecast, Forecast]]:
    size = _series_length(series)
    splits: list[tuple[Forecast, Forecast]] = []
    for end in range(min_train_size, size - horizon + 1, step_size):
        if isinstance(series, dict):
            splits.append(
                (
                    {key: values[:end] for key, values in series.items()},
                    {key: values[end : end + horizon] for key, values in series.items()},
                )
            )
        else:
            splits.append((series[:end], series[end : end + horizon]))
    return splits


def _mae_between(actual: Forecast, predicted: Forecast) -> float:
    if isinstance(actual, dict):
        if not isinstance(predicted, dict) or set(actual) != set(predicted):
            raise ValueError("actual and predicted panel keys must match")
        errors = [
            abs(observed - predicted[key][idx])
            for key, values in actual.items()
            for idx, observed in enumerate(values)
        ]
    else:
        if isinstance(predicted, dict):
            raise ValueError("cannot compare single-series actuals to panel forecasts")
        errors = [abs(observed - predicted[idx]) for idx, observed in enumerate(actual)]
    return sum(errors) / len(errors)


def _clone_model(model: Any) -> Any:
    try:
        return deepcopy(model)
    except Exception:
        return model


def _validate_horizon(horizon: int) -> None:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
