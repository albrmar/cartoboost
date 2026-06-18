"""Forecasting metrics with horizon and series breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ForecastMetricSet:
    """Compute point, quantile, and interval forecast metrics."""

    seasonal_period: int = 1
    quantiles: tuple[float, ...] = field(default_factory=tuple)

    def evaluate(
        self,
        y_true: Any,
        y_pred: Any,
        *,
        horizon: Any | None = None,
        series_id: Any | None = None,
        y_train: Any | None = None,
        quantile_predictions: dict[float, Any] | None = None,
        lower: Any | None = None,
        upper: Any | None = None,
    ) -> dict[str, Any]:
        truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
        result = _point_metrics(truth, pred)
        result["mase"] = _mase(truth, pred, y_train, self.seasonal_period)

        if horizon is not None:
            result["per_horizon"] = _grouped_metrics(truth, pred, horizon)
        if series_id is not None:
            result["per_series"] = _grouped_metrics(truth, pred, series_id)

        quantile_predictions = quantile_predictions or {}
        quantile_scores: dict[str, float] = {}
        for q in self.quantiles:
            if q not in quantile_predictions:
                continue
            _, q_pred = _paired(truth, quantile_predictions[q], "y_true", f"q{q}")
            quantile_scores[str(q)] = pinball_loss(truth, q_pred, q)
        if quantile_scores:
            result["pinball"] = quantile_scores
            result["pinball_mean"] = float(np.mean(list(quantile_scores.values())))

        if lower is not None or upper is not None:
            if lower is None or upper is None:
                raise ValueError("lower and upper interval bounds must be provided together")
            _, lower_arr = _paired(truth, lower, "y_true", "lower")
            _, upper_arr = _paired(truth, upper, "y_true", "upper")
            widths = upper_arr - lower_arr
            if np.any(widths < 0):
                raise ValueError("lower bounds must be less than or equal to upper bounds")
            covered = (truth >= lower_arr) & (truth <= upper_arr)
            result["coverage"] = float(np.mean(covered))
            result["interval_width"] = float(np.mean(widths))

        return result

    def evaluate_frame(
        self,
        frame: Any,
        *,
        actual_col: str = "actual",
        prediction_col: str = "prediction",
        horizon_col: str = "horizon",
        series_id_col: str = "series_id",
        lower_col: str | None = "lower",
        upper_col: str | None = "upper",
    ) -> dict[str, Any]:
        lower = _optional_column(frame, lower_col)
        upper = _optional_column(frame, upper_col)
        return self.evaluate(
            _column(frame, actual_col),
            _column(frame, prediction_col),
            horizon=_optional_column(frame, horizon_col),
            series_id=_optional_column(frame, series_id_col),
            lower=lower,
            upper=upper,
        )


def pinball_loss(y_true: Any, y_pred: Any, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
    residual = truth - pred
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    abs_error = np.abs(error)
    denominator = np.abs(y_true)
    nonzero = denominator > 0
    smape_denominator = np.abs(y_true) + np.abs(y_pred)
    smape_mask = smape_denominator > 0
    total_abs_truth = float(np.sum(denominator))
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mape": (
            float(np.mean(abs_error[nonzero] / denominator[nonzero])) if np.any(nonzero) else 0.0
        ),
        "smape": (
            float(np.mean(2.0 * abs_error[smape_mask] / smape_denominator[smape_mask]))
            if np.any(smape_mask)
            else 0.0
        ),
        "wape": float(np.sum(abs_error) / total_abs_truth) if total_abs_truth > 0 else 0.0,
        "bias": float(np.mean(error)),
    }


def _mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Any | None,
    seasonal_period: int,
) -> float:
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive")
    if y_train is None:
        return float("nan")
    train = _vector(y_train, "y_train")
    if train.size <= seasonal_period:
        raise ValueError("y_train must be longer than seasonal_period for MASE")
    scale = float(np.mean(np.abs(train[seasonal_period:] - train[:-seasonal_period])))
    if scale == 0.0:
        raise ValueError("MASE scale is zero")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Any,
) -> dict[str, dict[str, float]]:
    group_arr = np.asarray(groups)
    if group_arr.shape != y_true.shape:
        raise ValueError("group arrays must match y_true shape")
    out: dict[str, dict[str, float]] = {}
    for group in np.unique(group_arr):
        mask = group_arr == group
        out[str(group)] = _point_metrics(y_true[mask], y_pred[mask])
    return out


def _paired(
    left: Any,
    right: Any,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_arr = _vector(left, left_name)
    right_arr = _vector(right, right_name)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")
    if left_arr.size == 0:
        raise ValueError(f"{left_name} and {right_name} must contain at least one value")
    return left_arr, right_arr


def _vector(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _column(frame: Any, name: str) -> Any:
    try:
        return frame[name]
    except Exception as exc:
        raise ValueError(f"frame must contain column {name!r}") from exc


def _optional_column(frame: Any, name: str | None) -> Any | None:
    if name is None:
        return None
    try:
        return frame[name]
    except Exception:
        return None
