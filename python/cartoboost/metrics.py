"""Metrics and calibration helpers for CartoBoost regression workflows."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np

from . import _native

__path__ = [str(Path(__file__).with_suffix(""))]

__all__ = [
    "calibrated_intervals",
    "conformal_residual_quantile",
    "interval_coverage",
    "jitter_volatility",
    "mean_interval_width",
    "m_competition_metrics",
    "m5_equal_level_wrmsse",
    "pinball_loss",
    "residual_morans_i",
    "rmsse_scale",
    "wrmsse",
]


def conformal_residual_quantile(
    y_true: object,
    y_pred: object,
    *,
    alpha: float = 0.1,
) -> float:
    """Return the finite-sample split-conformal absolute residual quantile.

    ``alpha`` is the tolerated miscoverage rate, so ``alpha=0.1`` targets 90%
    marginal coverage. The returned quantile can be added to and subtracted
    from point predictions to form symmetric calibrated intervals.
    """

    _validate_alpha(alpha)
    y_true_arr, y_pred_arr = _paired_vectors(y_true, y_pred, "y_true", "y_pred")
    scores = np.abs(y_true_arr - y_pred_arr)
    if scores.size == 0:
        raise ValueError("calibration data must contain at least one row")

    quantile_level = min(1.0, np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size)
    return float(np.quantile(scores, quantile_level, method="higher"))


def calibrated_intervals(
    y_pred: object,
    residual_quantile: float | None = None,
    *,
    y_calibration: object | None = None,
    calibration_predictions: object | None = None,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric conformal intervals around predictions.

    Pass either a precomputed ``residual_quantile`` or calibration truth and
    predictions. The helper returns ``(lower, upper)`` arrays matching
    ``y_pred``.
    """

    predictions = _as_float_vector(y_pred, "y_pred")
    if residual_quantile is None:
        if y_calibration is None or calibration_predictions is None:
            raise ValueError(
                "provide residual_quantile or both y_calibration and calibration_predictions"
            )
        residual_quantile = conformal_residual_quantile(
            y_calibration,
            calibration_predictions,
            alpha=alpha,
        )
    q = float(residual_quantile)
    if not np.isfinite(q) or q < 0.0:
        raise ValueError("residual_quantile must be a finite non-negative value")

    return predictions - q, predictions + q


def pinball_loss(
    y_true: object,
    y_pred: object,
    *,
    quantile: float,
    sample_weight: object | None = None,
) -> float:
    """Return mean pinball loss for a quantile prediction."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    y_true_arr, y_pred_arr = _paired_vectors(y_true, y_pred, "y_true", "y_pred")
    residual = y_true_arr - y_pred_arr
    losses = np.maximum(quantile * residual, (quantile - 1.0) * residual)
    return _weighted_mean(losses, sample_weight)


def interval_coverage(
    y_true: object,
    lower: object,
    upper: object,
    *,
    sample_weight: object | None = None,
) -> float:
    """Return the share of targets inside inclusive prediction intervals."""

    y_true_arr, lower_arr = _paired_vectors(y_true, lower, "y_true", "lower")
    _, upper_arr = _paired_vectors(y_true_arr, upper, "y_true", "upper")
    if np.any(lower_arr > upper_arr):
        raise ValueError("lower bounds must be less than or equal to upper bounds")
    covered = (y_true_arr >= lower_arr) & (y_true_arr <= upper_arr)
    return _weighted_mean(covered.astype(float), sample_weight)


def mean_interval_width(
    lower: object,
    upper: object,
    *,
    sample_weight: object | None = None,
) -> float:
    """Return the mean width of prediction intervals."""

    lower_arr, upper_arr = _paired_vectors(lower, upper, "lower", "upper")
    widths = upper_arr - lower_arr
    if np.any(widths < 0.0):
        raise ValueError("lower bounds must be less than or equal to upper bounds")
    return _weighted_mean(widths, sample_weight)


def rmsse_scale(y_train: object, *, seasonal_period: int = 1) -> float:
    """Return the squared seasonal-naive scale used by RMSSE and WRMSSE."""

    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive")
    train = _as_float_vector(y_train, "y_train")
    return float(_native.rmsse_scale_value(train.tolist(), int(seasonal_period)))


def wrmsse(
    y_train: object,
    y_true: object,
    y_pred: object,
    weights: object,
    *,
    seasonal_period: int = 1,
    series_ids: object | None = None,
    return_breakdown: bool = False,
) -> float | dict[str, object]:
    """Return official-style weighted RMSSE for hierarchical forecasting.

    Inputs are shaped ``(n_series, n_time)`` for training history and
    ``(n_series, horizon)`` for actual and predicted forecast windows. Weights
    are non-negative value weights, such as M5 dollar-sales weights; they are
    normalized inside the metric.
    """

    train = _as_float_matrix(y_train, "y_train")
    truth = _as_float_matrix(y_true, "y_true")
    pred = _as_float_matrix(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if train.shape[0] != truth.shape[0]:
        raise ValueError("y_train, y_true, and y_pred must have the same number of series")
    if truth.shape[1] == 0:
        raise ValueError("y_true and y_pred must contain at least one horizon")

    weight_arr = _as_float_vector(weights, "weights")
    if weight_arr.shape[0] != truth.shape[0]:
        raise ValueError("weights length must match the number of series")
    if series_ids is None:
        ids = [str(idx) for idx in range(truth.shape[0])]
    else:
        ids_arr = np.asarray(series_ids, dtype=object)
        if ids_arr.ndim != 1 or ids_arr.shape[0] != truth.shape[0]:
            raise ValueError("series_ids must be one-dimensional and match the number of series")
        ids = [str(value) for value in ids_arr]
        if any(not value for value in ids):
            raise ValueError("series_ids must be non-empty")

    series = [
        (
            ids[idx],
            train[idx].tolist(),
            truth[idx].tolist(),
            pred[idx].tolist(),
            float(weight_arr[idx]),
        )
        for idx in range(truth.shape[0])
    ]
    result = json.loads(_native.wrmsse_value(series, int(seasonal_period)))
    score = float(result["wrmsse"])

    if not return_breakdown:
        return score

    return result


def m5_equal_level_wrmsse(
    level_scores: Iterable[dict[str, object] | tuple[object, object]],
    *,
    return_breakdown: bool = False,
) -> float | dict[str, object]:
    """Aggregate the 12 M5 hierarchy-level WRMSSE values with equal level weight."""

    rows: list[tuple[str, float]] = []
    for row in level_scores:
        if isinstance(row, dict):
            level = str(row["level"])
            score = float(row["wrmsse"])
        else:
            level, score = row
            level = str(level)
            score = float(score)
        rows.append((level, score))
    result = json.loads(_native.m5_equal_level_wrmsse_value(rows))
    score = float(result["wrmsse"])
    if not return_breakdown:
        return score
    return result


def m_competition_metrics(
    training_series: object,
    y_true: object,
    y_pred: object,
    *,
    seasonality: int,
    baseline_smape: float | None = None,
    baseline_mase: float | None = None,
) -> dict[str, float | None]:
    """Return M/M1-M3-M4 style sMAPE, MASE, and optional OWA metrics."""

    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    train_rows = [
        _as_float_vector(row, "training_series row").tolist()
        for row in np.asarray(training_series, dtype=object)
    ]
    truth, pred = _paired_vectors(y_true, y_pred, "y_true", "y_pred")
    payload = _native.m_competition_metrics_value(
        train_rows,
        truth.tolist(),
        pred.tolist(),
        int(seasonality),
        baseline_smape,
        baseline_mase,
    )
    return json.loads(payload)


def jitter_volatility(
    predictions: object,
    *,
    baseline: object | None = None,
    sample_weight: object | None = None,
) -> float:
    """Measure prediction instability under repeated coordinate jitter.

    ``predictions`` is expected to have shape ``(n_repeats, n_samples)``. Without
    a baseline, the metric is the weighted mean per-sample standard deviation
    across repeats. With a baseline, it is the weighted mean root-mean-square
    deviation from that baseline across repeats.
    """

    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2:
        raise ValueError("predictions must have shape (n_repeats, n_samples)")
    if values.shape[0] < 2:
        raise ValueError("predictions must contain at least two jitter repeats")
    if values.shape[1] == 0:
        raise ValueError("predictions must contain at least one sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("predictions must contain only finite values")

    if baseline is None:
        per_sample = np.std(values, axis=0)
    else:
        baseline_arr = _as_float_vector(baseline, "baseline")
        if baseline_arr.shape[0] != values.shape[1]:
            raise ValueError("baseline length must match the number of samples")
        per_sample = np.sqrt(np.mean((values - baseline_arr[None, :]) ** 2, axis=0))
    return _weighted_mean(per_sample, sample_weight)


def residual_morans_i(
    coordinates: object,
    residuals: object,
    *,
    weights: Literal["inverse_distance", "radius"] = "inverse_distance",
    radius: float | None = None,
    distance_epsilon: float = 1e-12,
) -> float:
    """Return Moran's I for residual spatial autocorrelation.

    The implementation uses dense pairwise weights and is intended for
    evaluation samples, not very large production batches.
    """

    coords = _as_coordinates(coordinates)
    residual_arr = _as_float_vector(residuals, "residuals")
    if coords.shape[0] != residual_arr.shape[0]:
        raise ValueError("coordinates and residuals must contain the same number of rows")
    if coords.shape[0] < 2:
        raise ValueError("at least two rows are required")
    if not np.isfinite(distance_epsilon) or distance_epsilon <= 0.0:
        raise ValueError("distance_epsilon must be a finite positive value")

    centered = residual_arr - np.mean(residual_arr)
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("residuals must have non-zero variance")

    distances = _pairwise_distances(coords)
    if weights == "inverse_distance":
        weight_matrix = 1.0 / np.maximum(distances, distance_epsilon)
        np.fill_diagonal(weight_matrix, 0.0)
    elif weights == "radius":
        if radius is None or not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be a finite positive value when weights='radius'")
        weight_matrix = ((distances <= radius) & (distances > 0.0)).astype(float)
    else:
        raise ValueError("weights must be 'inverse_distance' or 'radius'")

    weight_sum = float(np.sum(weight_matrix))
    if weight_sum == 0.0:
        raise ValueError("spatial weights contain no neighbor pairs")

    numerator = float(centered @ weight_matrix @ centered)
    return coords.shape[0] / weight_sum * numerator / denominator


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")


def _as_float_vector(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_float_matrix(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _paired_vectors(
    left: object,
    right: object,
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[np.ndarray, np.ndarray]:
    left_arr = _as_float_vector(left, left_name)
    right_arr = _as_float_vector(right, right_name)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")
    if left_arr.size == 0:
        raise ValueError(f"{left_name} and {right_name} must contain at least one value")
    return left_arr, right_arr


def _weighted_mean(values: np.ndarray, sample_weight: object | None) -> float:
    if values.size == 0:
        raise ValueError("values must contain at least one row")
    if sample_weight is None:
        return float(np.mean(values))
    weights = _as_float_vector(sample_weight, "sample_weight")
    if weights.shape != values.shape:
        raise ValueError("sample_weight must match the metric value shape")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be non-negative")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("sample_weight must have positive total weight")
    return float(np.average(values, weights=weights))


def _as_coordinates(coordinates: object) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] == 0:
        raise ValueError("coordinates must have shape (n_samples, n_dimensions)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates must contain only finite values")
    return coords


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))
