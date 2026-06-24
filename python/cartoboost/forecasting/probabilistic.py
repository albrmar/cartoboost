"""Probabilistic forecasting helpers for quantiles, conformal intervals, and RPS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuantileForecast:
    quantiles: tuple[float, ...]
    values: tuple[float, ...]

    def repaired(self) -> QuantileForecast:
        return QuantileForecast(self.quantiles, tuple(repair_non_crossing_quantiles(self.values)))


@dataclass(frozen=True)
class ConformalInterval:
    lower: np.ndarray
    upper: np.ndarray
    residual_quantile: float
    alpha: float


class ConformalCalibrator:
    """Split-conformal calibration with explicit train/calibration/test ordering."""

    def __init__(self, *, alpha: float = 0.1) -> None:
        _validate_quantile(alpha, "alpha")
        self.alpha = float(alpha)
        self.residual_quantile_: float | None = None
        self._test_start: int | None = None

    def fit(
        self,
        calibration_actual: Any,
        calibration_prediction: Any,
        *,
        train_end_exclusive: int,
        calibration_start: int,
        calibration_end_exclusive: int,
        test_start: int,
    ) -> ConformalCalibrator:
        _validate_strict_ordering(
            train_end_exclusive,
            calibration_start,
            calibration_end_exclusive,
            test_start,
        )
        actual, prediction = _paired(
            calibration_actual,
            calibration_prediction,
            "calibration_actual",
            "calibration_prediction",
        )
        scores = np.sort(np.abs(actual - prediction))
        rank = int(np.ceil((scores.size + 1) * (1.0 - self.alpha)))
        self.residual_quantile_ = float(scores[min(max(rank - 1, 0), scores.size - 1)])
        self._test_start = int(test_start)
        return self

    def predict_interval(self, test_prediction: Any, *, test_start: int) -> ConformalInterval:
        if self.residual_quantile_ is None or self._test_start is None:
            raise ValueError("ConformalCalibrator must be fit before prediction")
        if int(test_start) < self._test_start:
            raise ValueError("test_start must not precede the calibrated test split")
        prediction = _vector(test_prediction, "test_prediction")
        q = self.residual_quantile_
        return ConformalInterval(
            lower=prediction - q,
            upper=prediction + q,
            residual_quantile=q,
            alpha=self.alpha,
        )


def pinball_loss(y_true: Any, y_pred: Any, quantile: float) -> float:
    _validate_quantile(quantile, "quantile")
    truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
    residual = truth - pred
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def repair_non_crossing_quantiles(values: Any) -> np.ndarray:
    arr = _vector(values, "values")
    if arr.size == 0:
        raise ValueError("values must contain at least one quantile prediction")
    return np.maximum.accumulate(arr)


def rank_probability_score(probabilities: Any, observed_rank: int) -> float:
    probs = _probabilities(probabilities)
    rank = int(observed_rank)
    if rank < 0 or rank >= probs.size:
        raise ValueError("observed_rank must be a zero-based index inside probabilities")
    if probs.size == 1:
        return 0.0
    predicted_cdf = np.cumsum(probs[:-1])
    observed_cdf = (rank <= np.arange(probs.size - 1)).astype(float)
    return float(np.mean((predicted_cdf - observed_cdf) ** 2))


def _validate_strict_ordering(
    train_end_exclusive: int,
    calibration_start: int,
    calibration_end_exclusive: int,
    test_start: int,
) -> None:
    if int(train_end_exclusive) <= 0:
        raise ValueError("training split must contain at least one row")
    if int(train_end_exclusive) > int(calibration_start):
        raise ValueError("training rows must end before calibration rows start")
    if int(calibration_start) >= int(calibration_end_exclusive):
        raise ValueError("calibration split must contain at least one row")
    if int(calibration_end_exclusive) > int(test_start):
        raise ValueError("calibration rows must end before test rows start")


def _validate_quantile(value: float, name: str) -> None:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1)")


def _paired(
    left: Any, right: Any, left_name: str, right_name: str
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


def _probabilities(values: Any) -> np.ndarray:
    arr = _vector(values, "probabilities")
    if arr.size == 0:
        raise ValueError("probabilities must contain at least one rank")
    if np.any(arr < 0.0):
        raise ValueError("probabilities must be non-negative")
    if not np.isclose(float(np.sum(arr)), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("probabilities must sum to 1")
    return arr
