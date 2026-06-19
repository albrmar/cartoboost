"""M6 metric helpers for probabilistic forecasts and portfolio scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class M6MetricSummary:
    pinball_loss: float
    rank_probability_score: float
    combined_score: float


def pinball_loss(y_true: Any, y_pred: Any, quantile: float) -> float:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    truth, pred = _paired(y_true, y_pred, "y_true", "y_pred")
    residual = truth - pred
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def rank_probability_score(probabilities: Any, observed_rank: int) -> float:
    probs = _probabilities(probabilities)
    rank = int(observed_rank)
    if rank < 0 or rank >= probs.size:
        raise ValueError("observed_rank must be inside probabilities")
    if probs.size == 1:
        return 0.0
    predicted_cdf = np.cumsum(probs[:-1])
    observed_cdf = (rank <= np.arange(probs.size - 1)).astype(float)
    return float(np.mean((predicted_cdf - observed_cdf) ** 2))


def m6_combined_score(pinball: float, rps: float) -> float:
    pinball = float(pinball)
    rps = float(rps)
    if not np.isfinite(pinball) or pinball < 0.0:
        raise ValueError("pinball must be finite and non-negative")
    if not np.isfinite(rps) or rps < 0.0:
        raise ValueError("rps must be finite and non-negative")
    return 0.5 * pinball + 0.5 * rps


def evaluate_m6_metrics(
    actual_returns: Any,
    quantile_predictions: Any,
    quantile: float,
    rank_probabilities: Any,
    observed_rank: int,
) -> M6MetricSummary:
    pinball = pinball_loss(actual_returns, quantile_predictions, quantile)
    rps = rank_probability_score(rank_probabilities, observed_rank)
    return M6MetricSummary(
        pinball_loss=pinball,
        rank_probability_score=rps,
        combined_score=m6_combined_score(pinball, rps),
    )


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
