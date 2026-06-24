"""Evaluation helpers for link ranking and graph-neighborhood tasks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def _safe_float_array(values: Sequence[float] | Sequence[int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("metric input must be 1D")
    return array


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC-AUC without external dependencies."""
    y = _safe_float_array(labels)
    s = _safe_float_array(scores)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have equal length")
    positive_mask = y > 0.5
    negative_mask = y <= 0.5
    if not positive_mask.any() or not negative_mask.any():
        return 0.0

    pos_scores = s[positive_mask]
    neg_scores = s[negative_mask]
    greater = 0.0
    ties = 0.0
    for pos in pos_scores:
        greater += float(np.sum(neg_scores < pos))
        ties += float(np.sum(neg_scores == pos))
    return (greater + 0.5 * ties) / (float(len(pos_scores)) * float(len(neg_scores)))


def binary_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute area under the precision-recall staircase."""
    y = _safe_float_array(labels)
    s = _safe_float_array(scores)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have equal length")

    order = np.argsort(-s)
    sorted_labels = y[order]
    positive_count = float(np.sum(sorted_labels > 0.5))
    if positive_count == 0.0:
        return 0.0

    tp = 0.0
    precision_acc = 0.0
    for index, label in enumerate(sorted_labels, start=1):
        if label > 0.5:
            tp += 1.0
            precision_acc += tp / float(index)
    return precision_acc / positive_count


def top_k_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    query_ids: Sequence[Any],
    k: int,
) -> dict[str, float]:
    """Group-wise top-k recall-style metrics."""
    y = _safe_float_array(labels)
    s = _safe_float_array(scores)
    query_values = list(query_ids)
    if not (y.shape == s.shape == (len(query_values),)):
        raise ValueError("labels, scores, and query_ids must align")
    if k <= 0:
        raise ValueError("k must be positive")

    query_to_indices: dict[Any, list[int]] = defaultdict(list)
    for index, key in enumerate(query_values):
        query_to_indices[key].append(index)

    recalls: list[float] = []
    hit_rates: list[float] = []
    for indices in query_to_indices.values():
        if not indices:
            continue
        ordered = sorted(indices, key=lambda index: s[index], reverse=True)
        topk = ordered[:k]
        positives = [i for i in indices if y[i] > 0.5]
        positives_in_topk = [i for i in topk if y[i] > 0.5]
        if positives:
            recalls.append(len(positives_in_topk) / float(len(positives)))
            hit_rates.append(1.0 if positives_in_topk else 0.0)

    if not recalls:
        return {"recall_at_k": 0.0, "hit_rate_at_k": 0.0}
    return {
        "recall_at_k": float(np.mean(recalls)),
        "hit_rate_at_k": float(np.mean(hit_rates)),
    }


def mean_reciprocal_rank(
    labels: Sequence[int],
    scores: Sequence[float],
    query_ids: Sequence[Any],
) -> float:
    """Mean reciprocal rank over grouped candidate sets."""
    y = _safe_float_array(labels)
    s = _safe_float_array(scores)
    query_values = list(query_ids)
    if not (y.shape == s.shape == (len(query_values),)):
        raise ValueError("labels, scores, and query_ids must align")

    query_to_indices: dict[Any, list[int]] = defaultdict(list)
    for index, query_id in enumerate(query_values):
        query_to_indices[query_id].append(index)

    ranks: list[float] = []
    for indices in query_to_indices.values():
        positives = [index for index in indices if y[index] > 0.5]
        if not positives:
            continue
        ordered = sorted(indices, key=lambda idx: s[idx], reverse=True)
        best_pos = min(
            (rank + 1 for rank, idx in enumerate(ordered) if idx in positives),
            default=None,
        )
        if best_pos is not None:
            ranks.append(1.0 / float(best_pos))

    if not ranks:
        return 0.0
    return float(np.mean(ranks))


def link_prediction_report(
    labels: Sequence[int],
    scores: Sequence[float],
    query_ids: Sequence[Any] | None = None,
    *,
    k: int = 10,
) -> dict[str, float]:
    """Return a compact link-prediction and link-ranking metric bundle."""
    report = {
        "auc": binary_auc(labels, scores),
        "average_precision": binary_average_precision(labels, scores),
    }
    if query_ids is None:
        return report
    report.update(top_k_metrics(labels, scores, query_ids=query_ids, k=k))
    report["mean_reciprocal_rank"] = mean_reciprocal_rank(labels, scores, query_ids)
    return report
