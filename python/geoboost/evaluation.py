"""Evaluation protocol helpers for blocked GeoBoost validation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = [
    "grouped_blocked_cv",
    "spatial_blocked_cv",
    "temporal_blocked_cv",
]


def spatial_blocked_cv(
    coordinates: object,
    *,
    n_splits: int = 5,
    grid_shape: tuple[int, int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices using held-out spatial grid blocks.

    Coordinates can be one-dimensional or multi-dimensional. For dimensions
    beyond the first two, blocking uses the first two columns.
    """

    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] == 0:
        raise ValueError("coordinates must have shape (n_samples, n_dimensions)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates must contain only finite values")
    _validate_n_splits(n_splits, coords.shape[0])

    if coords.shape[1] == 1:
        yield from _ordered_blocked_cv(coords[:, 0], n_splits)
        return

    rows, cols = _resolve_grid_shape(n_splits, grid_shape)
    x_bins = _coordinate_bins(coords[:, 0], cols)
    y_bins = _coordinate_bins(coords[:, 1], rows)
    block_ids = y_bins * cols + x_bins
    if np.unique(block_ids).size < n_splits:
        ranges = np.ptp(coords[:, :2], axis=0)
        yield from _ordered_blocked_cv(coords[:, int(np.argmax(ranges))], n_splits)
        return
    yield from grouped_blocked_cv(block_ids, n_splits=n_splits)


def temporal_blocked_cv(
    times: object,
    *,
    n_splits: int = 5,
    gap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices from contiguous time-ordered folds.

    ``gap`` removes that many neighboring sorted rows on each side of the test
    block from the corresponding training split.
    """

    time_values = np.asarray(times)
    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional")
    _validate_n_splits(n_splits, time_values.shape[0])
    if gap < 0:
        raise ValueError("gap must be non-negative")

    order = np.argsort(time_values, kind="mergesort")
    positions = np.arange(order.size)
    for test_positions in np.array_split(positions, n_splits):
        test_idx = order[test_positions]
        blocked = np.zeros(order.size, dtype=bool)
        start = max(0, int(test_positions[0]) - gap)
        stop = min(order.size, int(test_positions[-1]) + gap + 1)
        blocked[start:stop] = True
        train_idx = order[~blocked]
        yield np.sort(train_idx), np.sort(test_idx)


def grouped_blocked_cv(
    groups: object,
    *,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices with whole groups held out together."""

    group_values = np.asarray(groups)
    if group_values.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    _validate_n_splits(n_splits, group_values.shape[0])

    unique_groups, inverse, counts = np.unique(
        group_values,
        return_inverse=True,
        return_counts=True,
    )
    if unique_groups.size < n_splits:
        raise ValueError("n_splits cannot exceed the number of unique groups")

    fold_group_indices = _balanced_group_folds(counts, n_splits)
    all_indices = np.arange(group_values.shape[0])
    for fold_groups in fold_group_indices:
        test_mask = np.isin(inverse, fold_groups)
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]
        yield train_idx, test_idx


def _validate_n_splits(n_splits: int, n_samples: int) -> None:
    if not isinstance(n_splits, int):
        raise ValueError("n_splits must be an integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed the number of samples")


def _resolve_grid_shape(
    n_splits: int,
    grid_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if grid_shape is not None:
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must contain two integers")
        rows, cols = int(grid_shape[0]), int(grid_shape[1])
        if rows <= 0 or cols <= 0:
            raise ValueError("grid_shape values must be positive")
        if rows * cols < n_splits:
            raise ValueError("grid_shape must provide at least n_splits blocks")
        return rows, cols

    cols = int(np.ceil(np.sqrt(n_splits)))
    rows = int(np.ceil(n_splits / cols))
    return rows, cols


def _coordinate_bins(values: np.ndarray, bin_count: int) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        return np.zeros(values.shape[0], dtype=int)
    scaled = (values - minimum) / (maximum - minimum)
    return np.minimum((scaled * bin_count).astype(int), bin_count - 1)


def _balanced_group_folds(counts: np.ndarray, n_splits: int) -> list[np.ndarray]:
    fold_loads = np.zeros(n_splits, dtype=int)
    fold_groups: list[list[int]] = [[] for _ in range(n_splits)]
    for group_index in np.argsort(-counts, kind="mergesort"):
        fold_index = int(np.argmin(fold_loads))
        fold_groups[fold_index].append(int(group_index))
        fold_loads[fold_index] += int(counts[group_index])
    return [np.asarray(groups, dtype=int) for groups in fold_groups]


def _ordered_blocked_cv(
    values: np.ndarray,
    n_splits: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(values, kind="mergesort")
    sorted_folds = np.array_split(order, n_splits)
    for test_idx in sorted_folds:
        yield _complement_indices(values.shape[0], test_idx), np.sort(test_idx)


def _complement_indices(n_samples: int, test_idx: np.ndarray) -> np.ndarray:
    mask = np.ones(n_samples, dtype=bool)
    mask[test_idx] = False
    return np.nonzero(mask)[0]
