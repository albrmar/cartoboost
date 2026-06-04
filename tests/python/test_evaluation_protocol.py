import numpy as np
import pytest
from geoboost import grouped_blocked_cv, spatial_blocked_cv, temporal_blocked_cv


def test_grouped_blocked_cv_holds_out_complete_groups():
    groups = np.array(["a", "a", "b", "b", "c", "d"])

    splits = list(grouped_blocked_cv(groups, n_splits=3))

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        assert len(test_idx) > 0


def test_temporal_blocked_cv_uses_contiguous_sorted_test_blocks_with_gap():
    times = np.array([5, 1, 4, 2, 3, 6])

    splits = list(temporal_blocked_cv(times, n_splits=3, gap=1))
    first_train, first_test = splits[0]

    assert len(splits) == 3
    assert set(first_test) == {1, 3}
    assert set(first_train) == {0, 2, 5}
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))


def test_spatial_blocked_cv_holds_out_grid_blocks():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.0, 10.0],
            [0.2, 10.2],
            [10.0, 0.0],
            [10.2, 0.2],
            [10.0, 10.0],
            [10.2, 10.2],
        ]
    )

    splits = list(spatial_blocked_cv(coordinates, n_splits=4, grid_shape=(2, 2)))

    assert len(splits) == 4
    assert sorted(np.concatenate([test_idx for _, test_idx in splits]).tolist()) == list(range(8))
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(test_idx) == 2


def test_spatial_blocked_cv_supports_one_dimensional_coordinates():
    splits = list(spatial_blocked_cv([3.0, 1.0, 2.0, 4.0], n_splits=2))

    assert [test_idx.tolist() for _, test_idx in splits] == [[1, 2], [0, 3]]


def test_blocked_cv_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        list(grouped_blocked_cv([1, 2, 3], n_splits=1))
    with pytest.raises(ValueError, match="unique groups"):
        list(grouped_blocked_cv(["a", "a", "b"], n_splits=3))
    with pytest.raises(ValueError, match="gap must be non-negative"):
        list(temporal_blocked_cv([1, 2, 3], n_splits=2, gap=-1))
    with pytest.raises(ValueError, match="grid_shape"):
        list(spatial_blocked_cv([[0.0, 0.0], [1.0, 1.0]], n_splits=2, grid_shape=(1, 1)))
