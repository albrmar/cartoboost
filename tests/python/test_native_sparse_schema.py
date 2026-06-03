from pathlib import Path

import pytest
from geoboost import GeoBoostRegressor


def _fit_or_skip(model, *args, **kwargs):
    try:
        model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    return model


def test_python_native_sparse_list_route_cells_train_predict():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
        backend="rust",
    )

    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)

    assert model.predict(x, sparse_sets=sparse_sets) == pytest.approx(y)
    with pytest.raises(ValueError, match="sparse_sets"):
        model.predict(x)


def test_python_native_sparse_list_two_tree_boosting():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [10.0, 10.0, 0.0, 0.0]
    sparse_sets = {"route_cells": [[7, 11], [11], [3], []]}
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
        backend="rust",
    )

    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)

    assert model.predict(x, sparse_sets=sparse_sets) == pytest.approx([8.75, 8.75, 1.25, 1.25])


def test_python_native_sparse_list_save_load_identity(tmp_path: Path):
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
        backend="rust",
    )
    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)
    path = tmp_path / "sparse-list.json"
    before = model.predict(x, sparse_sets=sparse_sets)

    model.save(path)
    loaded = GeoBoostRegressor.load(path)

    assert loaded.predict(x, sparse_sets=sparse_sets) == pytest.approx(before)
    assert loaded.feature_schema_["names"] == ["feature_0", "route_cells"]


def test_python_schema_periodic_feature_used_without_full_period_coverage():
    x = [[7.0], [8.0], [9.0], [10.0]]
    y = [3.0, 3.0, -1.0, -1.0]
    schema = {"dense": [{"name": "hour", "kind": "periodic", "period": 24}]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["periodic:24"],
        backend="rust",
    )

    _fit_or_skip(model, x, y, feature_schema=schema)

    assert model.predict(x) == pytest.approx(y)
    assert model.feature_schema_["names"] == ["hour"]


def test_python_schema_rejects_length_mismatch():
    model = GeoBoostRegressor(max_depth=0, backend="python")

    with pytest.raises(ValueError, match="feature_schema length"):
        model.fit(
            [[0.0, 1.0], [2.0, 3.0]],
            [1.0, 2.0],
            feature_schema={"dense": [{"name": "only_one", "kind": "numeric"}]},
        )


def test_real_native_save_load_restores_public_params(tmp_path: Path):
    x = [[0.0, 0.0], [0.2, 0.0], [3.0, 0.0], [0.0, 3.0]]
    y = [5.0, 5.0, -1.0, -1.0]
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.25,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["gaussian_2d"],
        fuzzy=True,
        fuzzy_bandwidth=0.5,
        leaf_predictor="constant",
        backend="rust",
    )
    _fit_or_skip(model, x, y)
    path = tmp_path / "native.json"

    model.save(path)
    loaded = GeoBoostRegressor.load(path)

    assert loaded.splitters == ["gaussian_2d"]
    assert loaded.fuzzy is True
    assert loaded.fuzzy_bandwidth == 0.5
    assert loaded.learning_rate == 0.25
    assert loaded.n_estimators == 2
    assert loaded.predict(x) == pytest.approx(model.predict(x))
