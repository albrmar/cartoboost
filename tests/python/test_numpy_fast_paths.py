import numpy as np
import pytest
from geoboost import FeatureSchema, GeoBoostRegressor
from geoboost.regressor import _encode_sparse_columns, _rust_feature_schema_json

try:
    from geoboost._native import GeoBoostRegressor as NativeGeoBoostRegressor
except ImportError:  # pragma: no cover - extension may be absent in source-only runs
    NativeGeoBoostRegressor = None


def _fit_or_skip(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def test_numpy_dense_fast_path_matches_list_path():
    x = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]]
    y = [0.0, 0.5, 1.0, 1.5, 2.0]
    config = dict(
        n_estimators=3,
        learning_rate=0.4,
        max_depth=2,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis"],
    )
    list_model = GeoBoostRegressor(**config)
    array_model = GeoBoostRegressor(**config)

    _fit_or_skip(list_model, x, y)
    _fit_or_skip(array_model, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))

    probes = np.asarray([[0.5, 1.0], [2.5, 0.0], [4.0, 1.0]], dtype=np.float64)
    assert array_model.predict(probes) == pytest.approx(list_model.predict(probes.tolist()))


def test_numpy_sparse_fast_path_matches_list_path():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    config = dict(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )
    list_model = GeoBoostRegressor(**config)
    array_model = GeoBoostRegressor(**config)

    _fit_or_skip(list_model, x, y, sparse_sets=sparse_sets)
    _fit_or_skip(
        array_model,
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        sparse_sets=sparse_sets,
    )

    assert array_model.predict(np.asarray(x, dtype=np.float64), sparse_sets=sparse_sets) == (
        pytest.approx(list_model.predict(x, sparse_sets=sparse_sets))
    )


def test_numpy_sample_weight_fast_path_matches_list_path():
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 0.0, 10.0, 10.0]
    sample_weight = [1.0, 3.0, 1.0, 3.0]
    config = dict(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis"],
    )
    list_model = GeoBoostRegressor(**config)
    array_model = GeoBoostRegressor(**config)

    _fit_or_skip(list_model, x, y, sample_weight=sample_weight)
    _fit_or_skip(
        array_model,
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        sample_weight=np.asarray(sample_weight, dtype=np.float64),
    )

    assert array_model.predict(np.asarray(x, dtype=np.float64)) == pytest.approx(
        list_model.predict(x)
    )


def test_numpy_feature_schema_fast_path_matches_list_path():
    x = [[23.0], [1.0], [12.0], [15.0]]
    y = [5.0, 5.0, -5.0, -5.0]
    schema = FeatureSchema(dense=[("hour", {"periodic": 24})])
    config = dict(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["periodic:24"],
    )
    list_model = GeoBoostRegressor(**config)
    array_model = GeoBoostRegressor(**config)

    _fit_or_skip(list_model, x, y, feature_schema=schema)
    _fit_or_skip(
        array_model,
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        feature_schema=schema,
    )

    assert array_model.feature_schema_ == list_model.feature_schema_
    assert array_model.predict(np.asarray(x, dtype=np.float64)) == pytest.approx(
        list_model.predict(x)
    )


def test_native_numpy_fast_paths_match_native_list_paths_exactly():
    if NativeGeoBoostRegressor is None:
        pytest.skip("native extension is not available")

    x = [[23.0, 0.0], [1.0, 0.0], [12.0, 0.0], [15.0, 0.0], [2.0, 0.0]]
    y = [4.0, 4.5, -2.0, -2.5, 5.0]
    sample_weight = [1.0, 2.0, 1.0, 2.0, 1.0]
    sparse_sets = [[[10, 20], [20], [30], [], [10]]]
    sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_sets)
    schema_json = _rust_feature_schema_json(
        {
            "dense": [
                {"name": "hour", "kind": "periodic", "period": 24},
                {"name": "dummy", "kind": "numeric"},
            ],
            "sparse_sets": [{"name": "route_cells", "kind": "sparse_set"}],
        },
        dense_width=2,
        sparse_names=["route_cells"],
    )
    config = dict(
        n_estimators=3,
        learning_rate=0.3,
        max_depth=2,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis", "periodic:24", "sparse_set"],
    )
    list_model = NativeGeoBoostRegressor(**config)
    array_model = NativeGeoBoostRegressor(**config)

    list_model.fit(x, y, sample_weight, sparse_sets, schema_json)
    array_model.fit_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(sample_weight, dtype=np.float64),
        sparse_offsets,
        sparse_ids,
        schema_json,
    )

    fast_predictions = np.asarray(
        array_model.predict_arrays(np.asarray(x, dtype=np.float64), sparse_offsets, sparse_ids)
    )
    assert fast_predictions == pytest.approx(list_model.predict(x, sparse_sets))


def test_axis_histogram_splitter_is_accepted_by_python_api():
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["axis_histogram:8"],
    )

    _fit_or_skip(
        model,
        np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float64),
    )

    predictions = model.predict(np.asarray([[0.0], [3.0]], dtype=np.float64))
    assert predictions[0] < predictions[1]
