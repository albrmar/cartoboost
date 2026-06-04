from pathlib import Path

import pytest
from geoboost import FeatureSchema, GeoBoostRegressor


def _fit_or_skip(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def test_python_native_sparse_list_route_cells_train_predict():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    schema = FeatureSchema(dense=[("bias", "numeric")], sparse_sets=[("route_cells", "sparse_set")])
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )

    _fit_or_skip(model, x, y, sparse_sets=sparse_sets, feature_schema=schema)

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
    )
    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)
    path = tmp_path / "sparse-list.json"
    before = model.predict(x, sparse_sets=sparse_sets)

    model.save(path)
    loaded = GeoBoostRegressor.load(path)

    assert loaded.predict(x, sparse_sets=sparse_sets) == pytest.approx(before)
    assert loaded.feature_schema_["names"] == ["feature_0", "route_cells"]


def test_unseen_sparse_ids_route_correctly():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [5.0, 5.0, -1.0, -1.0]
    train_sparse = {"route_cells": [[7], [7, 11], [3], []]}
    test_sparse = {"route_cells": [[99], [7], [], [3, 99]]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )
    _fit_or_skip(model, x, y, sparse_sets=train_sparse)

    predictions = model.predict(x, sparse_sets=test_sparse)

    assert predictions[1] > predictions[0]
    assert predictions[0] == pytest.approx(predictions[2])


def test_empty_sparse_rows_route_correctly():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [5.0, 5.0, -1.0, -1.0]
    sparse_sets = {"route_cells": [[7], [7], [], []]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )
    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)

    predictions = model.predict(x, sparse_sets=sparse_sets)

    assert predictions[:2] == pytest.approx([5.0, 5.0])
    assert predictions[2:] == pytest.approx([-1.0, -1.0])


def test_duplicate_sparse_ids_do_not_change_predictions():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [5.0, 5.0, -1.0, -1.0]
    train_sparse = {"route_cells": [[7], [7], [3], []]}
    deduped_sparse = {"route_cells": [[7], [7], [3], []]}
    duplicated_sparse = {"route_cells": [[7, 7, 7], [7], [3, 3], []]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )
    _fit_or_skip(model, x, y, sparse_sets=train_sparse)

    assert model.predict(x, sparse_sets=duplicated_sparse) == pytest.approx(
        model.predict(x, sparse_sets=deduped_sparse)
    )


def test_sparse_prediction_dict_order_uses_fitted_column_order():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [5.0, 5.0, -1.0, -1.0]
    sparse_sets = {
        "route_cells": [[7], [7], [], []],
        "unused_cells": [[], [], [100], [100]],
    }
    reordered = {
        "unused_cells": sparse_sets["unused_cells"],
        "route_cells": sparse_sets["route_cells"],
    }
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )
    _fit_or_skip(model, x, y, sparse_sets=sparse_sets)

    assert model.predict(x, sparse_sets=reordered) == pytest.approx(
        model.predict(x, sparse_sets=sparse_sets)
    )


def test_sparse_prediction_list_input_is_positional():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [5.0, 5.0, -1.0, -1.0]
    sparse_columns = [[[7], [7], [], []]]
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
    )
    _fit_or_skip(model, x, y, sparse_sets=sparse_columns)

    assert model.predict(x, sparse_sets=sparse_columns) == pytest.approx(y)


def test_sparse_row_count_mismatch_raises():
    model = GeoBoostRegressor(max_depth=0)

    with pytest.raises(ValueError, match="same number of rows"):
        model.fit([[0.0], [1.0]], [0.0, 1.0], sparse_sets={"route_cells": [[7]]})


def test_non_integer_sparse_id_raises():
    model = GeoBoostRegressor(max_depth=0)

    with pytest.raises(ValueError, match="non-negative integers"):
        model.fit([[0.0]], [0.0], sparse_sets={"route_cells": [[1.5]]})


def test_negative_sparse_id_raises():
    model = GeoBoostRegressor(max_depth=0)

    with pytest.raises(ValueError, match="negative ID"):
        model.fit([[0.0]], [0.0], sparse_sets={"route_cells": [[-1]]})


def test_mixed_fit_preserves_sample_weight_validation():
    model = GeoBoostRegressor(max_depth=0)

    with pytest.raises(ValueError, match="sample_weight length"):
        model.fit(
            [[0.0], [1.0]],
            [0.0, 1.0],
            sample_weight=[1.0],
            sparse_sets={"route_cells": [[7], [8]]},
        )
