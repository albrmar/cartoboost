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


def test_feature_schema_helper_builds_rust_payload():
    schema = FeatureSchema(
        dense=[
            ("distance_m", "numeric"),
            ("hour_of_day", {"periodic": 24}),
        ],
        sparse_sets=[("route_cells", "sparse_set")],
    )

    payload = schema.to_rust_payload(dense_width=2, sparse_names=["route_cells"])

    assert payload == {
        "names": ["distance_m", "hour_of_day", "route_cells"],
        "kinds": ["Numeric", {"Periodic": {"period": 24}}, "SparseSet"],
    }


def test_python_schema_periodic_feature_used_without_full_period_coverage():
    x = [[7.0], [8.0], [9.0], [10.0]]
    y = [3.0, 3.0, -1.0, -1.0]
    schema = FeatureSchema(dense=[("hour", {"periodic": 24})])
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
    assert model.feature_schema_["kinds"] == [{"Periodic": {"period": 24}}]


def test_python_schema_sparse_columns_restrict_sparse_splitter():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [4.0, 4.0, -2.0, -2.0]
    schema = FeatureSchema(
        dense=[("bias", "numeric")],
        sparse_sets=[("route_cells", "sparse_set")],
    )
    sparse_sets = {"route_cells": [[5], [5, 9], [2], []]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
        backend="rust",
    )

    _fit_or_skip(model, x, y, sparse_sets=sparse_sets, feature_schema=schema)

    assert model.predict(x, sparse_sets=sparse_sets) == pytest.approx(y)
    assert model.feature_schema_["names"] == ["bias", "route_cells"]
    assert model.feature_schema_["kinds"] == ["Numeric", "SparseSet"]


def test_python_schema_rejects_unknown_kind():
    model = GeoBoostRegressor(max_depth=0, backend="python")
    schema = FeatureSchema(dense=[("hour", "clockish")])

    with pytest.raises(ValueError, match="unknown feature kind"):
        model.fit([[0.0]], [0.0], feature_schema=schema)


def test_python_schema_rejects_length_mismatch():
    model = GeoBoostRegressor(max_depth=0, backend="python")
    schema = FeatureSchema(dense=[("only_one", "numeric")])

    with pytest.raises(ValueError, match="feature_schema length"):
        model.fit([[0.0, 1.0], [2.0, 3.0]], [1.0, 2.0], feature_schema=schema)


def test_python_schema_accepts_geographic_sparse_set_aliases():
    schema = FeatureSchema(
        dense=[("distance", "numeric")],
        sparse_sets=[("ozip_zip5", "zip_sparse_set"), ("dzip_zip5", "h3_sparse_set")],
    )

    payload = schema.to_rust_payload(dense_width=1, sparse_names=["ozip_zip5", "dzip_zip5"])

    assert payload["kinds"] == ["Numeric", "SparseSet", "SparseSet"]

    schema_with_zip3 = FeatureSchema(
        dense=[("distance", "numeric")],
        sparse_sets=[("ozip_zip_p3", "zip3_sparse_set")],
    )
    zip3_payload = schema_with_zip3.to_rust_payload(dense_width=1, sparse_names=["ozip_zip_p3"])
    assert zip3_payload["kinds"] == ["Numeric", "SparseSet"]

    schema_with_zone = FeatureSchema(
        dense=[("distance", "numeric")],
        sparse_sets=[("zone_id", "zone_sparse_set"), ("region_id", "region_sparse_set")],
    )
    zone_payload = schema_with_zone.to_rust_payload(dense_width=1, sparse_names=["zone_id", "region_id"])
    assert zone_payload["kinds"] == ["Numeric", "SparseSet", "SparseSet"]


def test_python_schema_rejects_sparse_length_mismatch():
    model = GeoBoostRegressor(max_depth=0, backend="python")
    schema = FeatureSchema(dense=[("x", "numeric")])

    with pytest.raises(ValueError, match="feature_schema length"):
        model.fit(
            [[0.0], [1.0]],
            [0.0, 1.0],
            sparse_sets={"route_cells": [[7], [8]]},
            feature_schema=schema,
        )


def test_schema_survives_save_load(tmp_path: Path):
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [4.0, 4.0, -2.0, -2.0]
    schema = FeatureSchema(
        dense=[("bias", "numeric")],
        sparse_sets=[("route_cells", "sparse_set")],
    )
    sparse_sets = {"route_cells": [[5], [5, 9], [2], []]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
        backend="rust",
    )
    _fit_or_skip(model, x, y, sparse_sets=sparse_sets, feature_schema=schema)
    path = tmp_path / "schema.json"

    model.save(path)
    loaded = GeoBoostRegressor.load(path)

    assert loaded.feature_schema_ == model.feature_schema_
    assert loaded.predict(x, sparse_sets=sparse_sets) == pytest.approx(
        model.predict(x, sparse_sets=sparse_sets)
    )
