import pytest
from geoboost import GeoBoostRegressor


def test_unknown_splitter_is_rejected_before_backend_selection():
    model = GeoBoostRegressor(splitters=["axis", "not_a_splitter"], backend="auto")

    with pytest.raises(ValueError, match="unknown splitter"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


def test_splitters_must_be_a_sequence_of_names():
    model = GeoBoostRegressor(splitters="axis", backend="auto")

    with pytest.raises(ValueError, match="splitters must be a list"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


def test_feature_schema_metadata_is_retained(tmp_path):
    schema = {"distance": {"role": "numeric"}}
    model = GeoBoostRegressor(max_depth=0, backend="python")

    model.fit([[0.0], [1.0]], [2.0, 4.0], feature_schema=schema)
    model_path = tmp_path / "schema.json"
    model.save(model_path)
    restored = GeoBoostRegressor.load(model_path)

    assert model.feature_schema_ == schema
    assert restored.feature_schema_ == schema
