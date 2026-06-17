import pytest
from cartoboost import build_geo_sparse_sets, build_zip_sparse_sets, coerce_geo_to_feature_id


def test_build_zip_sparse_sets_adds_geo_context_columns():
    features = build_zip_sparse_sets(
        origin_zip=["94103", "10001", None],
        destination_zip=["94103", "94107", "94107"],
        parent_prefixes=(3, 2),
        include_match_indicator=True,
    )

    assert features["ozip_zip5"] == [[94103], [10001], []]
    assert features["ozip_zip_p3"] == [[941], [100], []]
    assert features["ozip_zip_p2"] == [[94], [10], []]
    assert features["dzip_zip5"] == [[94103], [94107], [94107]]
    assert features["dzip_zip_p3"] == [[941], [941], [941]]
    assert features["dzip_zip_p2"] == [[94], [94], [94]]
    assert features["zip_match"] == [[1], [], []]


def test_build_zip_sparse_sets_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same number of rows"):
        build_zip_sparse_sets(origin_zip=["94103"], destination_zip=["10001", "94107"])


def test_build_zip_sparse_sets_zip3_only():
    features = build_zip_sparse_sets(
        origin_zip=["94103", "10001", None],
        destination_zip=["94103", "94107", "94107"],
        parent_prefixes=(5, 3, 2),
        include_raw=False,
        include_match_indicator=False,
        zip3_only=True,
    )

    assert "ozip_zip5" not in features
    assert list(features.keys()) == ["ozip_zip_p3", "dzip_zip_p3"]
    assert features["ozip_zip_p3"] == [[941], [100], []]
    assert features["dzip_zip_p3"] == [[941], [941], [941]]


def test_build_geo_sparse_sets_handles_zones():
    features = build_geo_sparse_sets(
        {
            "pickup_zone": ["Z1", "Z2", "Z1", None],
            "delivery_zone": ["Z2", "Z2", 10003, ""],
        },
        namespace="md",
    )

    assert features["pickup_zone"][0] == [
        coerce_geo_to_feature_id("Z1", namespace="md:pickup_zone")
    ]
    assert features["pickup_zone"][3] == []
    assert features["delivery_zone"][0] == [
        coerce_geo_to_feature_id("Z2", namespace="md:delivery_zone")
    ]
    assert features["delivery_zone"][2] == [10003]


def test_build_geo_sparse_sets_rejects_column_length_mismatch():
    with pytest.raises(ValueError, match="expected 3"):
        build_geo_sparse_sets(
            {"pickup_zone": ["Z1", "Z2", "Z3"], "delivery_zone": ["Z2"]},
        )


def test_build_geo_sparse_sets_rejects_empty_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        build_geo_sparse_sets({})


def test_coerce_geo_to_feature_id_is_deterministic():
    first = coerce_geo_to_feature_id("Z1", namespace="market")
    second = coerce_geo_to_feature_id("Z1", namespace="market")
    assert first is not None and second is not None
    assert first == second
