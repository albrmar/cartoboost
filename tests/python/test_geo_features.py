import pytest

from geoboost import build_zip_sparse_sets


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
