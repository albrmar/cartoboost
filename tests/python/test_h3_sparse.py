import pytest
from cartoboost import FeatureKind, FeatureSchema
from cartoboost.h3 import (
    expand_h3_sparse_set,
    normalize_h3_id,
    scaffold_h3_parent_id,
)


def test_h3_sparse_schema_serializes_as_rust_sparse_set_and_keeps_metadata():
    entry = {
        "name": "route_h3",
        "kind": FeatureKind.H3_SPARSE_SET,
        "resolution": 9,
        "parent_resolutions": [5, 7],
    }
    schema = FeatureSchema(dense=[("distance_m", FeatureKind.NUMERIC)], sparse_sets=[entry])

    assert schema.to_rust_payload(dense_width=1, sparse_names=["route_h3"]) == {
        "names": ["distance_m", "route_h3"],
        "kinds": ["Numeric", "SparseSet"],
    }
    assert schema.to_dict()["sparse_sets"] == [entry]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, 0),
        (12345, 12345),
        ("12345", 12345),
        ("0x8928308280fffff", int("8928308280fffff", 16)),
        ("8928308280fffff", int("8928308280fffff", 16)),
        ("  8928308280fffff  ", int("8928308280fffff", 16)),
    ],
)
def test_normalize_h3_id_accepts_int_decimal_and_hex_strings(raw, expected):
    assert normalize_h3_id(raw) == expected


@pytest.mark.parametrize("raw", [-1, "-1", "", "not-an-h3", 1.25, True])
def test_normalize_h3_id_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="H3 IDs"):
        normalize_h3_id(raw)


def test_expand_h3_sparse_set_adds_deterministic_scaffold_parents():
    cell = normalize_h3_id("8928308280fffff")
    expected = sorted(
        {
            cell,
            scaffold_h3_parent_id(cell, 9, 5),
            scaffold_h3_parent_id(cell, 9, 7),
        }
    )

    assert (
        expand_h3_sparse_set(
            ["8928308280fffff", cell],
            resolution=9,
            parent_resolutions=[5, 7],
        )
        == expected
    )


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "route_h3", "kind": "h3_sparse_set"},
        {
            "name": "route_h3_legacy",
            "kind": "H3SparseSet",
            "resolution": 16,
            "parent_resolutions": [],
        },
        {
            "name": "route_h3",
            "kind": FeatureKind.H3_SPARSE_SET,
            "resolution": 9,
            "parent_resolutions": [9],
        },
        {
            "name": "route_h3",
            "kind": FeatureKind.H3_SPARSE_SET,
            "resolution": 16,
            "parent_resolutions": [],
        },
    ],
)
def test_h3_sparse_schema_validates_resolution_metadata(entry):
    schema = FeatureSchema(dense=[], sparse_sets=[entry])

    with pytest.raises(ValueError, match="h3_sparse_set"):
        schema.to_rust_payload(dense_width=0, sparse_names=["route_h3"])
