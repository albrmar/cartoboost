import sys
import types

import pytest
from cartoboost.s2 import (
    build_s2_sparse_sets,
    encode_s2_cells,
    latlng_to_s2_id,
    normalize_s2_id,
    s2_parent_id,
)


class _FakeLatLng:
    def __init__(self, lat: float, lng: float) -> None:
        self.lat = lat
        self.lng = lng

    @classmethod
    def from_degrees(cls, lat: float, lng: float) -> "_FakeLatLng":
        return cls(lat, lng)


class _FakeCellId:
    def __init__(self, value: int, level: int = 30) -> None:
        self._value = value
        self._level = level

    @classmethod
    def from_lat_lng(cls, lat_lng: _FakeLatLng) -> "_FakeCellId":
        value = 10_000_000 + int(lat_lng.lat * 1000) * 1000 + int(abs(lat_lng.lng) * 10)
        return cls(value)

    def parent(self, level: int) -> "_FakeCellId":
        return _FakeCellId((self._value // 100) * 100 + level, level=level)

    def id(self) -> int:
        return self._value

    def level(self) -> int:
        return self._level


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, 0),
        (12345, 12345),
        ("12345", 12345),
        ("0x2a", 42),
    ],
)
def test_normalize_s2_id_accepts_int_decimal_and_prefixed_hex(raw, expected):
    assert normalize_s2_id(raw) == expected


@pytest.mark.parametrize("raw", [-1, "-1", "", "not-s2", 1.25, True])
def test_normalize_s2_id_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="S2 IDs"):
        normalize_s2_id(raw)


def test_s2_auto_encoding_uses_optional_s2sphere_package(monkeypatch):
    fake_s2sphere = types.SimpleNamespace(LatLng=_FakeLatLng, CellId=_FakeCellId)
    monkeypatch.setitem(sys.modules, "s2sphere", fake_s2sphere)

    child = latlng_to_s2_id(40.7, -73.9, level=12)

    assert child == 50_700_712
    assert s2_parent_id(child, parent_level=8) == 50_700_708
    assert encode_s2_cells([40.7], [-73.9], level=12) == [child]
    assert build_s2_sparse_sets(
        {"pickup_s2": ([40.7, 40.8], [-73.9, -74.0])},
        level=12,
        parent_levels=[8],
    ) == {
        "pickup_s2": [
            sorted({50_700_712, 50_700_708}),
            sorted({50_800_712, 50_800_708}),
        ]
    }


def test_s2_auto_encoding_hard_fails_without_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "s2sphere", None)

    with pytest.raises(ImportError, match="optional 's2sphere' package"):
        latlng_to_s2_id(40.7, -73.9, level=12)


def test_build_s2_sparse_sets_validates_coordinate_rows(monkeypatch):
    fake_s2sphere = types.SimpleNamespace(LatLng=_FakeLatLng, CellId=_FakeCellId)
    monkeypatch.setitem(sys.modules, "s2sphere", fake_s2sphere)

    with pytest.raises(ValueError, match="same number of rows"):
        build_s2_sparse_sets({"pickup_s2": ([40.7], [-73.9, -74.0])}, level=12)
