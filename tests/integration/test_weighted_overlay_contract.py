"""Integration contract for the future GeoBoost Python API.

The repository is currently missing an importable Python package. These tests
are written as executable documentation and skip until an implementation
provides the expected module-level API.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

geoboost = pytest.importorskip(
    "geoboost",
    reason="GeoBoost Python package is not available in this scaffold yet",
)


def test_weighted_overlay_matches_neighborhood_golden(
    neighborhood_points: dict[str, Any],
    delivery_zones: dict[str, Any],
    expected_neighborhood_boosts: dict[str, Any],
) -> None:
    engine = getattr(geoboost, "weighted_overlay", None)
    if engine is None:
        pytest.skip("geoboost.weighted_overlay is not implemented yet")

    actual = engine(
        points=neighborhood_points,
        zones=delivery_zones,
        weights=expected_neighborhood_boosts["config"]["weights"],
        zone_priority_multiplier=True,
        precision=expected_neighborhood_boosts["config"]["rounding"]["places"],
    )

    assert actual == expected_neighborhood_boosts


def test_geojson_reader_round_trips_fixture(
    fixtures_dir,
    neighborhood_points: dict[str, Any],
) -> None:
    io_module = importlib.import_module("geoboost.io")
    read_geojson = getattr(io_module, "read_geojson", None)
    if read_geojson is None:
        pytest.skip("geoboost.io.read_geojson is not implemented yet")

    assert read_geojson(fixtures_dir / "neighborhood_points.geojson") == neighborhood_points
