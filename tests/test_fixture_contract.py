"""Validate committed GeoBoost fixtures and golden files.

These tests intentionally exercise only the static contract data. They give
implementation workers immediate feedback when fixture updates drift from the
documented GeoBoost test schema.
"""

from __future__ import annotations

from typing import Any


def _assert_position(position: Any) -> None:
    assert isinstance(position, list)
    assert len(position) == 2
    x, y = position
    assert isinstance(x, (int, float))
    assert isinstance(y, (int, float))
    assert 0 <= x <= 40
    assert 0 <= y <= 40


def test_neighborhood_points_fixture_shape(neighborhood_points: dict[str, Any]) -> None:
    assert neighborhood_points["type"] == "FeatureCollection"
    assert neighborhood_points["name"] == "neighborhood_points"
    assert len(neighborhood_points["features"]) == 3

    seen_ids: set[str] = set()
    for feature in neighborhood_points["features"]:
        assert feature["type"] == "Feature"
        assert feature["id"] not in seen_ids
        seen_ids.add(feature["id"])
        assert feature["geometry"]["type"] == "Point"
        _assert_position(feature["geometry"]["coordinates"])

        properties = feature["properties"]
        assert properties["kind"] == "synthetic_site"
        assert 0 <= properties["base_score"] <= 1
        assert 0 <= properties["traffic_index"] <= 1
        assert 0 <= properties["population_index"] <= 1


def test_delivery_zones_fixture_shape(delivery_zones: dict[str, Any]) -> None:
    assert delivery_zones["type"] == "FeatureCollection"
    assert delivery_zones["name"] == "delivery_zones"
    assert len(delivery_zones["features"]) == 2

    for feature in delivery_zones["features"]:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"
        assert feature["properties"]["service_level"] in {"same_day", "standard"}
        assert feature["properties"]["priority"] > 0

        rings = feature["geometry"]["coordinates"]
        assert len(rings) == 1
        exterior = rings[0]
        assert exterior[0] == exterior[-1]
        assert len(exterior) >= 4
        for position in exterior:
            _assert_position(position)


def test_golden_boosts_reference_fixture_ids(
    neighborhood_points: dict[str, Any],
    delivery_zones: dict[str, Any],
    expected_neighborhood_boosts: dict[str, Any],
) -> None:
    point_ids = {feature["id"] for feature in neighborhood_points["features"]}
    zone_ids = {feature["id"] for feature in delivery_zones["features"]}
    result_ids = [feature["id"] for feature in expected_neighborhood_boosts["features"]]
    ranks = [feature["rank"] for feature in expected_neighborhood_boosts["features"]]

    assert expected_neighborhood_boosts["schema_version"] == 1
    assert set(result_ids) == point_ids
    assert ranks == sorted(ranks)

    for feature in expected_neighborhood_boosts["features"]:
        assert feature["id"] in point_ids
        assert feature["zone_id"] in zone_ids
        assert feature["boost_score"] >= 0
        assert isinstance(feature["rank"], int)
