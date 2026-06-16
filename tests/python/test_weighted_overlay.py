from __future__ import annotations

from cartoboost import weighted_overlay


def test_weighted_overlay_adds_distance_decay_debug_output(
    neighborhood_points,
    delivery_zones,
) -> None:
    result = weighted_overlay(
        points=neighborhood_points,
        zones=delivery_zones,
        weights={"base_score": 1.0},
        origin=(10.0, 10.0),
        kernel="gaussian",
        bandwidth_meters=1000.0,
        distance_alpha=0.5,
        include_debug=True,
    )

    assert result["config"]["distance_term"] == {
        "enabled": True,
        "source": "origin",
        "kernel": "gaussian",
        "bandwidth_meters": 1000.0,
        "distance_alpha": 0.5,
    }

    first = result["features"][0]
    assert first["id"] == "site-alpha"
    assert first["boost_score"] == 1.875
    assert first["debug"] == {
        "linear": 1.0,
        "priority": 1.25,
        "spatial_term": 0.5,
    }

    assert result["features"][-1]["id"] == "site-charlie"
    assert result["features"][-1]["debug"]["spatial_term"] == 0.0
