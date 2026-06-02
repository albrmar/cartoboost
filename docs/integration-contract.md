# Integration Contract

The initial pytest integration scaffold expects a Python package named
`geoboost`. Tests skip until the package is present, but the contract below
should guide implementation work.

## `geoboost.weighted_overlay`

Expected call shape:

```python
geoboost.weighted_overlay(
    points=point_feature_collection,
    zones=zone_feature_collection,
    weights={"base_score": 0.5, "traffic_index": 0.3, "population_index": 0.2},
    zone_priority_multiplier=True,
    precision=6,
)
```

Expected return shape:

```json
{
  "schema_version": 1,
  "scenario": "neighborhood_points_x_delivery_zones",
  "config": {
    "algorithm": "weighted_overlay",
    "weights": {
      "base_score": 0.5,
      "traffic_index": 0.3,
      "population_index": 0.2
    },
    "zone_priority_multiplier": true,
    "rounding": {
      "places": 6,
      "mode": "half_even"
    }
  },
  "features": []
}
```

Each feature result must include:

- `id`, copied from the input point feature id
- `zone_id`, copied from the containing zone feature id
- `boost_score`, rounded according to the request
- `rank`, one-based descending score rank

## `geoboost.io.read_geojson`

Expected behavior:

- Accept a `str` or `pathlib.Path`.
- Decode UTF-8 JSON.
- Return the parsed GeoJSON mapping without mutating coordinates or properties.
- Raise a clear exception for malformed JSON or unsupported GeoJSON top-level
  types.
