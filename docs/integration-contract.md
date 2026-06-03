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

## Priority Contracts

These contracts are the ten highest-priority boundaries for alpha hardening:

1. Rust remains authoritative for training, prediction, serialization, and CLI
   model behavior.
2. Python keeps sklearn-compatible estimator ergonomics while delegating
   advanced behavior to the native backend.
3. `backend="rust"` must fail clearly when `geoboost._native` is unavailable;
   validation jobs must install the extension before using it.
4. `backend="python"` supports only axis splits with constant leaves and must
   reject native-only options explicitly.
5. Artifact version `1` JSON round trips must preserve predictions across Rust,
   Python, and CLI loaders.
6. Split artifacts must retain stable routing semantics for axis, diagonal 2D,
   gaussian/radial 2D, periodic interval, sparse scalar-ID, and fuzzy splits.
7. Fuzzy prediction must conserve branch mass by keeping left and right weights
   normalized for every routed sample.
8. Linear leaves must use weighted ridge fitting and fall back safely only when
   the fitting path cannot produce a valid model.
9. Validation artifacts under `target/validation/` must be generated from the
   checked-in scripts and record splitter and lane-level phase coverage.
10. CI must keep routine lint/unit checks separate from native-extension
    validation so matrix tests and rust-backed artifact generation fail at the
    correct boundary.
