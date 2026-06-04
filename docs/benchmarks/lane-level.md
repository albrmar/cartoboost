# Lane-Level Acceptance

Lane-level checks exercise combined dense, temporal, spatial, and sparse
route-cell behavior on deterministic synthetic data.

## Command

```sh
uv run --group dev maturin develop
uv run --group dev python scripts/run_lane_level_acceptance_metrics.py
```

## Dataset Shape

- 4 origin regions x 4 destination regions = 16 lanes.
- 24 hourly observations per lane.
- Observable columns: origin x/y, destination x/y, lane ID, hour, midpoint x/y,
  and distance.
- No hidden simulator metadata is passed into training.

## Outputs

Generated outputs live under `docs/assets/lane_level_tests/`:

- `acceptance_metrics.json`
- `acceptance_metrics.md`
- `route_midpoint_geometry.png`
- `hour_profile.png`
- `lane_heatmap.png`

Use these files to inspect route geometry, hour effects, lane-level residuals,
and combined split behavior.

## What The Check Proves

The lane-level dataset is intended to show whether GeoBoost captures:

- Route-cell sparse-set encoding.
- Temporal profile behavior.
- Spatial route geometry behavior.
- Combined split behavior when several feature families are present.

It is not a production quality benchmark. Any broader benchmark claim needs a
documented dataset, feature handling, baseline configuration, and repeated-run
summary.
