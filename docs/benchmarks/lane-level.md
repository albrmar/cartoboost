# Lane-Level Acceptance

Lane-level acceptance checks exercise combined dense, temporal, spatial, and
sparse route-cell behavior on deterministic synthetic data.

## Command

```sh
uv run --group dev maturin develop
uv run --group dev python scripts/run_lane_level_acceptance_metrics.py
```

`scripts/run_full_validation.py` also runs this check and copies the generated
lane-level directory into `target/validation/`.

## Fixture Shape

- 4 origin regions x 4 destination regions = 16 lanes.
- 24 hourly observations per lane.
- Observable columns: origin x/y, destination x/y, lane ID, hour, midpoint x/y,
  and distance.
- No hidden simulator metadata is passed into training.

## Artifacts

Generated committed artifacts live under `docs/assets/lane_level_tests/`:

- `acceptance_metrics.json`
- `acceptance_metrics.md`
- `route_midpoint_geometry.png`
- `hour_profile.png`
- `lane_heatmap.png`

Treat these files as evidence artifacts. Refresh them only when the lane-level
fixture, model behavior, or acceptance gates intentionally change.

## What The Check Proves

The lane-level fixture is intended to catch regressions in:

- Route-cell sparse-set encoding.
- Temporal profile behavior.
- Spatial route geometry behavior.
- Combined split behavior when several feature families are present.

It is not a production quality benchmark. Any broader benchmark claim needs a
documented dataset, feature handling, baseline configuration, and repeated-run
summary.
