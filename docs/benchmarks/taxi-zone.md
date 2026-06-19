# Taxi Zone Acceptance Benchmark

## What It Tests

The taxi-zone acceptance benchmark checks whether CartoBoost can express
taxi-lane structures before those structures are used in real NYC taxi
benchmarks.

It is not a public model-comparison benchmark.

## Checks

| Contract | Question |
| --- | --- |
| Sparse lane membership | Can pickup/dropoff lane membership be modeled without treating lane IDs as ordinal values? |
| Route midpoint geometry | Can route midpoint and distance features express central, radial, or corridor effects? |
| Periodic hour | Can 23:00 and 01:00 be treated as nearby hours? |
| Combined geotemporal signal | Can lane, route, and hour features work together in one fixture? |

## Reproduce

```sh
uv run python scripts/run_lane_level_acceptance_metrics.py
```

Generated artifacts:

- `docs/assets/lane_level_tests/acceptance_metrics.json`
- `docs/assets/lane_level_tests/acceptance_metrics.md`
- `docs/assets/lane_level_tests/hour_profile.png`
- `docs/assets/lane_level_tests/lane_heatmap.png`
- `docs/assets/lane_level_tests/route_midpoint_geometry.png`

## Passing Means

Passing this benchmark means the implementation can represent the targeted
feature family on a deterministic fixture. It supports engineering confidence
and debugging.

It does not mean CartoBoost is more accurate than another model on real TLC
data. Real-data claims need real TLC data, serious baselines, and
deployment-matched splits.

## If It Fails

If this benchmark fails, do not refresh NYC taxi quality claims until the
feature behavior is repaired or the affected feature family is removed from the
benchmark interpretation.
