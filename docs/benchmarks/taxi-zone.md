# Taxi Zone Acceptance Benchmark

## Decision Question

Before using real NYC taxi metrics as evidence, can the implementation express
the taxi-lane structures it claims to model: lane membership, route geometry,
hour-of-day periodicity, and combined geographic-temporal effects?

This is an acceptance test. It is useful for catching feature regressions and
for explaining why a model family is plausible. It is not evidence that
CartoBoost is more accurate on real TLC data.

## Dataset

The fixture is deterministic and shaped like a small taxi-lane dataset:

- 4 pickup regions x 4 dropoff regions.
- 16 pickup/dropoff lanes.
- 24 hourly observations per lane.
- Observable features only: pickup coordinates, dropoff coordinates, lane ID,
  hour, route midpoint, and trip distance.
- No hidden simulator metadata is passed into training.

The target is synthetic lane demand or route intensity. Each phase isolates a
different source of signal so the result can be attributed to a feature family.

## Feature Contracts

| Phase | Feature family | Question |
| --- | --- | --- |
| Sparse lane membership | Pickup/dropoff lane ID as sparse membership. | Can the model identify a hot lane without treating lane ID as an ordinal number? |
| Route midpoint cartometry | Route midpoint and distance geometry. | Can radial/route geometry distinguish central and outer lanes? |
| Wraparound lane hour | Hour-of-day periodicity. | Can 23:00 and 01:00 be treated as close in time? |
| Regional lane boosting | Combined lane, spatial, temporal features. | Does the combined feature set improve holdout RMSE? |

## Reproduce

```sh
uv run python scripts/run_lane_level_acceptance_metrics.py
```

Generated evidence:

- `docs/assets/lane_level_tests/acceptance_metrics.json`
- `docs/assets/lane_level_tests/acceptance_metrics.md`
- `docs/assets/lane_level_tests/hour_profile.png`
- `docs/assets/lane_level_tests/lane_heatmap.png`
- `docs/assets/lane_level_tests/route_midpoint_geometry.png`

## Metrics

Each phase reports an RMSE or inspection margin specific to the signal being
tested. Acceptance gates are binary checks that the targeted feature family
behaves as expected.

## Current Result

The generated acceptance report shows:

- Sparse lane membership reaches exact train RMSE where axis lane ID does not.
- Route midpoint cartometry reaches exact train RMSE where axis midpoint splits
  do not.
- Periodic hour handling treats wraparound hours correctly, while axis hour
  splits create an artificial edge gap.
- The combined lane/spatial/temporal model reduces holdout RMSE from 12.68 to
  4.86 on the deterministic regional lane fixture.

## Scientific Read

This fixture validates feature contracts, not production accuracy. Passing it
means the model can represent structures that appear in pickup/dropoff lane
data. Failing it means a real-data benchmark result would be hard to interpret,
because the implementation could be missing the very feature family being
tested.

Use the NYC taxi benchmark for real-data model choice. Use this page to debug
why a taxi-zone feature family is or is not working.

## Limitations

- The fixture is synthetic and intentionally small.
- Targets are constructed to isolate feature behavior.
- Broader claims require a documented real dataset, baseline configuration,
  repeated split protocol, and quality metrics.
