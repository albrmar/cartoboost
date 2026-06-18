# Taxi Zone Acceptance

Taxi-zone checks exercise combined dense, temporal, spatial, and sparse pickup
and dropoff behavior on deterministic synthetic data shaped like the NYC taxi
benchmark inputs.

## Command

Install from PyPI for normal use:

```sh
uv add cartoboost
```

From a source checkout, regenerate the committed acceptance artifacts with:

```sh
uv run --group dev python scripts/run_lane_level_acceptance_metrics.py
```

## Dataset Shape

- 4 pickup regions x 4 dropoff regions = 16 pickup-dropoff pairs.
- 24 hourly observations per pickup-dropoff pair.
- Observable columns: pickup x/y, dropoff x/y, pickup-dropoff ID, hour,
  midpoint x/y, and trip distance.
- No hidden simulator metadata is passed into training.

## Outputs

Generated outputs live under `docs/assets/lane_level_tests/`:

- `acceptance_metrics.json`
- `acceptance_metrics.md`
- `route_midpoint_cartometry.png`
- `hour_profile.png`
- `lane_heatmap.png`

Use these files to inspect trip cartometry, hour effects, taxi-zone residuals,
and combined split behavior.

## What The Check Proves

The taxi-zone dataset is intended to show whether CartoBoost captures:

- Pickup/dropoff sparse-set encoding.
- Temporal profile behavior.
- Spatial trip cartometry behavior.
- Combined split behavior when several feature families are present.

It is not a production quality benchmark. Any broader benchmark claim needs a
documented dataset, feature handling, baseline configuration, and repeated-run
summary.
