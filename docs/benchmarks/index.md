# Benchmarks

CartoBoost benchmark docs are for reproducible model comparison. Use them to
answer data-science questions such as whether temporal-spatial splitters improve
random, spatial, or temporal holdouts against an axis-only CartoBoost model,
XGBoost, LightGBM, or a mean baseline.

| Page | Question |
| --- | --- |
| [NYC Taxi Benchmarks](nyc-taxi.md) | Does CartoBoost help on real temporal-spatial taxi tasks with random and spatial holdouts? |
| [Lane-Level Acceptance](lane-level.md) | Does the model capture route-cell, temporal, spatial, and combined lane behavior on a controlled dataset? |

## Evaluation Helpers

Objective, calibration, spatial-diagnostic, and blocked-validation helpers are
available from the Python package:

```python
from cartoboost import (
    out_of_time_split,
    residual_morans_i,
    spatial_blocked_cv,
    temporal_blocked_cv,
)
```

Use `out_of_time_split` for the latest-period holdout, then compare that score
with random and spatial holdouts. Temporal-spatial models should improve where
the deployment split is hardest, not only on random validation rows.

Prediction speed should be reported only with the benchmark command, data size,
model settings, and comparison baseline.

## Quick Commands

Run dependency-light NYC smoke validation:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```

Run the maintained NYC comparison target:

```sh
just nyc-quality-benchmark-repeated
```

## Reporting Rules

- State the exact command, sample size, task set, model settings, and feature
  handling.
- Distinguish single-run diagnostics from repeated summaries.
- Do not present synthetic checks or NYC taxi outputs as universal model
  superiority claims.
- Treat synthetic checks as behavior evidence, not broad quality claims.
