# Temporal-Spatial Modeling

CartoBoost is built for regression problems where time, place, route membership,
or local neighborhoods drive the target. Examples include pickup demand by hour
and zone, delivery ETA residuals by lane, fare or cost adjustments by location,
and operational metrics grouped by route cells.

## Why Use CartoBoost

XGBoost and LightGBM are excellent tabular baselines. CartoBoost is useful when
the feature engineering needed for those models starts to hide the structure of
the problem:

- Wraparound time needs sine/cosine, buckets, or custom encodings.
- Spatial boundaries need many axis-aligned splits or hand-built region
  features.
- Local hotspots need precomputed distance-to-center features.
- Route or cell memberships can become very wide one-hot matrices.
- Hard split boundaries can produce abrupt predictions for nearby points.

CartoBoost gives those structures direct model controls through periodic,
diagonal 2D, Gaussian/radial, sparse-set, and fuzzy split behavior.

## Feature Patterns

| Pattern | CartoBoost feature path | Why it helps |
| --- | --- | --- |
| Hour-of-day, weekday, seasonality | Dense periodic feature with `periodic:<period>` | Preserves wraparound adjacency. |
| Latitude/longitude or projected x/y | Dense numeric features with `diagonal_2d` or `gaussian_2d` | Learns spatial boundaries and neighborhoods without only stair-step axis cuts. |
| Route cells, zones, encoded H3 cells | `sparse_sets={...}` with `splitters=["sparse_set"]` | Uses list-valued memberships directly. |
| Smooth transitions near a boundary | `fuzzy=True` with `fuzzy_bandwidth` and optional `fuzzy_kernel` | Routes samples fractionally instead of forcing a hard left/right decision. |
| Local trend inside a region | `leaf_predictor="linear"` | Fits a ridge residual model inside leaves. |

## Example

```python
from cartoboost import CartoBoostRegressor

schema = {
    "dense": [
        {"name": "pickup_x", "kind": "numeric"},
        {"name": "pickup_y", "kind": "numeric"},
        {"name": "hour", "kind": "periodic", "period": 24},
        {"name": "distance_m", "kind": "numeric"},
    ],
    "sparse_sets": [
        {"name": "route_cells", "kind": "sparse_set"},
    ],
}

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=30,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24", "sparse_set"],
    fuzzy=True,
    fuzzy_bandwidth=0.05,
  fuzzy_kernel="tricube",
)

model.fit(
    X_train_dense,
    y_train,
    sparse_sets={"route_cells": route_cells_train},
    feature_schema=schema,
)
```

## Encoded H3 Or Grid Cells

CartoBoost does not compute H3 cells from latitude and longitude. If you already
have H3, S2, grid, zone, or corridor IDs, encode them as non-negative integers
and pass them through a sparse-set column:

```python
model.fit(
    X_dense,
    y,
    sparse_sets={"pickup_h3": [[617700169957507071], [617700169957507583]]},
)
```

A sparse split routes left when a row contains one of the learned IDs and right
otherwise. Empty rows and unseen IDs route as no match.

## Evaluation

Temporal-spatial models should be tested with the kind of holdout they will
face in use:

- Use random holdouts to measure general regression quality.
- Use spatial holdouts to test new zones, cells, routes, or corridors.
- Use [out-of-time validation](evaluation_protocol.md#out-of-time-validation)
  to test later periods.
- Report residuals by zone, route cell, hour, or lane to find localized failure
  modes.
- Compare against axis-only CartoBoost, XGBoost, or LightGBM baselines with the
  same train/test split and feature set.

CartoBoost can be better suited than a generic tabular booster when these
temporal-spatial holdouts improve because the model can express the real
structure with fewer ad hoc preprocessing steps. Keep claims tied to your data,
features, split strategy, and metrics.
