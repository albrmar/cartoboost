# geoboost

GeoBoost is a Rust-backed Python regressor for temporal-spatial problems:
demand by zone and time, route or lane performance, delivery ETA residuals,
pickup/dropoff pricing effects, and other targets where place, time, and sparse
location memberships matter.

GeoBoost is useful when a plain tabular booster needs a lot of manual feature
engineering to see the structure in the data. It can train with:

- Periodic splitters for hour-of-day, weekday, seasonality, or other wraparound
  time features.
- Diagonal 2D and Gaussian/radial splitters for spatial boundaries and local
  neighborhoods.
- Sparse-set splitters for route cells, zones, grid cells, encoded H3 cells, or
  other list-valued location memberships.
- Fuzzy routing for smoother behavior near split boundaries.
- Linear leaves for local residual trends after the tree finds a region.

The Python API follows sklearn conventions, so a data scientist can fit,
evaluate, tune, explain, and save a model without working directly in Rust.

## Install

Development requires stable Rust, Python 3.10 or newer, and `uv` 0.7 or newer.

```sh
uv sync --group dev
uv run --group dev maturin develop
```

`maturin develop` builds the native extension used by the Python estimator.
Training and prediction require that extension.

## Basic Regression

```python
from geoboost import GeoBoostRegressor

model = GeoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    splitters=["axis"],
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The estimator supports sklearn-style `get_params`, `set_params`, `clone`,
`Pipeline`, `GridSearchCV`, and NumPy-array predictions.

## Temporal-Spatial Example

Use dense columns for numeric location and time features, and sparse-set columns
for memberships such as route cells or zones.

```python
from geoboost import GeoBoostRegressor

schema = {
    "dense": [
        {"name": "pickup_x", "kind": "numeric"},
        {"name": "pickup_y", "kind": "numeric"},
        {"name": "hour_of_day", "kind": "periodic", "period": 24},
        {"name": "trip_distance", "kind": "numeric"},
    ],
    "sparse_sets": [
        {"name": "route_cells", "kind": "sparse_set"},
    ],
}

model = GeoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=30,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24", "sparse_set"],
    fuzzy=True,
    fuzzy_bandwidth=0.05,
)

model.fit(
    X_train_dense,
    y_train,
    sparse_sets={"route_cells": route_cells_train},
    feature_schema=schema,
)

predictions = model.predict(
    X_test_dense,
    sparse_sets={"route_cells": route_cells_test},
)
```

Why this helps:

- `periodic:24` can split midnight-adjacent hours as neighbors instead of
  treating `23` and `0` as far apart.
- `diagonal_2d` can learn oblique spatial boundaries that axis-only trees need
  many steps to approximate.
- `gaussian_2d` can isolate radial neighborhoods around a local hotspot.
- `sparse_set` can split on list-valued route or cell membership without
  building a wide one-hot matrix.
- `fuzzy=True` can reduce hard jumps near spatial or temporal split boundaries.

## Save, Load, And Explain

```python
model.save("model.geoboost.json")
loaded = GeoBoostRegressor.load("model.geoboost.json")

explanation = loaded.explain_shap(
    X_test_dense,
    background=X_train_dense,
    sparse_sets={"route_cells": route_cells_test},
    background_sparse_sets={"route_cells": route_cells_train},
)
```

Native artifacts are versioned JSON and include optional metadata, feature
schema, and training configuration fields. See
[Model Artifacts](docs/model_artifact.md) and [SHAP Support](docs/shap.md).

## CLI

The CLI handles dense numeric CSV train, predict, eval, and inspect workflows.
Use the Python API for list-valued sparse route-cell features.

```sh
geoboost train --data train.csv --config configs/regression.toml --model-out model.json
geoboost predict --model model.json --input test.csv --predictions-out predictions.csv
geoboost eval --model model.json --data test_with_target.csv
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Python Estimator](docs/user-guide/python-estimator.md)
- [Parameters](docs/user-guide/parameters.md)
- [Spatial Modeling](docs/spatial_modeling.md)
- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Feature Schema](docs/feature_schema.md)
- [Sparse Features](docs/sparse_features.md)
- [Model Artifacts](docs/model_artifact.md)
- [Limitations](docs/limitations.md)
