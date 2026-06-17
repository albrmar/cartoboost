# CartoBoost

[![PyPI](https://img.shields.io/pypi/v/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![Python](https://img.shields.io/pypi/pyversions/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![CI](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml)
[![Docs](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml)
[![Publish](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CartoBoost is a Rust-backed Python regressor for temporal-spatial problems:
demand by zone and time, route or lane performance, delivery ETA residuals,
pickup/dropoff pricing effects, and other targets where place, time, and sparse
location memberships matter.

CartoBoost is useful when a plain tabular booster needs a lot of manual feature
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

Install the released Python package from PyPI:

```sh
uv add cartoboost
```

The PyPI release includes prebuilt Rust extension wheels for CPython 3.10-3.15
on Linux, macOS, and Windows. For optional integrations:

```sh
uv add "cartoboost[explain]"  # SHAP support
uv add "cartoboost[optuna]"   # Optuna tuning
uv add "cartoboost[polars]"   # Polars inputs
uv add "cartoboost[onnx]"     # ONNX export subset
```

Verify the install:

```sh
python -c "import cartoboost; print(cartoboost.__version__)"
cartoboost --help
```

From a source checkout, use the development environment:

```sh
uv sync --group dev
uv run --group dev maturin develop
```

`maturin develop` builds `cartoboost._native` into the local `uv` environment.

To tune CartoBoost with Optuna, install the optional dependency:

```sh
uv add "cartoboost[optuna]"
```

## Basic Regression

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
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

Optuna works with the estimator through the standard sklearn contract:

```python
import optuna
from sklearn.model_selection import cross_val_score

from cartoboost import CartoBoostRegressor


def objective(trial):
  model = CartoBoostRegressor(
    n_estimators=trial.suggest_int("n_estimators", 50, 300),
    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
    max_depth=trial.suggest_int("max_depth", 1, 6),
    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 32),
  )
  scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=3,
    scoring="neg_mean_squared_error",
  )
  return float(-scores.mean())


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
best_model = CartoBoostRegressor(**study.best_params).fit(X_train, y_train)
```

## Temporal-Spatial Example

Use dense columns for numeric location and time features, and sparse-set columns
for memberships such as route cells or zones.

```python
from cartoboost import CartoBoostRegressor

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

model = CartoBoostRegressor(
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
model.save("model.cartoboost.json")
loaded = CartoBoostRegressor.load("model.cartoboost.json")

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
cartoboost train --data train.csv --config configs/regression.toml --model-out model.json
cartoboost predict --model model.json --input test.csv --predictions-out predictions.csv
cartoboost eval --model model.json --data test_with_target.csv
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
