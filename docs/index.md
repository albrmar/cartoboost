# CartoBoost Documentation

[![PyPI](https://img.shields.io/pypi/v/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![Python](https://img.shields.io/pypi/pyversions/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![CI](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml)
[![Docs](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml)
[![Publish](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TheCulliganMan/CartoBoost/blob/main/LICENSE)

CartoBoost is a Python regression package for temporal-spatial problems: demand
by zone and time, route or lane performance, ETA residuals, local pricing
effects, and other targets where place, time, and sparse location memberships
carry signal. It is designed for data scientists who want an estimator workflow
that feels familiar from scikit-learn, XGBoost, and LightGBM while exposing
split types that directly model temporal and spatial structure.

## Start Here

- [Installation](installation.md) covers PyPI installs, extras, verification,
  and source-checkout development.
- [Getting Started](getting-started.md) trains a first model from an installed
  package.
- [Python Estimator](user-guide/python-estimator.md) shows the sklearn-style
  fit, predict, save, load, and explanation workflow.
- [Parameters](user-guide/parameters.md) lists the model controls and supported
  splitters.
- [Objectives](objectives.md) covers L2 and quantile regression.
- [Spatial Modeling](spatial_modeling.md) explains coordinate features,
  route-cell sparse sets, fuzzy routing, and blocked evaluation.
- [CLI](user-guide/cli.md) covers dense numeric CSV training and prediction.
- [Benchmarks](benchmarks/index.md) explains reproducible comparison reports.

## What CartoBoost Supports

- L2 and quantile regression.
- Constant and linear residual leaves.
- Axis, histogram-axis, diagonal 2D, Gaussian/radial 2D, periodic, sparse-set,
  and fuzzy split behavior.
- Dense numeric arrays plus list-valued sparse-set columns in Python.
- Feature schemas for numeric, periodic, and sparse-set declarations.
- Versioned JSON model and weights artifacts, with optional ONNX export for a
  dense axis-tree subset.
- sklearn-compatible estimator workflows including `Pipeline`, `GridSearchCV`,
  `get_params`, `set_params`, and `clone`.

## Why It Helps Temporal-Spatial Models

Standard tabular boosters are strong baselines, but they usually see location
and time as ordinary scalar columns unless you pre-engineer richer features.
CartoBoost adds splitters that match common temporal-spatial patterns:

- Periodic splitters keep wraparound time features, such as hour `23` and hour
  `0`, adjacent.
- Diagonal 2D splitters model oblique spatial boundaries more directly than
  axis-only trees.
- Gaussian/radial splitters isolate local hotspots, depots, zones, or corridors.
- Sparse-set splitters consume route-cell and zone memberships without a wide
  one-hot matrix.
- Fuzzy routing softens hard boundaries where nearby locations or times should
  behave similarly.

The PyPI package ships the Rust native extension required for training and
prediction.

## Install

```sh
pip install cartoboost
```

Optional extras are available for SHAP, Optuna, Polars, and ONNX:

```sh
pip install "cartoboost[explain,optuna,polars,onnx]"
```

## Typical Workflow

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
model.save("model.cartoboost.json")
```

Use CartoBoost like other gradient-boosting regressors: choose features, split the
data, fit the estimator, compare metrics against baselines, inspect residuals,
and save the fitted model artifact.
