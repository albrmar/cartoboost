# CartoBoost Documentation

[![PyPI](https://img.shields.io/pypi/v/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![Python](https://img.shields.io/pypi/pyversions/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![CI](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml)
[![Docs](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml)
[![Publish](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TheCulliganMan/CartoBoost/blob/main/LICENSE)

CartoBoost is a Python regression toolkit for temporal, spatial, geotemporal,
and graph-derived prediction problems. It is built for teams that want a
familiar estimator workflow while giving the model explicit structure for place,
time, sparse route membership, source-target directionality, and learned graph
context.

## Start Here

- [Installation](installation.md): PyPI installs, optional extras, source
  development, and troubleshooting.
- [Getting Started](getting-started.md): train a first model, use neural
  embeddings, save artifacts, and run local checks.
- [Python Estimator](user-guide/python-estimator.md): sklearn-style fit,
  predict, save, load, and explanation workflow.
- [Parameters](user-guide/parameters.md): estimator controls and supported
  splitters.
- [Spatial Modeling](spatial_modeling.md): coordinate features, taxi-zone
  sparse sets, fuzzy routing, and blocked evaluation.
- [Graph Models And Features](graph-features.md): standalone graph regressors,
  standalone link predictors, Node2Vec, GraphSAGE, HeteroGraphSAGE, HinSAGE,
  directed source-target features, metapaths, and graph feature bundles.
- [Neural Embedding Models And Features](neural-features.md): standalone ID
  embedding regression, neural artifacts, fallback behavior, and optional
  feature-generation workflows.
- [Evaluation Protocol](evaluation_protocol.md): out-of-time, spatial-blocked,
  grouped, and leakage-aware validation.
- [Benchmarks](benchmarks/index.md): reproducible comparison reports and
  acceptance metrics.

## What CartoBoost Supports

- L2 and quantile regression.
- Constant and linear residual leaves.
- Axis, histogram-axis, diagonal 2D, Gaussian/radial 2D, periodic, sparse-set,
  and fuzzy split behavior.
- Dense numeric arrays plus list-valued sparse-set columns in Python.
- Feature schemas for numeric, periodic, sparse-set, and contract validation.
- Versioned JSON model and weights artifacts.
- Optional SHAP explanations, Optuna tuning, Polars input support, and ONNX
  export for the supported dense axis-tree subset.
- Neural embedding features.
- node2vec, GraphSAGE, heterogeneous GraphSAGE, and typed-schema HinSAGE graph
  encoders plus standalone graph regressors and link predictors.

## Why It Helps Temporal-Spatial Models

Standard tabular boosters are strong baselines, but they usually see location,
time, and graph relationships as ordinary scalar columns unless you pre-engineer
the structure. CartoBoost adds primitives that match common temporal-spatial and
geotemporal patterns:

- Periodic splitters keep wraparound time features, such as hour `23` and hour
  `0`, adjacent.
- Diagonal 2D splitters model oblique spatial boundaries more directly than
  axis-only trees.
- Gaussian/radial splitters isolate local hotspots, depots, zones, or corridors.
- Sparse-set splitters consume taxi-zone and zone memberships without a wide
  one-hot matrix.
- Fuzzy routing softens hard boundaries where nearby locations or times should
  behave similarly.
- Directional graph features preserve pickup/dropoff semantics such as
  `pickup_zone -> dropoff_zone`, pickup-hour demand, and reverse-trip contrast.

## Install

```sh
uv add cartoboost
```

Optional extras are available for SHAP, Optuna, Polars, and ONNX:

```sh
uv add "cartoboost[explain,optuna,polars,onnx]"
```

## Typical Workflow

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    splitters=["axis", "periodic:24"],
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
model.save("model.cartoboost.json")
```

Use CartoBoost like other gradient-boosting regressors: choose features, split
the data correctly, fit the estimator, compare against baselines, inspect
residuals, and save the fitted artifact.
