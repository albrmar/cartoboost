# CartoBoost Documentation

This directory contains the user documentation for installing, fitting,
evaluating, explaining, extending, and saving CartoBoost models for temporal,
spatial, geotemporal, and graph-derived regression workflows.

## Contents

- [Getting Started](getting-started.md)
- [Python Estimator](user-guide/python-estimator.md)
- [Parameters](user-guide/parameters.md)
- [Objectives](objectives.md)
- [Constraints](constraints.md)
- [Spatial Modeling](spatial_modeling.md)
- [Evaluation Protocol](evaluation_protocol.md)
- [Feature Schema](feature_schema.md)
- [Sparse Features](sparse_features.md)
- [Graph Features](graph-features.md)
- [Neural Features](neural-features.md)
- [SHAP Support](shap.md)
- [CLI](user-guide/cli.md)
- [Model Artifacts](model_artifact.md)
- [Python API Reference](reference/python-api.md)
- [CLI Reference](reference/cli.md)
- [Limitations](limitations.md)
- [Benchmarks](benchmarks/index.md)
  - [Neural Embedding Strategy Assessment](benchmarks/neural-embedding-strategy.md)

CartoBoost is most useful when time, place, route membership, and directed graph
relationships are more than ordinary scalar columns. Periodic hours, spatial
boundaries, local hotspots, route cells, source-target flows, and fuzzy boundary
behavior can be modeled directly through the estimator and graph-feature
contracts.

## Quick Start

```sh
uv add cartoboost
```

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    splitters=["axis"],
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The CLI is available for dense numeric CSV workflows:

```sh
cartoboost train --data train.csv --config configs/regression.toml --model-out model.json
cartoboost predict --model model.json --input test.csv --predictions-out predictions.csv
```

Use the Python API for sparse-set, neural embedding, and graph-derived feature
pipelines.
