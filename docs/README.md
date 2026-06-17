# CartoBoost Documentation

This directory contains the user documentation for fitting, evaluating,
explaining, and saving CartoBoost regression models for temporal-spatial data.

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
- [SHAP Support](shap.md)
- [CLI](user-guide/cli.md)
- [Model Artifacts](model_artifact.md)
- [Limitations](limitations.md)
- [Benchmarks](benchmarks/index.md)

CartoBoost is most useful when time, place, and route membership are more than
ordinary scalar columns: periodic hours, spatial boundaries, local hotspots,
route cells, and fuzzy boundary behavior can be modeled directly through the
estimator parameters.

## Quick Start

```sh
pip install cartoboost
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
