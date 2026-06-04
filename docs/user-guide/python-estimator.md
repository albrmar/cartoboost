# Python Estimator

`GeoBoostRegressor` is the public Python estimator. It follows sklearn
conventions for parameter inspection, cloning, pipelines, and grid search over
the supported API surface.

## Basic Usage

```python
from geoboost import GeoBoostRegressor

model = GeoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    splitters=["axis"],
    backend="auto",
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

`predict` returns a NumPy array.

## Temporal-Spatial Usage

GeoBoost's native backend can use splitters that match common time and location
patterns:

```python
model = GeoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24", "sparse_set"],
    fuzzy=True,
    fuzzy_bandwidth=0.05,
    backend="rust",
)
```

Use dense columns for coordinates, projected x/y values, distances, and
periodic time features. Use `sparse_sets=` for route cells, zones, grid cells,
or encoded H3 cells when a row can belong to multiple locations.

## Backend Selection

| Backend | Behavior |
| --- | --- |
| `rust` | Requires `geoboost._native`; raises `ImportError` if unavailable. |
| `auto` | Uses Rust when available and falls back only for supported dense axis configurations. |
| `python` | Uses the fallback directly. |

Use `backend="rust"` for any model that depends on native behavior. The
fallback is limited to dense axis splits with constant leaves.

## Sparse-Set Features

Sparse-set columns are passed separately from dense features:

```python
route_cells = [[7, 11], [11], [3], []]

model = GeoBoostRegressor(
    n_estimators=2,
    learning_rate=0.5,
    max_depth=1,
    min_samples_leaf=1,
    splitters=["sparse_set"],
    backend="rust",
)
model.fit(X_dense, y, sparse_sets={"route_cells": route_cells})
predictions = model.predict(X_dense, sparse_sets={"route_cells": route_cells})
```

Sparse IDs must be non-negative integers. Duplicate IDs inside a row are
normalized by the Rust dataset layer. Models that contain sparse-list splits
require `sparse_sets=` at prediction time.

## Feature Schema

Feature schemas make dense periodic features and sparse-set columns explicit:

```python
schema = {
    "dense": [
        {"name": "distance_m", "kind": "numeric"},
        {"name": "hour_of_day", "kind": "periodic", "period": 24},
    ],
    "sparse_sets": [
        {"name": "route_cells", "kind": "sparse_set"},
    ],
}

model.fit(
    X_dense,
    y,
    sparse_sets={"route_cells": route_cells},
    feature_schema=schema,
)
```

See [Feature Schema](../feature_schema.md) for validation rules and the Rust
artifact representation.

## Sample Weights

`sample_weight` must have the same length as `y` and contain finite
non-negative values.

```python
model.fit(X_train, y_train, sample_weight=weights)
```

Weights are passed to the Rust trainer and are supported by the Python fallback
for its dense axis path.

## Additive Values And SHAP

`predict_additive_values(X)` returns per-row additive components whose sums
match `predict(X)`. SHAP support is exposed through:

```python
explainer = model.make_shap_explainer(X_background)
explanation = model.explain_shap(X_test, background=X_background)
```

Install `geoboost[explain]` or add the optional SHAP dependency in your local
environment.
Sparse-list models can be explained through the helper by supplying matching
foreground and background sparse sets.

## Save And Load

```python
model.save("model.geoboost.json")
loaded = GeoBoostRegressor.load("model.geoboost.json")

model.save_weights("model.weights.json")
weights_loaded = GeoBoostRegressor.load_weights("model.weights.json")
```

Native JSON artifacts preserve training metadata when available, including
splitters, leaf predictor, fuzzy settings, loss, schema, and sparse-set
requirements.

## Common Errors

| Error | Cause |
| --- | --- |
| `ImportError` | `backend="rust"` was requested but the native extension is unavailable. |
| `NotImplementedError` | The Python fallback was asked to train unsupported native-only behavior. |
| `ValueError` | Invalid parameters, unknown splitters, mismatched row counts, or incompatible sparse/schema inputs. |
| `RuntimeError` | Prediction or save was called before fit. |
