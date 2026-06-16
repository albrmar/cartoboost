# Python Estimator

`CartoBoostRegressor` is the public Python estimator. It follows sklearn
conventions for parameter inspection, cloning, pipelines, and grid search over
the supported API surface.

Optuna tuning works through the same estimator contract. Install the optional
dependency with `cartoboost[optuna]` or `uv sync --group dev --extra optuna` from
a source checkout, then optimize an objective that constructs a fresh
`CartoBoostRegressor` for each trial.

## Basic Usage

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

`predict` returns a NumPy array.

## Temporal-Spatial Usage

CartoBoost can use splitters that match common time and location patterns:

```python
model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24", "sparse_set"],
    fuzzy=True,
    fuzzy_bandwidth=0.05,
    fuzzy_kernel="gaussian",
)
```

Use dense columns for coordinates, projected x/y values, distances, and
periodic time features. Use `sparse_sets=` for route cells, zones, grid cells,
or encoded H3 cells when a row can belong to multiple locations.

For robust residuals, set `loss="mae"` or `loss="huber"`. For asymmetric
intervals or service-level targets, use `loss="quantile"` with
`quantile_alpha`.

## Native Extension

`CartoBoostRegressor` requires `cartoboost._native`. Build it with
`uv run --group dev maturin develop` from a source checkout. If the extension is
missing, fitting or loading a model raises `ImportError`.

## Sparse-Set Features

Sparse-set columns are passed separately from dense features:

```python
route_cells = [[7, 11], [11], [3], []]

model = CartoBoostRegressor(
    n_estimators=2,
    learning_rate=0.5,
    max_depth=1,
    min_samples_leaf=1,
    splitters=["sparse_set"],
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

Weights are passed to the native trainer.

## Optuna Tuning

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

This pattern works because CartoBoost exposes sklearn-compatible constructor
parameters and cloning behavior. Keep each trial self-contained: create the
estimator inside the objective and fit it only on that trial's data split.

## Additive Values And SHAP

`predict_additive_values(X)` returns per-row additive components whose sums
match `predict(X)`. SHAP support is exposed through:

```python
explainer = model.make_shap_explainer(X_background)
explanation = model.explain_shap(X_test, background=X_background)
```

Install `cartoboost[explain]` or add the optional SHAP dependency in your local
environment.
Sparse-list models can be explained through the helper by supplying matching
foreground and background sparse sets.

## Save And Load

```python
model.save("model.cartoboost.json")
loaded = CartoBoostRegressor.load("model.cartoboost.json")

model.save_weights("model.weights.json")
weights_loaded = CartoBoostRegressor.load_weights("model.weights.json")
```

Native JSON artifacts preserve training metadata when available, including
splitters, leaf predictor, fuzzy settings, loss, schema, and sparse-set
requirements.

## Common Errors

| Error | Cause |
| --- | --- |
| `ImportError` | The native extension is unavailable. |
| `NotImplementedError` | The installed native extension does not support a requested feature. |
| `ValueError` | Invalid parameters, unknown splitters, mismatched row counts, or incompatible sparse/schema inputs. |
| `RuntimeError` | Prediction or save was called before fit. |
