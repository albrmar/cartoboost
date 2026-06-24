# SHAP Support

SHAP explanations help audit fitted CartoBoost models after training. For
taxi-trip studies, use them to ask which modeled covariates contributed to a
prediction: distance, hour, pickup/dropoff memberships, graph-derived columns,
neural embedding columns, or fitted tree weights.

SHAP is an explanation layer, not a new model and not proof of causality. It is
most useful after the validation design is fixed, because explanations inherit
the same feature-generation choices, split protocol, and data limitations as
the model being explained.

Install the optional dependency before using SHAP:

```sh
uv add "cartoboost[explain]"
```

For a source checkout:

```sh
uv sync --extra explain
```

## When To Use It

Use SHAP when you need to:

- compare the contribution of route distance, hour, zone memberships, and
  spatial features for individual taxi predictions;
- inspect whether sparse pickup/dropoff IDs dominate a model unexpectedly;
- audit graph or neural feature-generation blocks after they have been appended
  as dense columns;
- verify additive prediction decomposition for debugging and reporting.

Feature names matter. If graph or neural embeddings are appended as generated
columns, SHAP explains those generated columns, not the standalone graph or
neural training process that produced them.

## Basic Usage

```python
import shap
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
explanation = explainer(X_test)
```

`explanation` is a `shap.Explanation`, so it works with SHAP plotting helpers:

```python
shap.plots.beeswarm(explanation)
shap.plots.waterfall(explanation[0])
```

CartoBoost also provides convenience helpers:

```python
explanation = model.explain_shap(X_test, background=X_train)
explainer = model.make_shap_explainer(X_train)
```

## Additive Weight Decomposition

By default, SHAP decomposes predictions over input features. CartoBoost can also
decompose fitted additive prediction weights: the initial prediction and one
component per fitted tree.

```python
explanation = model.explain_shap(
    X_test,
    background=X_train,
    decomposition="weights",
)
```

The explanation feature names are `init_prediction`, `tree_0`, `tree_1`, and so
on. The raw additive matrix is also available directly:

```python
additive = model.predict_additive_values(X_test)
prediction = additive.sum(axis=1)
```

## Sparse Sets

Models trained with `sparse_sets=` can be explained through the CartoBoost
helper. Sparse IDs are exposed to SHAP as binary features named `column=id`.
This makes pickup/dropoff memberships auditable while preserving the model's
sparse-list training contract.

```python
explanation = model.explain_shap(
    X_test,
    background=X_train,
    sparse_sets={"taxi_zones": taxi_zones_test},
    background_sparse_sets={"taxi_zones": taxi_zones_train},
)
```

For reusable explainers, pass the background sparse sets when creating the
explainer:

```python
explainer = model.make_shap_explainer(
    X_train,
    sparse_sets={"taxi_zones": taxi_zones_train},
)
```

## Additivity Check

For regression, SHAP values should add back to the model prediction:

```python
prediction = model.predict(X_test)
reconstructed = explanation.base_values + explanation.values.sum(axis=1)
```

This additivity property is the main sanity check for dense and sparse-set
explanations.

## Current Limits

- CartoBoost estimators are callable after fitting, so `shap.Explainer(model,
  background)` works directly for dense prediction workflows.
- Dense Python, NumPy, and pandas inputs are supported through existing
  estimator input handling.
- Sparse-set models are supported through CartoBoost helpers because they need
  the sparse-ID encoding described above.
- SHAP explains generated graph or neural columns only after those columns are
  part of the model input; standalone graph and neural artifacts have their own
  modeling contracts.
