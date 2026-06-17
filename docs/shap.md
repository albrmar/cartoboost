# SHAP Support

CartoBoost supports the Python `shap` package through the estimator prediction
API. Install the optional dependency before using SHAP:

```sh
uv add "cartoboost[explain]"
```

For a source checkout:

```sh
uv sync --extra explain --group dev
```

## Basic Usage

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
).fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
explanation = explainer(X_test)
```

`explanation` is a `shap.Explanation`, so it works with SHAP plotting helpers:

```python
import shap

shap.plots.beeswarm(explanation)
shap.plots.waterfall(explanation[0])
```

CartoBoost also provides convenience helpers that call SHAP for you:

```python
explanation = model.explain_shap(X_test, background=X_train)
explainer = model.make_shap_explainer(X_train)
```

## Additive Weight Decomposition

By default, SHAP decomposes predictions over input features. CartoBoost can also
decompose the fitted additive prediction weights: the initial prediction and one
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

Models trained with `sparse_sets=` can be explained through the CartoBoost helper.
Sparse IDs are exposed to SHAP as binary features named `column=id`.

```python
explanation = model.explain_shap(
    X_test,
    background=X_train,
    sparse_sets={"route_cells": route_cells_test},
    background_sparse_sets={"route_cells": route_cells_train},
)
```

For reusable explainers, pass the background sparse sets when creating the
explainer:

```python
explainer = model.make_shap_explainer(
    X_train,
    sparse_sets={"route_cells": route_cells_train},
)
```

## Additivity

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
- Dense Python, NumPy, and pandas inputs are supported through the existing
  estimator input handling.
- Sparse-set models are supported through the CartoBoost helpers because they need
  the sparse-ID encoding described above.
