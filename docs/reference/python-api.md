# Python API Reference

This is the hand-maintained public reference for the current Python surface.
Keep it synchronized with `python/geoboost`.

## `geoboost.GeoBoostRegressor`

```python
GeoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    min_gain=1e-8,
    loss="l2",
    quantile_alpha=0.5,
    splitters=None,
    leaf_predictor="constant",
    linear_leaf_features=None,
    fuzzy=False,
    fuzzy_bandwidth=0.0,
    l2_regularization=1.0,
    random_state=None,
    n_threads=None,
    backend="auto",
    monotonic_constraints=None,
)
```

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, sample_weight=None, feature_schema=None, sparse_sets=None, eval_set=None)` | `self` | `eval_set` is accepted but currently ignored. |
| `predict(X, sparse_sets=None)` | `numpy.ndarray` | Requires matching dense width and sparse columns. |
| `predict_additive_values(X, sparse_sets=None)` | `numpy.ndarray` | Row sums equal `predict(X)`. |
| `make_shap_explainer(background, **kwargs)` | SHAP explainer | Requires optional SHAP dependency. |
| `explain_shap(X, background=..., **kwargs)` | `shap.Explanation` | Convenience SHAP entry point. |
| `save(path)` | `None` | Writes a model artifact. |
| `save_weights(path, format="auto")` | `None` | Writes JSON weights or supported ONNX. |
| `GeoBoostRegressor.load(path)` | estimator | Loads native or fallback model artifacts. |
| `GeoBoostRegressor.load_weights(path)` | estimator | Loads weights artifacts. |
| `get_params(deep=True)` | `dict` | sklearn-compatible parameter inspection. |
| `set_params(**params)` | `self` | Validates known parameter names. |

## `geoboost.FeatureSchema`

```python
FeatureSchema(dense, sparse_sets=None)
```

Helper for converting Python schema entries to Rust feature metadata.

| Method | Returns | Notes |
| --- | --- | --- |
| `to_dict()` | `dict` | Compact Python representation. |
| `to_rust_payload(dense_width, sparse_names)` | `dict` | Validated `names` and `kinds` payload. |
| `to_json(dense_width, sparse_names)` | `str` | JSON-encoded Rust payload. |

## SHAP Helpers

```python
geoboost.make_shap_explainer(model, background, **kwargs)
geoboost.explain_shap(model, X, background=..., **kwargs)
```

These functions are also available as estimator methods. See
[SHAP Support](../shap.md).

## I/O Helpers

```python
geoboost.io.read_geojson(path)
```

Reads a GeoJSON file into a Python dictionary.
