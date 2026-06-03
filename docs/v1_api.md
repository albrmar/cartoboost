# v1 API

GeoBoost v1 is a regression-only API with a Rust native backend and a limited
pure-Python fallback.

## Python Estimator

```python
from geoboost import GeoBoostRegressor

model = GeoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    min_gain=1e-8,
    loss="l2",
    splitters=["axis"],
    leaf_predictor="constant",
    fuzzy=False,
    backend="auto",
)
```

Supported public methods:

- `fit(X, y, sample_weight=None, feature_schema=None, sparse_sets=None)`
- `predict(X, sparse_sets=None)`
- `save(path)`
- `GeoBoostRegressor.load(path)`
- `get_params(deep=True)`
- `set_params(**params)`

## Backends

- `backend="rust"` requires the PyO3 extension and fails if it is unavailable.
- `backend="auto"` uses Rust when available and falls back only for supported
  dense axis-split constant-leaf configurations.
- `backend="python"` uses the fallback directly.

The Rust backend is required for diagonal, Gaussian/radial, periodic, sparse,
fuzzy, linear-leaf, schema-driven, and mixed dense+sparse workflows.

## Splitters

Accepted splitter names include:

- `axis`
- `diagonal_2d` or `diagonal2d`
- `gaussian_2d`, `gaussian2d`, or `radial`
- `periodic_time`, `periodic_24`, or `periodic:<period>`
- `sparse_set` or `sparse`

Unknown splitter names raise `ValueError`; they do not silently fall back to
axis-only training.

## Leaf Predictors

- `leaf_predictor="constant"` fits weighted mean residual leaves.
- `leaf_predictor="linear"` fits weighted ridge residual leaves on configured
  feature indices.

`linear_leaf_features` currently expects stringified integer feature indices in
the Python API.

## Sample Weights

`sample_weight` must match `y` length and contain finite non-negative values.
Weights are passed through to the Rust backend and used by the fallback for its
supported dense path.

## Sparse Features And Schema

List-valued sparse features are provided through `sparse_sets=`.
Feature schema is provided as a dictionary and converted to Rust schema metadata
for native training. See [Sparse Features](sparse_features.md) and
[Feature Schema](feature_schema.md).

## Error Policy

Python public APIs raise:

- `ValueError` for invalid input, unsupported names, or mismatched dimensions.
- `ImportError` when `backend="rust"` is requested but the native extension is
  unavailable.
- `NotImplementedError` when the pure-Python fallback is asked to train behavior
  outside its limited dense axis-split scope.
- `RuntimeError` when predicting or saving before fit.

## CLI Scope

The CLI v1 contract is dense numeric CSV train/predict/eval/inspect. Mixed
sparse row input is a Python API feature in v1.
