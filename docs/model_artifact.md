# Model Artifact

Native GeoBoost models are stored as versioned JSON artifacts.

## Contents

The artifact includes:

- `artifact_version`
- `initial_prediction`
- `learning_rate`
- `feature_count`
- `target_name`
- `trees`
- optional `metadata`
- optional `feature_schema`
- optional `training_config`

The optional fields make artifacts self-describing while preserving load
compatibility for version `1` artifacts that do not contain newer metadata.

## Save And Load

```python
model.save("model.geoboost.json")
loaded = GeoBoostRegressor.load("model.geoboost.json")
```

Native load restores public estimator parameters when training metadata is
present, including splitters, leaf predictor, linear leaf features, fuzzy
settings, regularization, learning rate, depth, and minimum split controls.

## Prediction Identity

Save/load tests should assert prediction identity with strict tolerance:

```text
atol <= 1e-12
```

Exact equality is preferred when the path is deterministic and uses identical
floating-point operations.

## Version Policy

Artifact version `1` is the current supported native artifact version. Unknown
future versions should fail clearly. Backward-compatible optional fields are
allowed; multi-version migrations are future hardening.

## Dense And Sparse Prediction Safety

Models with sparse-list splits require dataset-aware prediction. Python exposes
that through `predict(X, sparse_sets=...)`. Dense-only prediction on a model that
contains sparse-list splits should raise a clear error rather than silently
routing sparse data as missing.
