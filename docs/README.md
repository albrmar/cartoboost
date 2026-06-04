# GeoBoost Documentation

This documentation explains how to install, train, validate, and extend
GeoBoost. Start with the API and feature docs when using the package, then use
the testing and artifact docs when changing behavior.

## Contents

- [v1 API](v1_api.md) shows the supported Python estimator, Rust-backed options,
  and CLI commands.
- [Sparse Features](sparse_features.md) documents list-valued route-cell-style
  sparse columns for Python training and prediction.
- [Feature Schema](feature_schema.md) documents numeric, periodic, and
  sparse-set declarations.
- [SHAP Support](shap.md) explains how to generate SHAP explanations for dense
  predictions.
- [Model Artifact](model_artifact.md) explains native JSON model files,
  compatibility, metadata, and load/save behavior.
- [Limitations](limitations.md) states the current alpha limits so callers know
  which workflows are supported.
- [Implementation Status](implementation_status.md) lists implemented behavior
  and remaining hardening work.
- [Testing Strategy](testing_strategy.md) explains the validation commands and
  the unit, integration, fuzz, and benchmark coverage.
- [Fixture Contract](fixture-contract.md) describes committed test data and
  golden outputs under `tests/`.
- [Golden Data Workflow](golden-data-workflow.md) explains how to update fixture
  expectations without weakening tests.
- [Integration Contract](integration-contract.md) records the Python API shape
  that implementation work should preserve.
- [Repository Plan](repo_plan.md) records the target product, architecture,
  milestone plan, and definition of done.
- [v1 Release Checklist](v1_release_checklist.md) tracks release-candidate gates.
- [NYC Taxi Quality Benchmarks](nyc_taxi_benchmarks.md) documents optional
  real-data GeoBoost, XGBoost, LightGBM, and mean-baseline comparisons,
  including benchmark-only dependencies and fair feature handling.
- [Segmentation Proofs](assets/) contains generated PNGs showing learned
  spatial segmentation boundaries on deterministic synthetic datasets.

## Quick Start

Install the development environment and native extension:

```sh
uv sync --group dev
uv run --group dev maturin develop
```

Train and predict from Python:

```python
from geoboost import GeoBoostRegressor

model = GeoBoostRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    splitters=["axis"],
    backend="rust",
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Train and predict from dense numeric CSV files:

```sh
geoboost train --data train.csv --config configs/regression.toml --model-out model.json
geoboost predict --model model.json --input test.csv --predictions-out predictions.csv
```

Run the full local validation suite:

```sh
just validate
```

## Current Scope

The repository contains a regression implementation with spatial, temporal,
fuzzy, sparse, schema-aware, and linear-leaf support. Advanced splitters,
sparse features, schema-driven training, fuzzy training, and linear leaves
require the Rust native extension.
