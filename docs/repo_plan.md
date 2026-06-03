# GeoBoost-RS Repository Plan

This repository is a clean-room, GeoBoost-inspired implementation based only on
public descriptions of modular spatiotemporal boosted trees. It does not attempt
to reproduce Lyft's proprietary implementation.

## Core Principle

Rust owns correctness, training, serialization, and inference. Python owns
ergonomics, experiments, data loading, plots, and test orchestration.

The implementation uses:

- Rust workspace crates for core training, Python bindings, and CLI workflows.
- PyO3 and maturin for the Python extension module.
- pytest and ruff for Python validation.
- Cargo tests, clippy, and bench compile smoke checks for Rust validation.

Validation scripts that train advanced splitter, fuzzy, sparse, or linear-leaf
fixtures require the PyO3 native extension. CI handles that boundary in a
dedicated validation-artifacts job by running `maturin develop` before
`scripts/run_full_validation.py`.

## Target Product

Python API:

```python
from geoboost import GeoBoostRegressor

model = GeoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    splitters=[
        "axis",
        "diagonal_2d",
        "gaussian_2d",
        "periodic_time",
        "sparse_set",
    ],
    leaf_predictor="constant",
    fuzzy=True,
)

model.fit(X_train, y_train, feature_schema=schema)
pred = model.predict(X_test)
model.save("model.geoboost")
```

Rust API:

```rust
use geoboost_core::Model;

let model = Model::load("model.geoboost")?;
let y_hat = model.predict_one(&features);
```

CLI:

```bash
geoboost train --config configs/regression.toml --data data/train.csv --model-out artifacts/model.geoboost
geoboost predict --model artifacts/model.geoboost --input data/test.csv
geoboost eval --model artifacts/model.geoboost --data data/test.csv
geoboost inspect --model artifacts/model.geoboost
```

## First Complete Version Scope

The first complete version is regression-only.

Implemented scope:

1. L2 regression gradient boosting.
2. Constant leaves.
3. Axis-aligned numerical splits.
4. Deterministic training.
5. Diagonal 2D split candidates.
6. Gaussian/radial 2D split candidates.
7. Periodic interval split candidates with wraparound handling.
8. Fuzzy split wrapping and weighted prediction recursion.
9. Weighted ridge linear leaves.
10. Sparse scalar-ID contains-any split candidates.
11. Versioned JSON model serialization.
12. Rust/Python/CLI prediction workflows.
13. Validation through `just validate`.

Future hardening:

1. Native list-valued sparse set features for route-cell arrays.
2. Fully fractional fuzzy training propagation and fuzzy-specific gain scoring.
3. Artifact migrations beyond artifact version `1`.
4. scikit-learn and LightGBM comparison reports.
5. Long-running property and fuzz suites.
6. Parquet input support for CLI workflows.

## Rust Design

Main concepts:

- `BoosterConfig` defines the training parameters.
- `Booster` trains L2 gradient-boosted regression trees.
- `TreeBuilder` proposes splitter candidates and recursively builds trees.
- `Split` stores serializable routing logic.
- `Node` stores constant leaves, linear leaves, and branch nodes.
- `Model` owns artifact versioning, trees, learning rate, and prediction.

Split representation includes:

- `Axis`
- `Diagonal2D`
- `Gaussian2D`
- `PeriodicInterval`
- `SparseSetContainsAny`
- `Fuzzy`

Prediction is:

```text
prediction = init_prediction
for tree in trees:
    prediction += learning_rate * tree.predict(x)
```

For L2, each boosting round fits the negative gradient:

```text
gradient_i = pred_i - y_i
tree_target_i = y_i - pred_i
prediction_i += learning_rate * tree_update_i
```

This invariant is covered by tests.

## Splitter Plan

Axis:

- Uses sorted unique feature values.
- Candidate thresholds are midpoints between adjacent values.
- Respects `min_samples_leaf`.

Diagonal 2D:

- Uses fixed normal candidates such as `(1, 1)`, `(1, -1)`, and `(-1, 1)`.
- Scores thresholds along projected coordinates.

Gaussian/radial 2D:

- Uses a node-local center from feature means.
- Scores radius thresholds from sorted distances.

Periodic:

- Supports modulo normalization.
- Supports wraparound intervals such as `22..2`.
- Current candidate grid targets common day/hour intervals.

Sparse set:

- Current implementation supports scalar integer-ID columns.
- Candidate split is `contains any ID from selected set`, with one ID per
  candidate.
- Native list/set-valued columns are a future hardening item.

Fuzzy:

- Wraps a learned hard split.
- Prediction conserves branch mass with left/right weights summing to `1`.
- Full fractional training propagation is a future hardening item.

## Leaf Predictor Plan

Constant leaves:

- Fit the weighted mean of the residual target in each leaf.

Linear leaves:

- Fit residuals with weighted ridge regression.
- Supports configurable feature indices.
- Applies ridge regularization to non-intercept coefficients.
- Falls back to constant leaves if fitting fails.

## Python API Plan

`GeoBoostRegressor` exposes:

- `fit`
- `predict`
- `save`
- `load`
- `get_params`
- `set_params`

Supported constructor options include:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_leaf`
- `min_gain`
- `loss="l2"`
- `splitters`
- `leaf_predictor`
- `linear_leaf_features`
- `fuzzy`
- `fuzzy_bandwidth`
- `l2_regularization`
- `random_state`
- `n_threads`

The Rust backend handles the implemented advanced options. The pure-Python
fallback remains intentionally limited to axis splits with constant leaves.

## Testing Philosophy

The test suite proves exact behavior for fixed examples, checks mathematical
invariants over generated inputs, verifies Rust/Python/CLI parity, and validates
empirical behavior on synthetic datasets. It does not prove universal optimality
or production superiority on arbitrary real-world data.

Validation currently covers:

- L2 initial prediction, gradient, and residual helpers.
- Dataset validation.
- Golden one-stump behavior.
- Serialization round trips.
- Committed parity fixture generation and saved artifact prediction identity.
- Diagonal, radial, periodic, fuzzy, sparse, and linear-leaf behavior.
- Python estimator API behavior.
- sklearn clone, Pipeline, and GridSearchCV compatibility.
- Native backend smoke tests for special splitters.
- CLI train/predict/eval smoke behavior.
- Generated PNG proof images for diagonal and radial segmentation.
- Generated validation artifacts after installing the native extension for
  `GeoBoostRegressor(backend="rust")` workflows.

## Acceptance Criteria

The repo is credible when this command passes from a clean checkout:

```bash
just validate
```

Current `just validate` runs:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv run --group dev ruff format --check python tests scripts
uv run --group dev ruff check python tests scripts
uv run --group dev maturin develop
uv run --group dev pytest
uv run --group dev python scripts/run_full_validation.py
cargo bench --workspace --no-run
```

Planned validation extensions:

- `cargo nextest run --workspace`
- `mypy python/geoboost tests/python`
- Hypothesis property profiles.
- Fuzz smoke tests.
- Full synthetic validation report generation.
- scikit-learn and LightGBM comparison scripts.

## Milestones

Milestone 0: skeleton

- Rust workspace.
- Python package skeleton.
- maturin build.
- CI linting.
- README with scope.

Milestone 1: exact one-stump L2 GBT

- Dataset representation.
- L2 loss.
- Constant leaf predictor.
- Axis splitter.
- Golden one-stump test.
- Python binding.

Milestone 2: general axis-aligned trees

- `max_depth`.
- `min_samples_leaf`.
- `min_gain`.
- Multiple estimators.
- Learning rate.
- Batch prediction.

Milestone 3: spatial splitters

- Diagonal 2D splitter.
- Gaussian/radial 2D splitter.
- Synthetic spatial tests.

Milestone 4: periodic temporal splitter

- Periodic intervals.
- Wraparound handling.
- Synthetic temporal tests.

Milestone 5: fuzzy splits

- Fuzzy wrapper.
- Weighted inference recursion.
- Volatility-oriented tests.

Milestone 6: linear leaves

- Weighted ridge solver.
- Linear leaf model.
- Feature subset config.
- Regularization.

Milestone 7: sparse high-cardinality splitter

- Contains-any splitter.
- Candidate ID mining.
- Scalar-ID support now, native list-valued support later.

Milestone 8: artifact and CLI

- Versioned model format.
- CLI train/predict/eval/inspect.
- Metadata and compatibility tests.

Milestone 9: comparison suite

- scikit-learn comparison.
- LightGBM comparison.
- Synthetic benchmark report.
- Latency and artifact-size tracking.

## Definition Of Done

The working implementation should remain:

- Clean-room.
- Regression-only until classification is explicitly added.
- Deterministic for fixed inputs.
- Honest about empirical claims.
- Validated by `just validate`.
- Documented with clear implementation status.
