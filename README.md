# geoboost

Clean-room GeoBoost-inspired regression tools exposed through Rust, Python, and
a small command-line interface.

This project is based only on public descriptions of modular spatiotemporal
boosted trees. It does not reproduce or claim equivalence to Lyft's proprietary
GeoBoost implementation.

## Current Scope

GeoBoost is a regression-only alpha with a v1 release-candidate contract focused
on deterministic synthetic validation, artifact fidelity, and explicit API
limits. The Rust backend owns the advanced behavior:

- L2 gradient boosting with constant or linear leaves.
- Axis, diagonal 2D, Gaussian/radial 2D, periodic, sparse-set, and fuzzy splits.
- Fractional fuzzy training and weighted prediction recursion.
- Dense numeric features plus list-valued sparse route-cell-style columns.
- Optional feature schema metadata for numeric, periodic, and sparse-set
  feature declarations.
- Versioned JSON artifacts with Python and CLI loading paths.

The pure-Python fallback exists for sklearn ergonomics and dense axis-split
experiments. Advanced splitters, sparse sets, schema-driven training, fuzzy
training, and linear leaves require the Rust native extension.

## Project Layout

- `crates/geoboost-core`: Rust library for core training, prediction, and model
  artifacts.
- `crates/geoboost-cli`: Dense numeric CSV command-line interface.
- `crates/geoboost-py`: PyO3/maturin extension crate that publishes
  `geoboost._native`.
- `python/geoboost`: Python package surface.
- `configs`: Example training configuration files.
- `docs`: API, testing, artifact, sparse-feature, schema, and release docs.
- `tests`: Python and integration tests.
- `fuzz`: cargo-fuzz harnesses.
- `benches`: Criterion benchmark scaffolding.

## Installation

Development requires stable Rust, Python 3.10 or newer, and `uv` 0.7 or newer.

```sh
uv sync --group dev
uv run --group dev maturin develop
```

The second command builds and installs the native extension into the development
environment. Use `backend="rust"` when you want failures to be explicit if the
native extension is unavailable.

## Dense Python Example

```python
from geoboost import GeoBoostRegressor

X = [[0.0], [1.0], [2.0], [3.0]]
y = [0.0, 1.0, 2.0, 3.0]

model = GeoBoostRegressor(
    n_estimators=20,
    learning_rate=0.1,
    max_depth=2,
    splitters=["axis"],
    backend="rust",
)
model.fit(X, y)
predictions = model.predict(X)
```

The estimator supports sklearn-style `get_params`, `set_params`, `clone`,
`Pipeline`, `GridSearchCV`, and NumPy-array predictions for the covered API.

## Sparse Route-Cell Example

List-valued sparse features are passed as a mapping from sparse column name to
one list of IDs per row.

```python
from geoboost import GeoBoostRegressor

X_dense = [[0.0], [0.0], [0.0], [0.0]]
y = [10.0, 10.0, 0.0, 0.0]
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

Sparse IDs must be non-negative integers. Duplicate IDs in a row are normalized
by the Rust dataset layer. A model containing sparse-list splits requires
`sparse_sets=` at prediction time.

## Feature Schema Example

Schemas tell the Rust trainer which dense features are numeric or periodic and
which sparse columns are sparse-set columns.

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

model = GeoBoostRegressor(
    splitters=["axis", "periodic:24", "sparse_set"],
    backend="rust",
)
model.fit(
    X_dense,
    y,
    sparse_sets={"route_cells": route_cells},
    feature_schema=schema,
)
```

Schema length must equal the dense feature count plus the sparse-set column
count. Periodic splitters use declared periods when a schema is present instead
of relying on observed coverage heuristics.

## Save And Load

```python
model.save("model.geoboost.json")

loaded = GeoBoostRegressor.load("model.geoboost.json")
assert loaded.get_params()["splitters"] == model.get_params()["splitters"]
predictions = loaded.predict(X_dense, sparse_sets={"route_cells": route_cells})
```

Native artifacts are versioned JSON and include optional metadata, feature
schema, and training configuration fields. See
[`docs/model_artifact.md`](docs/model_artifact.md).

## CLI Dense CSV Example

The CLI v1 contract is dense numeric CSV train/predict/eval. Python is the v1
surface for list-valued sparse route-cell features.

```sh
geoboost train --data train.csv --config configs/regression.toml --model-out model.json
geoboost predict --model model.json --input test.csv --predictions-out predictions.csv
geoboost eval --model model.json --data test_with_target.csv
```

The CLI rejects unknown commands, malformed config, unknown splitters, unknown
leaf predictors, missing targets, and wrong feature counts with nonzero exit
status and actionable messages.

## Validation

Local validation source of truth:

```sh
just validate
```

Equivalent command sequence:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv sync --group dev
uv run --group dev ruff format --check python tests scripts
uv run --group dev ruff check python tests scripts
uv run --group dev maturin develop
uv run --group dev pytest
uv run --group dev python scripts/run_full_validation.py
uv run --group dev python scripts/run_v1_validation.py
cargo bench --workspace --no-run
```

Generated validation artifacts are written under `target/validation/`. They are
deterministic smoke evidence over synthetic fixtures and should not be read as a
claim of production superiority over other GBT libraries.

## Documentation

- [v1 API](docs/v1_api.md)
- [Sparse Features](docs/sparse_features.md)
- [Feature Schema](docs/feature_schema.md)
- [Model Artifact](docs/model_artifact.md)
- [Testing Strategy](docs/testing_strategy.md)
- [Limitations](docs/limitations.md)
- [Implementation Status](docs/implementation_status.md)
- [v1 Release Checklist](docs/v1_release_checklist.md)

## Limitations

- Regression only; no classification objectives.
- Advanced behavior requires the native Rust backend.
- CLI v1 is dense numeric CSV only.
- Validation uses deterministic synthetic fixtures, not broad production
  benchmarks.
- Artifact version `1` supports backward-compatible optional metadata fields,
  but there is no multi-version migration framework yet.
- Schema support covers numeric, periodic, and sparse-set declarations; named
  spatial-pair contracts remain future hardening.

## License

MIT
