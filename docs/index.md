# GeoBoost Documentation

GeoBoost is a Rust-backed Python regression package for tabular,
spatiotemporal, sparse-route, and fuzzy split experiments. The project is
currently alpha quality and regression-only, with deterministic training,
inspectable JSON artifacts, and explicit limits around unsupported workflows.

## Start Here

- [Getting Started](getting-started.md) gets a local development install running
  and trains a first model.
- [Python Estimator](user-guide/python-estimator.md) covers the sklearn-style
  API, native backend behavior, sparse features, schema metadata, save/load, and
  SHAP entry points.
- [Parameters](user-guide/parameters.md) explains the supported training
  controls and which ones require the Rust backend.
- [CLI](user-guide/cli.md) documents dense CSV train, predict, eval, and inspect
  workflows.
- [Developer Guide](developer-guide/index.md) explains the workspace layout,
  validation commands, extension points, docs workflow, and release process.
- [Benchmarks](benchmarks/index.md) collects benchmark runbooks and generated
  artifact policy.

## Current Scope

GeoBoost supports:

- L2 and quantile regression objectives.
- Constant and linear residual leaves.
- Axis, histogram-axis, diagonal 2D, Gaussian/radial 2D, periodic, sparse-set,
  and fuzzy split behavior.
- Dense numeric arrays plus list-valued sparse-set columns in Python.
- Feature schema metadata for dense numeric, dense periodic, and sparse-set
  declarations.
- Versioned JSON model and weights artifacts, with optional ONNX export for a
  dense axis-tree subset.
- Python API ergonomics compatible with common sklearn estimator workflows.

The Rust native extension is required for advanced splitters, sparse-set
features, feature schemas, fuzzy routing, linear leaves, and native artifact
loading. The pure-Python fallback is intentionally narrow: dense axis-split
constant-leaf experiments.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `crates/geoboost-core` | Core Rust training, prediction, tree, loss, split, metric, and artifact logic. |
| `crates/geoboost-cli` | Dense numeric CSV command-line interface. |
| `crates/geoboost-py` | PyO3 bindings that expose `geoboost._native`. |
| `python/geoboost` | Python estimator, schema helpers, SHAP integration, and fallback path. |
| `docs` | User docs, developer docs, contracts, validation notes, and benchmark reports. |
| `tests` | Python, integration, fixture, parity, and CLI tests. |
| `scripts` | Validation, benchmark, fixture, and proof-image utilities. |

## Validation

Use `just validate` as the local source of truth before release work. For docs
changes, also run:

```sh
uv sync --group docs --no-install-project
uv run --group docs --no-sync mkdocs build --strict
```
