# Architecture

GeoBoost is a Rust workspace with Python bindings and a Python package. The
main design goal is to keep model semantics in one place while exposing useful
surfaces for Python and command-line users.

## Layering

| Layer | Owns | Should not own |
| --- | --- | --- |
| `crates/geoboost-core` | Dataset representation, losses, split candidates, tree building, prediction, metrics, serialization. | Python-specific input coercion or CLI formatting. |
| `crates/geoboost-py` | PyO3 classes, array/list conversion, native method exposure. | New training semantics that bypass core. |
| `python/geoboost` | sklearn-style estimator API, validation, fallback path, schema helpers, SHAP integration. | Divergent model behavior for native-only features. |
| `crates/geoboost-cli` | Dense CSV parsing, config parsing, command output, CLI errors. | Sparse route-cell Python workflows. |
| `scripts` | Validation reports, benchmark runs, fixture generation, proof images. | Runtime library behavior. |

## Training Flow

1. Python or CLI validates user inputs.
2. Inputs are converted into dense rows plus optional sparse-set columns.
3. Core Rust constructs a `Dataset`.
4. `Booster::fit` builds trees from configured loss, splitters, leaves, and
   constraints.
5. Prediction routes through the stored tree ensemble and returns dense numeric
   outputs.
6. Save/load uses versioned JSON artifacts.

## Splitter Boundary

Splitter implementations live under `crates/geoboost-core/src/splitters`.
Candidate generation and routing should remain deterministic and testable in
Rust. Python should validate names and convert user-facing aliases, but should
not reimplement native split semantics.

## Serialization Boundary

The model artifact is a public contract. Backward-compatible optional fields
are acceptable within artifact version `1`; incompatible changes require a new
version and explicit loader behavior.

## Python Fallback Boundary

The fallback exists for sklearn ergonomics and dense axis-split constant-leaf
experiments. It should not grow into a second implementation of advanced Rust
features unless there is a deliberate design decision and parity tests.
