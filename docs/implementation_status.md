# Implementation Status

This repository is a clean-room GeoBoost-inspired implementation. It does not
attempt to reproduce Lyft's proprietary system.

## Implemented

- Rust workspace with `geoboost-core`, `geoboost-py`, and `geoboost-cli`.
- L2 regression gradient boosting with deterministic axis-aligned trees.
- Constant leaf values.
- Versioned JSON model artifact with Rust/Python save/load.
- PyO3-backed `GeoBoostRegressor` with a pure-Python fallback.
- sklearn-compatible estimator behavior for clone, Pipeline, GridSearchCV, and
  NumPy prediction outputs.
- CLI train/predict/eval for numeric CSV regression using the Rust core model.
- Split artifact variants and prediction routing for axis, diagonal 2D,
  Gaussian/radial 2D, periodic interval, sparse integer-ID, and fuzzy splits.
- Training-time candidate search for axis, diagonal 2D, Gaussian/radial 2D, and
  periodic interval splitters on dense numeric data.
- Training-time sparse scalar-ID contains-any split search.
- Training-time list-valued sparse-set contains-any split search for route-cell
  style rows, with scalar-ID sparse routing retained for dense compatibility.
- Feature schema metadata for numeric, periodic, and sparse-set features, with
  schema-aware periodic and sparse-set candidate selection in the Rust trainer.
- Fractional fuzzy split scoring and child training weights, plus weighted
  branch recursion at prediction time.
- Weighted ridge regression primitive and tree-training integration for linear
  leaf models.
- Sample weights through Rust, PyO3, and the Python estimator fallback.
- Self-describing model artifacts with optional metadata, feature schema, and
  training config fields while preserving v1 JSON load compatibility.
- Criterion benchmark scaffolding, deterministic property-style tests, cargo-fuzz
  harnesses, and a small sklearn baseline comparison report script.
- Unit tests for L2 loss, stump training, serialization, spatial/periodic/fuzzy
  routing, sparse routing, and linear leaf fitting.
- Committed parity fixtures and generated spatial segmentation proof images.

## Future Hardening

- Backward-compatible artifact migrations beyond artifact version `1`.
- Larger comparison suite against scikit-learn and LightGBM.
- Richer schema contracts for named spatial pairs and Python-to-Rust schema
  construction beyond metadata retention.

The current implementation covers the repo's regression-only clean-room target
with deterministic Rust training, Rust/Python artifact parity, CLI workflows,
spatial/temporal/sparse/fuzzy split support, and linear leaves. The future items
above are production hardening beyond the alpha contracts rather than missing
baseline functionality.

## Alpha Hardening Status

The project is still alpha software. The implemented path is credible for fixed
regression experiments and CI smoke validation, but the production contract is
intentionally narrow:

- Rust is the authoritative backend for advanced splitters, fuzzy training and
  prediction, sparse scalar-ID and list-valued sparse routing, linear leaves,
  artifacts, and CLI prediction.
- The pure-Python estimator fallback is an ergonomics and sklearn-compatibility
  path for axis splits with constant leaves only.
- Full validation artifact generation requires the PyO3 extension to be built
  and installed first because the generator trains with `backend="rust"`.
- The committed proof images and metrics are smoke evidence, not a claim of
  superiority over production geospatial boosting systems.
- Artifact version `1` is supported with backward-compatible optional metadata
  fields, but no older-version migration layer exists yet.
