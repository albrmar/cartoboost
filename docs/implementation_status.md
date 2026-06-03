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
- Fuzzy training support that wraps learned hard splits and uses weighted branch
  recursion at prediction time.
- Weighted ridge regression primitive and tree-training integration for linear
  leaf models.
- Unit tests for L2 loss, stump training, serialization, spatial/periodic/fuzzy
  routing, sparse routing, and linear leaf fitting.
- Committed parity fixtures and generated spatial segmentation proof images.

## Future Hardening

- Native list-valued sparse set feature columns rather than scalar ID routing.
- Fully fractional fuzzy training propagation and fuzzy-specific gain scoring.
- Backward-compatible artifact migrations beyond artifact version `1`.
- Full comparison suite against scikit-learn and LightGBM.

The current implementation covers the repo's regression-only clean-room target
with deterministic Rust training, Rust/Python artifact parity, CLI workflows,
spatial/temporal/sparse/fuzzy split support, and linear leaves. The future items
above are production hardening and richer data representation work rather than
missing baseline functionality.

## Alpha Hardening Status

The project is still alpha software. The implemented path is credible for fixed
regression experiments and CI smoke validation, but the production contract is
intentionally narrow:

- Rust is the authoritative backend for advanced splitters, fuzzy prediction,
  sparse scalar-ID routing, linear leaves, artifacts, and CLI prediction.
- The pure-Python estimator fallback is an ergonomics and sklearn-compatibility
  path for axis splits with constant leaves only.
- Full validation artifact generation requires the PyO3 extension to be built
  and installed first because the generator trains with `backend="rust"`.
- The committed proof images and metrics are smoke evidence, not a claim of
  superiority over production geospatial boosting systems.
- Artifact version `1` is supported, but no older-version migration layer exists
  yet.
