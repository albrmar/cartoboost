# Implementation Status

This repository is a clean-room GeoBoost-inspired implementation. It does not
attempt to reproduce Lyft's proprietary system.

## Implemented

- Rust workspace with `geoboost-core`, `geoboost-py`, and `geoboost-cli`.
- L2 regression gradient boosting with deterministic axis-aligned trees.
- Constant leaf values.
- Versioned JSON model artifact with Rust/Python save/load.
- PyO3-backed `GeoBoostRegressor` with a pure-Python fallback.
- CLI train/predict/eval for numeric CSV regression using the Rust core model.
- Split artifact variants and prediction routing for axis, diagonal 2D,
  Gaussian/radial 2D, periodic interval, sparse integer-ID, and fuzzy splits.
- Training-time candidate search for axis, diagonal 2D, Gaussian/radial 2D, and
  periodic interval splitters on dense numeric data.
- Weighted ridge regression primitive for linear leaf models.
- Unit tests for L2 loss, stump training, serialization, spatial/periodic/fuzzy
  routing, sparse routing, and linear leaf fitting.

## Planned

- Candidate search and training support for sparse and fuzzy splitters.
- Weighted fuzzy training propagation and gain calculation.
- Full linear-leaf integration into tree training.
- Native sparse set feature columns rather than scalar ID routing.
- Backward-compatible artifact migrations beyond artifact version `1`.
- Full comparison suite against scikit-learn and LightGBM.

The current implementation is a solid Milestone 1 plus core primitives for later
milestones. It is not yet a production-complete spatiotemporal boosted-tree
system.
