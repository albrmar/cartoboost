# Modeling v2 Alpha

Modeling v2 is the alpha documentation bucket for GeoBoost-inspired modeling
features that are beyond the narrow v1 regression contract. This repository is
a clean-room implementation inspired by public geospatial boosting ideas. It is
not Lyft's proprietary GeoBoost implementation, does not claim compatibility
with that system, and should not be described as a reproduction of it.

## Status Summary

| Area | Alpha status |
| --- | --- |
| L2 regression | Stable public objective for current training paths. |
| Quantile regression | Public `loss="quantile"`/`"pinball"` path for constant leaves. |
| Huber loss | Core loss/config scaffold exists; not exposed as a public Python or CLI training objective. |
| Log-L2 loss | Core config scaffold exists; not exposed as a public Python or CLI training objective. |
| Conformal intervals | Documentation and evaluation scaffold only; no public fitted interval API. |
| Monotonic constraints | Implemented for axis-style splitters with constant leaves and non-fuzzy training. |
| Linear leaves | Rust weighted ridge residual leaves exist; elastic-net linear leaves are not implemented. |
| Fuzzy routing | Rust native fractional routing is implemented for supported splitters. |
| H3 sparse features | Sparse integer-ID route-cell scaffold can carry encoded H3 cells; no H3-native encoder or hierarchy logic is included. |
| Blocked evaluation | Supported as benchmark/validation practice; not a first-class estimator cross-validation API. |

## Clean-Room Scope

Use "GeoBoost-inspired" when describing v2 modeling additions. The implemented
features are based on this repository's own Rust/Python contracts, tests, and
benchmark scripts. The docs should avoid any implication that behavior was
derived from, validated against, or intended to match Lyft-internal code,
artifacts, data, or production systems.

## Public Contract

The public alpha contract remains regression-only. The Rust native backend is
the authoritative path for advanced splitters, sparse-set features, fuzzy
routing, linear leaves, monotonic constraints, and native model artifacts. The
pure-Python fallback is intentionally limited to dense axis-split experiments
with constant leaves.

When documenting experiments, distinguish three levels:

| Level | Meaning |
| --- | --- |
| Public | Exposed through `GeoBoostRegressor` or CLI validation with documented parameter behavior. |
| Native scaffold | Present in Rust types/helpers, but not wired through all public entry points. |
| Evaluation scaffold | Available as scripts, fixtures, diagnostics, or docs guidance rather than estimator behavior. |

## v2 Additions

The v2 alpha surface collects work that is useful for spatial regression
experiments but still needs hardening before a stable contract:

- Robust and asymmetric objectives are tracked in [Objectives](objectives.md).
- Monotonic constraints are tracked in [Constraints](constraints.md).
- Spatial splitters, H3-style sparse IDs, diagnostics, fuzzy routing, blocked
  evaluation, and spatial validation practice are tracked in
  [Spatial Modeling](spatial_modeling.md).
- Benchmark output remains evidence for the exact committed setup, not a claim
  of broad superiority over other packages or production geospatial systems.

## Compatibility Notes

Models that use v2 alpha features should be saved through the native JSON
artifact path so splitters, schemas, sparse-set metadata, fuzzy settings,
constraints, loss configuration, and leaf predictor configuration remain
inspectable. Alpha artifacts are intended to be readable and deterministic, but
multi-version migration beyond artifact version `1` is not yet a production
guarantee.
