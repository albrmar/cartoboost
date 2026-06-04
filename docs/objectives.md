# Objectives

GeoBoost is currently a regression-only package. Objective support is split
between the stable public API and native scaffolding that is not yet exposed
end-to-end.

## Objective Status

| Objective | Public names | Status |
| --- | --- | --- |
| L2 squared error | `"l2"`, `"squared_error"` | Stable public objective for Rust and Python fallback paths. |
| Quantile pinball | `"quantile"`, `"pinball"` | Public alpha objective for constant leaves. |
| Huber | Not public | Native loss/config scaffold exists in core Rust, but Python and CLI parsers do not expose it. |
| Log-L2 | Not public | Native config scaffold exists in core Rust, but it is not wired into public training. |
| Conformal intervals | Not public | Evaluation scaffold only; no fitted interval estimator contract. |

## L2

L2 is the default and the most validated objective. It initializes from the
weighted mean target, trains on residuals, and supports the current splitters,
sample weights, constant leaves, and Rust linear leaves.

Use L2 when comparing splitter behavior, sparse routing, fuzzy routing, linear
leaves, or benchmark outputs unless the experiment is explicitly about quantile
behavior.

## Quantile

Quantile regression uses weighted quantile initialization and pinball loss. The
public names are `loss="quantile"` and `loss="pinball"`, with
`quantile_alpha` required to be finite and in `(0, 1)`.

Current limits:

- Quantile loss requires `leaf_predictor="constant"`.
- Quantile behavior is available through the Python estimator and native
  backend parameter surface.
- Quantile benchmark evidence should be reported separately from L2 benchmark
  evidence because the acceptance metrics are not interchangeable.

## Huber And Log-L2

Huber and log-L2 should be described as scaffolding, not supported user-facing
objectives. Core Rust types exist for Huber and log-L2 configuration, and Huber
has loss primitive behavior, but the public Python estimator and CLI objective
parsers currently accept only L2 and quantile aliases.

Before either objective becomes public, it needs:

- Public parameter names and validation rules.
- Rust booster wiring for initialization, gradients, hessians, leaf values, and
  split scoring.
- PyO3, Python fallback, CLI, artifact, and documentation coverage.
- Targeted tests showing behavior with sample weights and representative
  splitter combinations.

## Conformal Scaffold

Conformal prediction is not a fitted interval feature in the current estimator.
For v2 alpha docs, "conformal scaffold" means the planned evaluation pattern:
train a point or quantile model, reserve calibration data, compute residual or
pinball-style calibration scores, and report empirical coverage on blocked and
random holdouts.

Do not document conformal intervals as a production API until there is an
estimator method or artifact contract that stores calibration metadata and
reproduces intervals after load.

## Reporting

Objective reports should include:

- Objective name and parameters, especially `quantile_alpha`.
- Leaf predictor type.
- Backend used.
- Sample-weight handling if weights are present.
- Evaluation split type, including whether the split is random, temporal,
  spatial, or otherwise blocked.
