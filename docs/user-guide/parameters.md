# Parameters

This page summarizes the public training controls exposed by
`GeoBoostRegressor`.

## Core Boosting

| Parameter | Default | Notes |
| --- | --- | --- |
| `n_estimators` | `100` | Number of boosting rounds. Must be non-negative. |
| `learning_rate` | `0.05` | Shrinks each tree contribution. Must be finite and positive. |
| `max_depth` | `4` | Maximum tree depth. `0` produces a constant model. |
| `min_samples_leaf` | `20` | Minimum weighted row count per leaf candidate. |
| `min_gain` | `1e-8` | Minimum gain required to split. |
| `random_state` | `None` | Reserved for deterministic APIs; current training paths are deterministic. |
| `n_threads` | `None` | Public parameter retained for compatibility; current Rust binding does not expose threaded training control. |

## Loss

| Parameter | Default | Notes |
| --- | --- | --- |
| `loss` | `"l2"` | Accepts `"l2"`, `"squared_error"`, `"quantile"`, or `"pinball"`. |
| `quantile_alpha` | `0.5` | Required to be finite and in `(0, 1)` for quantile loss. |

Quantile loss currently requires `leaf_predictor="constant"`.

## Splitters

| Name | Purpose | Backend |
| --- | --- | --- |
| `axis` | Standard one-feature threshold splits. | Rust and Python fallback |
| `axis_histogram`, `axis_hist`, `histogram` | Axis histogram aliases; `axis_histogram:<bins>` and `axis_hist:<bins>` are accepted by Python validation. | Rust |
| `diagonal_2d`, `diagonal2d` | Oblique 2D spatial boundary candidates. | Rust |
| `gaussian_2d`, `gaussian2d`, `radial` | Radial 2D candidates. | Rust |
| `periodic_time`, `periodic_24`, `periodic:<period>` | Periodic wraparound splits. | Rust |
| `sparse_set`, `sparse` | List-valued sparse ID set splits. | Rust |

Unknown splitter names raise `ValueError`.

## Leaves

| Parameter | Default | Notes |
| --- | --- | --- |
| `leaf_predictor` | `"constant"` | Accepts `"constant"` or `"linear"`. |
| `linear_leaf_features` | `None` | Python API currently expects stringified integer feature indices, such as `["0", "2"]`. |
| `l2_regularization` | `1.0` | Ridge penalty for linear leaves. |

Linear leaves require the Rust backend.

## Fuzzy Routing

| Parameter | Default | Notes |
| --- | --- | --- |
| `fuzzy` | `False` | Enables fractional branch assignment during training and weighted prediction recursion. |
| `fuzzy_bandwidth` | `0.0` | Split transition bandwidth. Must be finite and non-negative. |

Fuzzy routing requires the Rust backend and is not compatible with monotonic
constraints.

## Monotonic Constraints

`monotonic_constraints` is a list of `-1`, `0`, or `1` values with one entry per
dense feature:

- `1` requires predictions to be non-decreasing in that feature.
- `-1` requires predictions to be non-increasing in that feature.
- `0` leaves the feature unconstrained.

Current constraints require constant leaves, non-fuzzy training, and axis-style
splitters.

## Backend Compatibility

| Feature | Rust | Python fallback |
| --- | --- | --- |
| Dense axis constant leaves | Yes | Yes |
| Sparse-set features | Yes | No |
| Feature schema | Yes | No |
| Diagonal, Gaussian, periodic splitters | Yes | No |
| Fuzzy training | Yes | No |
| Linear leaves | Yes | No |
| Native JSON artifacts | Yes | Load requires native extension |
