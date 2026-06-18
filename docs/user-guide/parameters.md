# Parameters

This page summarizes the public training controls exposed by
`CartoBoostRegressor`, with emphasis on choosing splitters for temporal-spatial
regression.

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
| `loss` | `"l2"` | Accepts `"l2"`, `"squared_error"`, `"l1"`, `"mae"`, `"absolute_error"`, `"huber"`, `"log_l2"`, `"quantile"`, or `"pinball"`. |
| `quantile_alpha` | `0.5` | Required to be finite and in `(0, 1)` for quantile loss. |
| `huber_delta` | `1.0` | Positive clipping threshold for Huber loss. |
| `log_offset` | `1.0` | Positive offset for `log_l2`; the current backend supports `1.0`. |

`l1`, `huber`, `log_l2`, and quantile loss currently require `leaf_predictor="constant"`.

## Splitters

The splitter list is the main CartoBoost modeling choice. By default,
`splitters=None` uses `auto`: exact `axis` on small or constrained fits, and a
fast histogram-axis search for larger dense L2 fits. Use `axis` explicitly when
you need exact threshold search, then add the splitters that match your data:

| Name | Purpose |
| --- | --- |
| `auto` | Default policy that chooses exact axis or histogram-axis training from the fit shape and objective. |
| `axis` | Standard one-feature threshold splits. |
| `axis_histogram`, `axis_hist`, `histogram`, `axis_histogram:<bins>` | Fast axis-threshold search for dense numeric features. |
| `diagonal_2d`, `diagonal2d` | Oblique 2D boundaries for coordinates or projected x/y features. |
| `gaussian_2d`, `gaussian2d`, `radial` | Radial neighborhoods around local hotspots, depots, zones, or corridors. |
| `periodic_time`, `periodic_24`, `periodic:<period>` | Wraparound time features such as hour-of-day, weekday, or seasonal phase. |
| `sparse_set`, `sparse` | List-valued route-cell, zone, grid, or encoded H3 memberships. |

Unknown splitter names raise `ValueError`.

Common temporal-spatial combinations:

| Problem shape | Suggested splitters |
| --- | --- |
| General tabular baseline | `None` or `["auto"]` |
| Exact axis baseline | `["axis"]` |
| Dense location and time | `["axis", "diagonal_2d", "gaussian_2d", "periodic:24"]` |
| Route or cell membership | `["axis", "periodic:24", "sparse_set"]` |
| Location plus route cells | `["axis", "diagonal_2d", "gaussian_2d", "periodic:24", "sparse_set"]` |

## Leaves

| Parameter | Default | Notes |
| --- | --- | --- |
| `leaf_predictor` | `"constant"` | Accepts `"constant"` or `"linear"`. |
| `linear_leaf_features` | `None` | Python API currently expects stringified integer feature indices, such as `["0", "2"]`. |
| `l2_regularization` | `1.0` | Ridge penalty for linear leaves. |

Use linear leaves when the tree can find a region, lane, or time bucket but the
remaining residual trend inside that region is still approximately linear.

## Fuzzy Routing

| Parameter | Default | Notes |
| --- | --- | --- |
| `fuzzy` | `False` | Enables fractional branch assignment during training and weighted prediction recursion. |
| `fuzzy_bandwidth` | `0.0` | Split transition bandwidth. Must be finite and non-negative. |
| `fuzzy_kernel` | `"linear"` | Transition shape. Accepts `"linear"`, `"gaussian"`, `"exponential"`, `"bisquare"`, `"epanechnikov"`, or `"tricube"`. |

Fuzzy routing is not compatible with monotonic constraints.

Use fuzzy routing for temporal-spatial features where nearby values should not
change abruptly at a learned boundary. Set `fuzzy_bandwidth` in the same units
as the feature values, such as projected coordinate units or hours. Use
`fuzzy_kernel="linear"` for simple piecewise interpolation, `"gaussian"` or
`"tricube"` for smoother transitions, and compact-support kernels like
`"bisquare"` or `"epanechnikov"` when you want the blend to drop off faster
near the edge of the band.

## Monotonic Constraints

`monotonic_constraints` is a list of `-1`, `0`, or `1` values with one entry per
dense feature:

- `1` requires predictions to be non-decreasing in that feature.
- `-1` requires predictions to be non-increasing in that feature.
- `0` leaves the feature unconstrained.

Current constraints require constant leaves, non-fuzzy training, and axis-style
splitters.

## Native Extension Requirement

All training and prediction through `CartoBoostRegressor` uses the native
extension. PyPI installs include `cartoboost._native` for supported wheel
platforms. In a source checkout, build it before fitting or loading models:

```sh
uv run --group dev maturin develop
```
