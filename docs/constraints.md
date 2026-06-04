# Constraints

Monotonic constraints are an alpha feature for regression models that need
predictions to move in a declared direction as a dense feature increases.

## Monotonic Status

`monotonic_constraints` accepts one entry per dense feature:

| Value | Meaning |
| --- | --- |
| `1` | Prediction must be non-decreasing as the feature increases. |
| `-1` | Prediction must be non-increasing as the feature increases. |
| `0` | Feature is unconstrained. |

Current support is intentionally narrow:

- Constant leaves only.
- Non-fuzzy training only.
- Axis-style splitters only, including histogram-axis splitters.
- Dense features only; sparse-set features do not have monotonic semantics.
- Regression only.

The Python estimator validates these constraints before training. Native model
artifacts preserve the configured constraint vector when present.

## Unsupported Combinations

The following combinations are outside the current alpha contract:

| Combination | Status |
| --- | --- |
| Monotonic constraints with fuzzy routing | Rejected. |
| Monotonic constraints with linear leaves | Rejected. |
| Monotonic constraints with diagonal, Gaussian/radial, periodic, or sparse-set splitters | Rejected. |
| Monotonic sparse-set constraints | Not defined. |
| Monotonic interaction constraints | Not implemented. |

## Modeling Guidance

Use constraints only when the monotonic direction is part of the data contract,
not just a visual preference. For spatial experiments, avoid applying monotonic
constraints directly to latitude/longitude unless the target truly has a
one-directional geographic relationship. Spatial effects are usually better
represented by spatial splitters, blocked evaluation, and residual diagnostics.

Report constrained experiments separately from unconstrained baselines because
constraints intentionally trade model flexibility for shape guarantees.

## Validation Guidance

Constraint validation should include more than aggregate RMSE:

- Probe rows that differ only in the constrained feature.
- Check both increasing and decreasing directions when both are configured.
- Include tied or nearly tied feature values.
- Include a blocked holdout if the constrained feature is correlated with
  geography, route, or time.

These checks are especially important because a model can satisfy aggregate
quality thresholds while violating local shape expectations.
