# Constraints

Monotonic constraints force predictions to move in a declared direction as a
dense feature increases. They are useful when the direction is a modeling
requirement, such as price increasing with distance or demand decreasing with
travel time.

Use constraints as scientific assumptions, not as tuning decoration. A
constraint should be defensible before training and should survive checks on
held-out taxi trips, route groups, and time blocks.

## Usage

`monotonic_constraints` has one entry per dense feature:

| Value | Meaning |
| --- | --- |
| `1` | Prediction must be non-decreasing as the feature increases. |
| `-1` | Prediction must be non-increasing as the feature increases. |
| `0` | Feature is unconstrained. |

```python
model = CartoBoostRegressor(
    splitters=["axis"],
    monotonic_constraints=[1, 0, -1],
)
```

Current support:

- Constant leaves.
- Non-fuzzy training.
- Axis-style splitters, including histogram-axis splitters.
- Dense features only.
- Regression.

## Temporal-Spatial Guidance

Use constraints only when the direction is real, not just visually convenient.
Trip distance, elapsed time, toll amount, or known service-level features can be
good candidates. Latitude, longitude, zone ID, and taxi-zone IDs usually are
not: their relationship to the target is often local, discontinuous, or
directional only within a specific market.

For temporal-spatial effects, prefer spatial splitters, periodic splitters,
sparse taxi-zone features, blocked evaluation, and residual diagnostics unless
a monotonic rule is part of the problem definition.

## Validation

Check more than aggregate RMSE:

- Probe rows that differ only in the constrained feature.
- Check increasing and decreasing constraints separately.
- Include tied or nearly tied feature values.
- Use spatial or temporal holdouts when the constrained feature is correlated
  with cartography, route, or time.
