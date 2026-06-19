# Forecasting Reconciliation

CartoBoost keeps hierarchical forecasting mechanics in the Rust core. The
current reconciliation helpers represent the hierarchy as sparse bottom-series
membership rows instead of a dense summing matrix, so each aggregate node stores
only the bottom taxi series that roll up into it.

## Spatial Hierarchies

Use `HierarchySpec` to describe parent-child relationships such as all taxi
trips, pickup zones, and pickup-dropoff zone pairs. Bottom-level forecasts can
then be reconciled back to every aggregate level.

Supported reconciliation methods:

- Bottom-up: trust bottom pickup-dropoff forecasts and aggregate them upward.
- Top-down: trust the root total and distribute it to bottom series by positive
  bottom forecast proportions.
- Middle-out: trust a selected hierarchy level, distribute each middle node to
  its bottom descendants, then aggregate upward.
- OLS: project all base forecasts into the coherent hierarchy using identity
  forecast error weights.
- WLS: project with user-provided positive forecast error variances.
- MinT shrink: estimate the residual covariance from validation residuals,
  shrink off-diagonal covariance toward a diagonal target, and project with the
  resulting precision matrix.

The OLS/WLS/MinT implementations solve bottom-dimensional normal equations
from sparse hierarchy rows. They do not materialize a full dense summing matrix.

## Temporal Hierarchies

`TemporalHierarchy` aggregates a base time series into nested, non-overlapping
windows. For hourly taxi demand, common levels are `hour`, `six_hour`, and
`day`. Aggregation factors must be positive, strictly increasing, and nested
multiples of the previous level. Series lengths must divide exactly by each
factor so partial windows cannot silently affect validation.

## Validation Notes

Reconciliation helpers validate finite inputs, matching horizon lengths,
acyclic hierarchy structure, missing parents, and duplicate node ids. Temporal
aggregation rejects empty series, non-finite values, non-nested factors, and
partial windows.
