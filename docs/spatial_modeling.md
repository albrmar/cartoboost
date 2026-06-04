# Spatial Modeling

GeoBoost's spatial modeling surface is a clean-room, GeoBoost-inspired alpha
implementation for regression experiments. It supports spatially useful
splitters and sparse route-cell features, but it does not include proprietary
Lyft GeoBoost behavior, H3-native feature engineering, or production geospatial
validation infrastructure.

## Spatial Feature Paths

| Path | Status |
| --- | --- |
| Dense latitude/longitude or projected coordinates | Supported as numeric columns. |
| Diagonal 2D splitters | Rust native splitter for oblique 2D boundaries. |
| Gaussian/radial 2D splitters | Rust native splitter for radial neighborhoods. |
| Periodic splitters | Rust native splitter for wraparound features such as hour-of-day. |
| Sparse-set route cells | Rust native list-valued integer-ID sparse feature path. |
| H3 cells | Supported only when pre-encoded as non-negative sparse integer IDs. |
| Named spatial schemas | Future hardening; current schema metadata does not declare named coordinate pairs. |

Dense spatial splitters operate over numeric columns. Feature schema metadata can
mark numeric, periodic, and sparse-set roles, but richer geospatial declarations
such as "pickup latitude/pickup longitude" pairs remain outside the current
schema contract.

## H3 Sparse Scaffold

The sparse-set API can represent H3-like cell membership if callers provide
pre-encoded non-negative integer IDs:

```python
model.fit(
    X_dense,
    y,
    sparse_sets={"pickup_h3": [[617700169957507071], [617700169957507583]]},
)
```

This is a sparse integer-ID scaffold, not an H3 integration. GeoBoost does not
currently:

- Convert latitude/longitude to H3 cells.
- Validate H3 resolutions or parent/child relationships.
- Expand neighbor rings.
- Apply H3 distance, hierarchy, compacting, or polygon coverage logic.
- Serialize H3 metadata beyond the generic sparse-set feature name and IDs used
  by trained splits.

Use the sparse-set docs for routing semantics: a sparse split routes left when a
row contains any split ID, and right otherwise.

## Fuzzy Routing

Fuzzy routing is implemented in the Rust backend as fractional branch assignment
around supported split boundaries. Training uses fractional child weights, and
prediction uses weighted branch recursion.

Current status:

- Requires the Rust backend.
- Controlled by `fuzzy=True` and non-negative `fuzzy_bandwidth`.
- Not compatible with monotonic constraints.
- Preserved by native model artifacts.
- Covered by deterministic routing and acceptance checks.

Fuzzy routing should be reported as a modeling choice, not as a default spatial
smoothing guarantee. The bandwidth is data-scale dependent, so experiments
should state the coordinate system and units used by the fuzzy boundary.

## Spatial Diagnostics

Spatial diagnostics should separate global model quality from spatial failure
modes. Recommended diagnostics include:

- Random holdout metrics for generalization under IID-style splits.
- Spatially blocked holdout metrics for generalization to withheld locations or
  zones.
- Residual summaries by route cell, pickup zone, grid cell, or other stable
  geography.
- Residual plots or maps when generated artifacts are part of the report.
- Comparison against axis-only baselines to isolate the value of spatial
  splitters.
- Comparison against sparse-only or dense-only variants when both feature paths
  are available.

Existing benchmark scripts produce examples of random versus spatial holdout
metrics and zone residual plots. Those artifacts are diagnostics for the exact
script setup, not universal production claims.

## Blocked Evaluation Metrics

Blocked evaluation means the test set withholds correlated groups instead of
sampling rows independently. For spatial work, blocks can be pickup zones,
delivery regions, route corridors, grid cells, H3 cells, or time windows.

Report blocked metrics alongside random metrics:

| Metric | Purpose |
| --- | --- |
| RMSE | Penalizes large residuals and is the main quality metric in current scripts. |
| MAE | Easier to interpret for typical absolute error. |
| R2 | Explains variance relative to the target distribution for the same split. |
| Block/random ratio | Shows whether spatial holdout quality degrades more than random holdout quality. |
| Residual-by-block summary | Identifies localized failure modes hidden by aggregate scores. |

Blocked evaluation is currently a benchmark and validation practice, not a
first-class `GeoBoostRegressor` cross-validation API. Keep blocked metric claims
tied to the script, data version, feature set, estimator settings, and committed
artifact that produced them.

## Linear Leaves And Elastic Net

Rust linear leaves fit weighted ridge residual models over configured dense
features. This is useful when a local residual trend is approximately linear.

Elastic-net linear leaves are not implemented. The current public knob is
`l2_regularization`, which is a ridge penalty. Do not document L1 or elastic-net
selection behavior unless a future implementation adds explicit parameters,
artifact fields, and tests.
