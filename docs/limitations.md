# Limitations

GeoBoost is a regression package with a temporal-spatial modeling focus. It is
not a universal replacement for XGBoost, LightGBM, scikit-learn, or production
geospatial systems. Treat those tools as baselines and choose GeoBoost when its
splitters match the structure in your data.

## Modeling Scope

- Regression only; classification, ranking, and survival objectives are not
  supported.
- L2 and quantile regression are the public objectives.
- GeoBoost does not claim equivalence to Lyft's proprietary GeoBoost system.
- GeoBoost should not be described as generally superior to other boosters.
  Compare on the same data, split strategy, features, and metrics.

## Backend Scope

- Advanced splitters, fuzzy training, list-valued sparse features,
  schema-driven training, and linear leaves require the Rust native backend.
- The pure-Python fallback is limited to dense axis-split constant-leaf
  regression.

## Data Scope

- Python sparse features require non-negative integer IDs.
- The CLI accepts dense numeric CSV workflows only.
- Missing values are not handled beyond current finite-value validation.
- pandas inputs are accepted through generic estimator handling; there is no
  pandas-specific schema API beyond feature-name capture where available.

## Temporal-Spatial Scope

- Periodic features require the caller to provide the period, such as `24` for
  hour-of-day.
- Diagonal and Gaussian/radial splitters operate on numeric feature pairs; the
  current schema does not declare named coordinate pairs.
- H3, S2, grid, zone, or route-cell IDs must be precomputed by the caller and
  passed as sparse integer IDs.
- Fuzzy bandwidth is on the scale of your input features, so coordinate system
  and units matter.

## Artifact Scope

- GeoBoost JSON model and weights artifacts are the recommended save/load path.
- ONNX export supports only dense axis-tree constant-leaf models.
- Unknown future artifact versions fail clearly instead of attempting an
  implicit migration.

## Benchmark Scope

- Benchmark reports are setup-specific evidence, not universal superiority
  claims.
- LightGBM and XGBoost are optional benchmark dependencies, not runtime
  dependencies of GeoBoost.
- Any cross-package comparison should state the command, data version, feature
  handling, estimator settings, dependency versions, and holdout strategy.
