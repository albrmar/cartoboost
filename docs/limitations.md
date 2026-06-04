# Limitations

GeoBoost is currently a strong alpha moving toward a v1 release candidate. The
implementation is useful for deterministic regression experiments and API
hardening, but it is not positioned as production-grade infrastructure.

## Product Scope

- Regression only; classification, ranking, and survival objectives are out of
  scope for the current public contract.
- L2 regression is the stable objective. Quantile work may exist in the
  development tree, but it is not part of the benchmark comparison contract
  until the Python/Rust/API path is fully validated.
- No claim of equivalence to Lyft's proprietary GeoBoost implementation.
- No claim of general superiority over LightGBM, XGBoost, scikit-learn, or
  production geospatial systems.

## Backend Scope

- Advanced splitters, fuzzy training, list-valued sparse features, schema-driven
  training, and linear leaves require the Rust native backend.
- The pure-Python fallback is intentionally limited to dense axis-split
  constant-leaf workflows.

## Data Scope

- Python sparse features require non-negative integer IDs.
- CLI v1 accepts dense numeric CSV workflows only.
- No missing-value policy beyond current finite-value validation.
- No native pandas-specific schema contract beyond generic iterable handling and
  feature-name capture where available.

## Schema Scope

- Current schema kinds are numeric, periodic, and sparse-set.
- Named spatial pairs and richer geospatial role declarations remain future
  hardening.

## Artifact Scope

- Artifact version `1` is supported.
- Optional metadata and training configuration restore public estimator params
  when present.
- There is no multi-version artifact migration framework yet.

## Validation Scope

- Validation fixtures are deterministic synthetic checks.
- They demonstrate intended behavior on narrow fixtures and do not certify
  production accuracy, latency, robustness, or superiority.
- NYC taxi quality benchmarks are optional real-data comparisons with documented
  setup and generated artifacts; they are not universal superiority claims.
- LightGBM and XGBoost are optional benchmark dependencies, not runtime
  dependencies of GeoBoost. Cross-package claims are valid only for the exact
  committed setup: estimator/depth settings, feature handling, dependency
  versions allowed by `pyproject.toml`, and the generated artifacts.
- Current NYC artifacts show that comparable target-mean zone features improve
  pickup-demand quality, but GeoBoost still misses all-task XGBoost speed parity
  in the regenerated 25k repeated report.
