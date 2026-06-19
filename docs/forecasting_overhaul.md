# Forecasting Overhaul

CartoBoost forecasting now exposes a Rust-first architecture for deterministic
multi-horizon forecasting experiments. Benchmark-critical behavior belongs in
native Rust modules; Python APIs stay thin wrappers and ergonomic helpers.

## Public Surface

- `CartoBoostDirectForecaster` trains native horizon-specific boosted models
  using leakage-safe lag features.
- `STLCartoBoostForecaster` and `MSTLCartoBoostForecaster` decompose series,
  fit native remainder forecasters, and recompose predictions.
- `AutoStatsBank` and `ClassicalExpertBank` provide deterministic classical
  experts over the existing native local model stack.
- `Reconciler` supports bottom-up, top-down, middle-out, OLS, WLS, and
  MinT-shrink style reconciliation over sparse bottom memberships.
- Quantile, conformal, WRMSSE, RPS, M6, and portfolio helpers are native Rust
  APIs with thin Python metric wrappers.
- `cartoboost_neural::forecasting` exposes deterministic CPU N-BEATS and
  N-HiTS style experts.

## Determinism Rules

The forecasting stack uses fixed model menus, stable iteration order, and
rolling-origin validation. No hyperopt, random search, benchmark-specific
tuning branches, or dataset-name-specific routing are part of the default
surface.

## Benchmark Boundary

Committed benchmark commands are available, but quality claims require real
artifacts with dataset hash, split policy, seed, model settings, metric
definition, wall time, and the full metric table. Proxy metrics must be labeled
as proxy metrics, especially for M6-style financial tasks.
