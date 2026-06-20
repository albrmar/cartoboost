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

## Production Hybridization Boundary

The production `AutoForecaster` wrapper delegates to the Rust-native
`AutoForecastModel`. The current implementation validates
`cartoboost_lag`, `scaled_lag`, `delta_lag`, `scaled_delta_lag`,
`cartoboost_direct`, `cartoboost_rectified_recursive`, `log1p_scaled_lag`,
`lag_plus`, and `intermittent_demand` when the panel is sparse and
nonnegative, plus `classical_expert_bank`, protects the lag spine unless
another candidate clears the configured validation gain, selects a hard winner
when the gap is decisive, and uses bounded blends for close races.
`scaled_lag` applies a native per-series local standard scaler before fitting
lag features and inverts forecasts back to the original target scale.
`delta_lag` and `scaled_delta_lag` train the same global lag model on
delta-from-last targets and invert recursively, giving the auto path a
validation-gated trend/stationarity route. `cartoboost_direct` and
`cartoboost_rectified_recursive` fit horizon-specific native boosted models for
dense panels, with validation scoring fit to the validation horizon and final
fit bounded by the auto direct horizon. `log1p_scaled_lag` adds a
nonnegative-demand transform path by applying log1p before local standard
scaling and expm1-clamping forecasts back to the original scale.
For panel frames, the selector records both global validation scores and
series-level validation scores, then uses series-specific weights at prediction
time. That lets different pickup zones or route panels use different blends
without adding benchmark-specific routing.
`intermittent_demand` validates Croston, SBA, TSB, ADIDA, and all-zero methods
per series before refitting the selected sparse-demand method on the full
history.
When all training targets are nonnegative, the auto selector scores candidates
with nonnegative-clamped validation predictions and emits nonnegative final
forecasts, which keeps count and demand tasks from paying WAPE/RMSE penalties
for impossible negative tree outputs.
`lag_plus` fits the normal lag model, calibrates horizon-specific and
seasonal-bucket residual corrections on a held-out validation window, shrinks
those corrections by support, and disables them when they fail to improve the
configured validation objective. The current native objectives are RMSE and
WAPE.

The broader hybrid direction remains Rust-first. See
[Forecasting Hybridization Assessment](forecasting_hybridization_assessment.md)
for the next implementation map: target transforms, richer specialist experts,
reconciliation, probabilistic calibration, neural experts, and latency
constraints.

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
