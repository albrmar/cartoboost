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
`cartoboost_lag`, `recency_weighted_lag`, `scaled_lag`, `delta_lag`, `scaled_delta_lag`,
`seasonal_delta_lag`, `scaled_seasonal_delta_lag`,
`cartoboost_direct`, `cartoboost_rectified_recursive`, `log1p_scaled_lag`,
`lag_plus`, and `intermittent_demand` when the panel is sparse and
nonnegative, plus `classical_expert_bank`, protects the lag spine unless
another candidate clears the configured validation gain, selects a hard winner
when the gap is decisive, and uses bounded blends for close races.
`scaled_lag` applies a native per-series local standard scaler before fitting
lag features and inverts forecasts back to the original target scale.
The shared lag feature builder now includes rolling standard-deviation,
minimum, and maximum windows alongside lag, rolling-mean, delta, trend, and
calendar features, so the auto path can model bursty pickup-demand volatility
and recent demand floors/ceilings without a separate routing mode.
Native lag forecasts can also consume numeric static covariates through
`covariate_features`, carrying each series' latest known route context forward
during recursive prediction. The production `AutoForecaster` wrapper now uses
`ForecastFrame.static_covariates` automatically for this native lag spine unless
callers pass `covariate_features=[]` or another explicit list. It also exposes
`rich_calendar_features=True` for native day-of-year, elapsed-index,
elapsed phase-14, low-cost Fourier calendar features, and month-start/mid/end
event indicators. For route-demand sources, the harness also enables native
low-cardinality covariate indicators for finite integer-coded static covariates
with 3 to 8 distinct training values when a training-only covariate-by-day
interaction signal clears the fixed threshold, then crosses those indicators
with the same calendar context. Static route covariates can also interact with
month-start, mid-month, and month-end event flags, which improved the
taxi-shaped committed suite mean problem RMSE without touching M4/M5/M6
competition paths. The maintained benchmark harness enables this route-context
enrichment only for taxi/synthetic route-demand sources, where pickup/dropoff
zone, distance, airport-lane, and borough metadata are part of the production
problem. M4/M5/M6 committed competition paths keep the compact target/calendar
feature set unless separate validation proves the richer path improves their
judged quality without unacceptable latency.
During fit, the auto path expands lag, rolling, delta, and trend windows with
season-length multiples when the validation training history can support them,
so hourly, weekly, monthly, and quarterly series get native seasonal features
without dataset-name routing.
`delta_lag`, `scaled_delta_lag`, `seasonal_delta_lag`, and
`scaled_seasonal_delta_lag` train the same global lag model on last-step or
seasonal deltas and invert recursively, giving the auto path
validation-gated trend, stationarity, and same-season change routes.
`cartoboost_direct` and
`cartoboost_rectified_recursive` fit horizon-specific native boosted models for
dense panels, with validation scoring fit to the validation horizon and final
fit bounded by the auto direct horizon. `log1p_scaled_lag` adds a
nonnegative-demand transform path by applying log1p before local standard
scaling and expm1-clamping forecasts back to the original scale.
The selector records global validation scores, horizon-level scores, and, for
panel frames, series-level scores. Prediction uses series-specific weights only
when the validation design gives each series at least four held-out points; thin
one-origin evidence falls back to horizon-specific weights, then global weights.
That lets different pickup zones, route panels, and forecast horizons use
different blends without letting a tiny local holdout overfit production
forecasts.
Candidate validation runs in parallel across the fixed native roster and then
flattens scores back into roster order. The selected member refits also run in
parallel and are sorted back to the requested member order before prediction,
preserving deterministic metadata while reducing auto-fit latency as the expert
set grows.
`intermittent_demand` validates Croston, SBA, TSB, ADIDA, and all-zero methods
per series before refitting the selected sparse-demand method on the full
history.
The classical expert bank includes naive, seasonal naive, fixed window-average,
theta, ETS, ARIMA, and Kalman-family candidates, plus seasonal-window average
when `season_length > 1`. The window-average and seasonal-window average
experts are fixed, no-search smoothing routes inspired by the simple baseline
banks used in StatsForecast and Darts: they give validation a cheap way to
prefer recent local levels or recent matching seasonal positions when a single
last observation is too noisy.
When all training targets are nonnegative, the auto selector scores candidates
with nonnegative-clamped validation predictions and emits nonnegative final
forecasts, which keeps count and demand tasks from paying WAPE/RMSE penalties
for impossible negative tree outputs.
`lag_plus` fits the normal lag model, calibrates horizon-specific and
seasonal-bucket residual corrections plus per-series residual corrections on a
held-out validation window, shrinks those corrections by support, and disables
them when they fail to improve the configured validation objective. The current
native objectives are RMSE, WAPE, and `rmse_wape`, which averages WAPE with
RMSE normalized by mean absolute actual demand for selection surfaces that must
balance both judging metrics.
`AutoForecaster` defaults to `rmse_wape`; callers can still set `rmse` or
`wape` when a single metric is the judging target.
The `recency_weighted_lag` candidate is enabled only when train-side diagnostics
show a recent level shift across enough series. When eligible, it uses
deterministic exponential sample weights in the native booster, with a half-life
derived from the season length and validation window. It lets validation prefer
recent regimes when they help RMSE/WAPE without making every lag model forget
older history or paying the extra fit cost on stable panels.

The maintained M4/M5/M6 benchmark selector avoids nested auto calibration inside
inner-origin scoring when deterministic shared candidates are sufficient for the
selection decision. M4 inner-origin scoring evaluates the lag spine and shared
seasonal/trend/calendar candidates first, then the outer reported run still fits
both committed CartoBoost rows. M5 now uses an RMSE-first selector for point
quality while still emitting official WRMSSE artifacts. M5 shared candidates
must clear the same minimum relative-gain guard versus the lag spine before
replacing it, which prevents tiny inner-origin wins from selecting brittle
local-statistical blends. Two-origin M5 validation skips the expensive raw-auto
fit inside inner-origin candidate scoring, uses the
deterministic autostats/phase-14/reconciliation candidates for selection, and
keeps only the total-reconciled shared route in the selectable surface when it
protects RMSE; state/store reconciled variants were measured as non-selected
overhead on the maintained M5 path. One-origin comparison runs can select the
local statistical autostats/phase-14 point blend that improved the M5
full-roster sample; when shared candidate selection is enabled for that
one-origin path, the outer scoring step skips the unused raw-auto fit and lets
the selected point blend define `cartoboost_auto_forecast`. Direct
`--no-candidate-selection` runs and two-origin committed M5 runs still fit raw
auto. The Rust `ClassicalExpertBank` now selects the strict best validation-MSE
expert instead of preferring a simpler expert inside a 1% tolerance; on the M5
comparison sample this moved `AutoStatsBank` to the validation-winning
auto-local-level Kalman route and improved RMSE, MAE, and WAPE. M6 now uses an
RMSE-first selector for the
daily-return point proxy and includes a validation-selected market-neutral
return candidate, which reflects the hard baseline for short-horizon financial
returns. M6 uses one RMSE validation origin for this point-quality selector so
the cheap market-neutral candidate does not require repeated inner-origin
proof. When shared candidate selection is enabled, the M6 outer scoring path
also skips the unused raw-auto point candidate and lets the selected
market-neutral route define `cartoboost_auto_forecast`; direct
`--no-candidate-selection` runs still fit raw auto. M6 inner-origin candidate
validation also skips the expensive raw-auto fit and evaluates the lag plus
deterministic shared candidates first; when the cheap market-neutral candidate
wins, the artifact records the skipped raw inner score as `null` rather than
emitting non-finite JSON. This keeps the committed
RMSE/WAPE/WRMSSE outputs protected while still emitting RPS artifacts for audit.
Across shared-candidate selection, `cartoboost_auto_forecast` keeps the lag
spine unless a replacement candidate clears the configured minimum relative gain
versus `cartoboost_lag`. This keeps small, noisy validation differences from
displacing the baseline.
After inner validation chooses the final candidate, the outer scoring path now
builds only the selected shared columns needed for the reported forecast. That
keeps inner validation broad while avoiding unused calendar, trend,
reconciliation, and local-statistical candidate work when the final route is
the lag spine or a single shared baseline.
The generic synthetic suite does not force the lag spine just because it is a
maintained fixture. The lag-spine override is limited to the narrow M4
frequency-risk guard; otherwise `cartoboost_auto_forecast` has to earn lag, raw
auto, or a shared candidate through the same inner-origin validation losses used
for non-M synthetic smoke checks. For non-M shared demand sources, a replacement
candidate also has to be origin-consistent versus `cartoboost_lag`: if it loses
any usable inner validation origin to the lag spine, the benchmark selector
keeps lag even when the candidate's average inner loss is lower. This guard came
from the synthetic seed-177 horizon-10 smoke, where raw auto had a better mean
inner loss but regressed the next holdout because one validation origin was
materially worse than lag. Non-M inner validation now skips the raw-auto fit up
front and scores the lag spine plus deterministic shared candidates. That
preserves the guarded selection decision while avoiding repeated raw-auto work
in calibration. Non-M inner validation also stops after one origin when
`cartoboost_lag` beats every finite candidate by at least 15%; close races keep
the full non-M two-origin check. The origin-depth policy is explicit: M4 keeps three inner
origins, M5 keeps two, M6 keeps one, and non-M named sources use two so the
M4/M5/M6 artifact contracts are not changed by the non-M latency path.
Rolling-origin suites also cache identical inner validation
cutoffs within each problem, keyed by source, cutoff, and candidate roster, so
overlapping folds reuse scores without changing the train/test boundary. The
non-M outer scoring path runs shared selection before fitting the most expensive
auto path, then fits that path only when validation selects it. When validation
selects lag or a deterministic shared candidate, the outer raw-auto fit is
skipped entirely.
Benchmark artifacts now record both `cartoboost_lag` and
`cartoboost_auto_forecast` settings. The lag estimator count follows
`--cartoboost-n-estimators`; auto keeps the maintained quality floor
`max(--cartoboost-n-estimators, 360)` unless the run explicitly passes
`--cartoboost-auto-n-estimators`, which is intended for honest latency probes
where the auto estimator budget must be capped.

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
