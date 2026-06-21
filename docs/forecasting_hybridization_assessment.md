# Forecasting Hybridization Assessment

This assessment is about production forecasting quality, not committed
benchmark narration. It compares CartoBoost's current forecasting shape with
the implementation patterns used by leading open-source forecasting libraries
and translates those patterns into a Rust-first CartoBoost roadmap.

## What Strong Forecasting Repos Do Better

The strongest repos do not rely on one clever model.

- StatsForecast uses per-series statistical specialists, compact grouped data,
  explicit fallback models, parallel execution, cross-validation, and conformal
  intervals. Its strength is robust local behavior on short, noisy, seasonal,
  and intermittent series.
- MLForecast uses a production tabular pipeline: grouped contiguous arrays,
  lag transforms, date features, target differencing, local scaling, recursive
  and direct horizon handling, and conformal calibration. Its strength is fast
  global feature generation for many related series.
- NeuralForecast wins when long context and nonlinear seasonality matter by
  making window sampling, input scaling, validation size, loss choice, and
  architecture family explicit. Its lesson is not "always use neural"; it is
  "use neural experts behind a validation gate."
- Darts is strong as a product surface because it treats backtesting,
  covariates, hierarchy, reconciliation, probabilistic forecasts, and ensembles
  as first-class workflows.
- GluonTS is strong on probabilistic rigor because dataset transformations,
  instance splitters, scaling, dynamic/static features, and evaluator metrics
  are central rather than bolted on after point forecasts.

The common pattern is hybridization: local statistical experts, global tabular
models, target transforms, validation-calibrated selection, reconciliation, and
probabilistic calibration. CartoBoost exposes these capabilities through a
deterministic Rust-first stack instead of treating them as disconnected
features.

## Current CartoBoost Gap

CartoBoost already has many of the right Rust modules:

- `classical_bank`, `autostats`, `intermittent`, `direct`, `global`,
  `decomposition`, `mstl`, `gating`, `ensemble`, `reconciliation`,
  `rank_probability`, `probabilistic`, and `lag_features`.
- `CartoBoostLagForecaster` is a real Rust global lag model with recursive
  prediction and Rayon-parallel panel prediction.
- `ClassicalExpertBank` provides deterministic native validation across
  classical experts.
- `RuleBasedGating` and native ensembles provide a compositional base.

The first production gap has been closed: `AutoForecaster` now delegates to a
Rust-native `AutoForecastModel` that validates `cartoboost_lag`, `lag_plus`,
`scaled_lag`, `delta_lag`, `scaled_delta_lag`, `seasonal_delta_lag`,
`scaled_seasonal_delta_lag`, `log1p_scaled_lag`,
sparse-panel `intermittent_demand`, dense-panel `cartoboost_direct` and
`cartoboost_rectified_recursive`, and the native `classical_expert_bank`, keeps
lag as the baseline unless another candidate clears a configured displacement
gain, uses hard-winner selection for decisive validation gaps, and bounds
close-race blends.

`scaled_lag` and `log1p_scaled_lag` are the first target-transform candidates.
`scaled_lag` fits a native per-series local standard scaler, trains the lag
forecaster on transformed targets, and inverts forecasts back to the original
target scale. `log1p_scaled_lag` is eligible only for nonnegative targets; it
applies log1p before local standard scaling and expm1-clamps forecasts back to
the original scale. This brings two important MLForecast-style production
patterns into the Rust auto path without moving model behavior into Python.
Sparse nonnegative panels also receive an `intermittent_demand` candidate that
selects Croston, SBA, TSB, ADIDA, or all-zero behavior per series using the
same validation objective.
Dense panels also receive direct and rectified-recursive candidates, so the
auto path can choose horizon-specific boosted models when recursive lag
rollouts are the wrong bias.
Validation now emits global, horizon-level, and per-series scores, and
prediction uses series-specific weights first, horizon-specific weights second,
and global weights as the fallback. That is the production equivalent of local
and horizon-aware routing without dataset-name branches.
Candidate validation is parallelized across the fixed Rust roster and collected
back in deterministic roster order, so adding useful experts does not force a
linear auto-fit latency penalty.
The shared lag spine also carries rolling standard-deviation, minimum, and
maximum features next to rolling means, optional partial rolling means, deltas,
trends, and calendar features, giving every lag-based candidate direct signals
for bursty demand, instability, and recent local demand floors/ceilings.
The auto path also expands the shared lag spine with supported season-length
multiples during fit, which is the same structural lesson used by strong
competition systems: encode known frequency and let validation decide whether
the resulting candidate should displace or blend with the baseline.
For nonnegative training frames, validation scoring and final auto predictions
are clamped nonnegative. That makes the selector honest for count/demand
metrics instead of letting impossible negative forecasts distort WAPE or RMSE.

`lag_plus` is now the first residual-correction spine: it fits the normal lag
model, predicts a held-out validation window, estimates horizon-specific and
seasonal-bucket mean residual corrections plus per-series residual
corrections, shrinks them by support, disables them if the configured
validation objective does not improve, and then refits the base lag model on
the full frame. The current native objectives are RMSE, WAPE, and a blended
`rmse_wape` objective that averages WAPE with normalized RMSE. The auto path
uses `rmse_wape` by default so ordinary point-forecasting selection is not
biased toward RMSE while ignoring absolute percentage demand error.

The remaining gap is depth, not direction. The auto selector needs richer
target transforms, specialist experts, hierarchy-aware reconciliation, and
task-specific probabilistic calibration. More models alone would make this
worse; every added expert must pass through the guarded selector.

## Recommended Architecture

The default production stack should be:

```text
Grouped Rust panel store
  -> target transforms and scale diagnostics
  -> LagPlus residual-correction spine
  -> specialist expert bank
  -> metric-aware validation table
  -> baseline guardrail or hard winner
  -> bounded blend only for close races
  -> reconciliation / probabilistic calibration when structure requires it
```

The first rule is that `cartoboost_lag` remains the spine until validation
proves another route improves the target metric. That prevents the auto path
from losing to its simplest internal baseline. The second rule is that metrics
must come from the actual task: RMSE, WAPE, or `rmse_wape` for ordinary point forecasting,
MASE/sMAPE-style losses for heterogeneous M4-style data, WRMSSE for weighted
hierarchical demand, and RPS for ordinal rank forecasts.

## Development Direction

The production path is to keep extending the Rust `AutoForecastModel` roster
behind the same validation and baseline-protection rules. Current candidates
cover the lag spine, target-scaled lag variants, delta routes, direct and
rectified-recursive tree routes, nonnegative log scaling, `lag_plus`,
intermittent demand methods, and the classical expert bank.

Future additions should keep the same user-visible contract: deterministic
validation, task-appropriate metrics, bounded blends for close races, and clear
metadata explaining which candidate was used. Useful additions include
decomposition candidates for strong multi-seasonality, additional target
transforms, hierarchy-aware reconciliation, calibrated probabilistic outputs,
and neural experts only when history length and validation support justify
them.

Latency remains part of the public contract. Forecasting improvements should
continue to use grouped Rust data structures, Rayon parallelism, stable
reductions, and benchmark metadata for train time, prediction time, thread
count, and peak memory.

## Immediate Code Direction

`RuleBasedGating` and `AutoForecastModel` now support the selection behavior
needed for the auto architecture:

- hard winner selection when validation loss has a decisive relative gap;
- baseline behavior until another expert clears a configured displacement gain;
- bounded blends for close races so weak experts cannot dominate a stable
  spine.

This is intentionally generic. The same primitive can protect lag on ordinary
demand forecasts, protect WRMSSE winners on hierarchy-aware demand, or protect
RPS-calibrated routes in ordinal financial forecasts without dataset-name
branches or hyperparameter search.

For production users, the practical rule is simple: compare models on the
metric that matches the decision being made, then inspect the auto-forecast
metadata to see whether CartoBoost chose the lag baseline, a specialist
candidate, or a bounded blend.
