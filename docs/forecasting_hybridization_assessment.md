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
probabilistic calibration. A dominant CartoBoost default must compose these
deterministically instead of exposing them as disconnected features.

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
`scaled_lag`, `delta_lag`, `scaled_delta_lag`, `log1p_scaled_lag`,
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
Panel validation now emits both global and per-series scores, and prediction
uses per-series weights when available. That is the production equivalent of
local routing without dataset-name branches.
For nonnegative training frames, validation scoring and final auto predictions
are clamped nonnegative. That makes the selector honest for count/demand
metrics instead of letting impossible negative forecasts distort WAPE or RMSE.

`lag_plus` is now the first residual-correction spine: it fits the normal lag
model, predicts a held-out validation window, estimates horizon-specific and
seasonal-bucket mean residual corrections, shrinks them by support, disables
them if the configured validation objective does not improve, and then refits
the base lag model on the full frame. The current native objectives are RMSE
and WAPE.

The remaining gap is depth, not direction. The auto selector needs richer
target transforms, specialist experts, hierarchy-aware reconciliation, and
task-specific probabilistic calibration. More models alone would make this
worse; every added expert must pass through the guarded selector.

## Dominant Architecture

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
must come from the actual task: RMSE/WAPE for ordinary point forecasting,
MASE/sMAPE-style losses for heterogeneous M4-style data, WRMSSE for weighted
hierarchical demand, and RPS for ordinal rank forecasts.

## Implementation Priorities

1. Extend the Rust `AutoForecastModel` roster. The current implementation
   owns the validation split, objective, lag baseline displacement rule, blend
   bounds, and metadata for `cartoboost_lag`, `scaled_lag`,
   `delta_lag`, `scaled_delta_lag`, `log1p_scaled_lag`, `lag_plus`,
   dense-panel `cartoboost_direct` and `cartoboost_rectified_recursive`,
   sparse-panel `intermittent_demand`, and `classical_expert_bank`. Next, add
   decomposition and calibrated probabilistic candidates behind the same
   guardrails.

2. Strengthen `LagPlus`:
   base `CartoBoostLagForecaster`, horizon-specific residual correction, and
   reliability shrinkage are implemented. RMSE/WAPE-aware enablement is also
   implemented. Seasonal-bucket residual correction is implemented through the
   auto season length. Next, add additional official competition
   objectives.

3. Expand MLForecast-style target transforms in Rust:
   local mean/standard scaling, log1p for nonnegative targets, and exact
   inverse forecasts are implemented. Next, add first/seasonal differences,
   transform diagnostics, and validation-based transform enablement.

4. Expand specialist experts, but gate them:
   classical bank for short/local series and intermittent methods for sparse
   nonnegative demand are implemented behind the auto selector. Dense direct
   and rectified-recursive tree candidates are also implemented. Next, add MSTL
   remainder models for strong multi-seasonality and neural experts only when
   history length and validation support justify them.

5. Make calibration task-specific:
   WRMSSE-aware demand selection and reconciliation for hierarchy; RPS
   confusion-matrix calibration for rank forecasts; conformal intervals for
   uncertainty where point metrics are not enough.

6. Treat latency as a design constraint:
   grouped contiguous storage for feature generation, Rayon across series and
   candidates, stable reductions, no Python hot loops in fit/predict, and
   metadata for train time, prediction time, thread count, and peak memory.

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

## Non-Negotiables

- Production auto behavior must be Rust-first.
- Python may expose configuration, but cannot become the model selector.
- More experts are useful only after the guardrail selector is in place.
- Validation metrics must match the decision metric.
- A default auto model that loses to `cartoboost_lag` without explaining why is
  not acceptable.
