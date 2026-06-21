# Forecasting Tool Benchmark

For the deterministic forecasting architecture, see
[Forecasting Overhaul](../forecasting_overhaul.md). The benchmark command
surface accepts fixed no-hyperopt runs:

```sh
python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt
python scripts/forecasting_m4.py --committed --no-hyperopt
python scripts/forecasting_m5.py --committed --official-wrmsse --no-hyperopt
python scripts/forecasting_m6.py --committed --official-style --no-hyperopt
python scripts/forecasting_generalization.py --compact --no-hyperopt
```

The commands above do not imply a benchmark win by themselves. Public claims
must cite the artifact, split, seed, model roster, and metric table. For M5 and
M6 claims, cite `official_metrics` in addition to the shared point metrics.

## Artifact Metric Contract

New benchmark artifacts written by the harness keep provenance fields at the top level:
`git_commit`, `dataset_hash`, `source_file_hashes`, `benchmark_integrity`, and
`resource_usage`. Shared point metrics live under `metrics`: RMSE, MAE, MASE,
WAPE, SMAPE, and bias. M5 and M6 artifacts also include `official_metrics`:

| Source | Primary field | Contents | Claim boundary |
| --- | --- | --- | --- |
| `m5` | `official_metrics.m5` | Level-aware WRMSSE by total, state, store, item, and item-store levels, plus model rankings and per-series breakdowns. | Uses one-step RMSSE scaling and recent unit-sales volume weights because sell prices are not present in the shared benchmark frame. |
| `m6` | `official_metrics.m6` | Five-bucket rank-probability scores, training-window calibration metadata, per-asset probability rows, deterministic long/short decision rows, and model rankings. | Still an audit proxy, not an official M6 submission file. |

## Forecasting Overhaul Committed Run

Run dates: June 19-20, 2026. These runs used the deterministic CartoBoost-only
roster, fixed seed 42, no hyperopt, and the committed benchmark sample settings.
Artifacts are committed under `docs/assets/nyc_taxi_benchmarks/`; use each
artifact's provenance fields for the exact command and git commit.

| Suite | Command | Artifact | Result |
| --- | --- | --- | --- |
| Real NYC taxi lane demand | `uv run --group dev python scripts/forecasting_library_benchmark.py --source nyc-taxi --year 2024 --months 1 --taxi-type yellow --lanes 24 --horizon 7 --no-download --no-hyperopt --model-roster cartoboost --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots` | `forecasting_library_benchmark_real.json` | `cartoboost_auto_forecast` beat `cartoboost_lag`: RMSE 39.033944, MAE 29.172619, WAPE 0.093742. |
| Synthetic committed suite | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --model-roster cartoboost --no-candidate-selection --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite.json` | `forecasting_overhaul_committed_suite.json` | `cartoboost_auto_forecast` ranked first with mean RMSE ratio 1.000000 and 4/4 wins-or-ties; `cartoboost_lag` had mean RMSE ratio 1.000000. |
| Synthetic committed suite, scalable external roster | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --model-roster scalable --no-candidate-selection --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite_scalable_roster.json` | `forecasting_overhaul_committed_suite_scalable_roster.json` | `cartoboost_auto_forecast` and `cartoboost_lag` ranked first at mean RMSE ratio 1.013744; `lightgbm_lag` was 1.279238. |
| Synthetic committed suite, full external roster | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite_full_roster.json` | `forecasting_overhaul_committed_suite_full_roster.json` | Older maintained full-roster artifact: `lightgbm_lag` ranked first by mean RMSE ratio at 1.069525. A refresh was attempted after the gated-indicator change but stopped in `statsforecast_autotbats`/ARIMA optimization before producing a new artifact. |
| Non-M scalable generalization check | `uv run --group dev python scripts/forecasting_generalization.py --compact --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_generalization_scalable_synthetic.json` | `forecasting_generalization_scalable_synthetic.json` | `cartoboost_auto_forecast` and `cartoboost_lag` tied for first at mean RMSE ratio 1.000000; `lightgbm_lag` was 1.196396 and `xgboost_lag` was 1.258816. |
| M4 committed sample | `uv run --group dev python scripts/forecasting_m4.py --committed --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m4_committed.json` | `forecasting_overhaul_m4_committed.json` | `cartoboost_auto_forecast` ranked first with mean RMSE ratio 1.000000 and 6/6 wins-or-ties; `cartoboost_lag` had mean RMSE ratio 12.104570. |
| M5 committed sample | `uv run --group dev --group bench python scripts/forecasting_m5.py --committed --official-wrmsse --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` | `forecasting_overhaul_m5_committed.json` | `cartoboost_auto_forecast` beat `cartoboost_lag`: RMSE 2.415225, MAE 1.139285, WAPE 0.910615, WRMSSE 0.568942. |
| M6 committed sample | `uv run --group dev --group bench python scripts/forecasting_m6.py --committed --official-style --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` | `forecasting_overhaul_m6_committed.json` | `cartoboost_auto_forecast` selected the market-neutral return candidate and beat `cartoboost_lag` on point quality: RMSE 0.013439, MAE 0.007342, WAPE 1.000000. Calibrated RPS is audit-only here: auto 0.208171 versus lag 0.200754. |

Interpretation: the protected auto route now uses the lag spine when validation
does not justify a riskier candidate. On the committed CartoBoost-only samples,
`cartoboost_auto_forecast` ties `cartoboost_lag` on the synthetic suite, beats
or ties every M4 group, beats lag on M5 WRMSSE, and beats lag on M6 point
metrics. This is not a blanket external-library win claim: the maintained M5
full-roster sample is now led by CartoBoost auto on RMSE, MAE, and WAPE, but
`statsforecast_autotbats` still leads WRMSSE. The maintained M6 full-roster
sample is now led by CartoBoost auto on RMSE, MAE, and WAPE, but simple
seasonal-naive baselines still lead calibrated RPS.

## Current Benchmark Status

| Artifact | Refresh status | Current read |
| --- | --- | --- |
| `forecasting_overhaul_committed_suite.json` | Refreshed after training-gated low-cardinality covariate indicators, native route-event interactions, no-selector non-M scoring, native covariate-calendar interactions, and the raw-auto lag-spine guard | Protected `cartoboost_auto_forecast` and `cartoboost_lag` tie on the CartoBoost-only synthetic suite at mean RMSE ratio 1.000000. |
| `forecasting_overhaul_committed_suite_scalable_roster.json` | Added after the full external roster proved too slow in `statsforecast_autotbats`/ARIMA optimization; refreshed with no-selector non-M scoring | Scalable external-roster synthetic committed run with CartoBoost auto/lag, LightGBM, XGBoost, and functime baselines. CartoBoost auto and lag rank first at mean RMSE ratio 1.013744; `lightgbm_lag` is 1.279238. |
| `forecasting_generalization_scalable_synthetic.json` | Refreshed after training-gated native low-cardinality covariate indicators, native route-event interactions, no-selector non-M scoring, native covariate-calendar interactions, rich Fourier/event calendar features, and the raw-auto lag-spine guard | Compact scalable synthetic run with CartoBoost auto/lag, LightGBM, XGBoost, and functime baselines. CartoBoost auto and lag tie for first by mean RMSE ratio at 1.000000. |
| `forecasting_overhaul_m4_committed.json` | Refreshed with provenance fields | Protected auto wins/ties all 6 M4 groups, with mean RMSE ratio 1.000000 versus 12.104570 for lag. |
| `forecasting_overhaul_m5_committed.json` | Refreshed after M5 point-quality selector changes | Protected auto beats lag on point metrics and WRMSSE 0.568942 versus 0.743721. |
| `forecasting_overhaul_m6_committed.json` | Refreshed after M6 point-quality selector update | Protected auto selects the market-neutral return candidate and beats lag on point metrics. Calibrated RPS is audit-only on this point-quality route: auto 0.208171 versus lag 0.200754. |
| `forecasting_overhaul_committed_suite_full_roster.json` | Older maintained full external-roster artifact; refresh attempted after the gated-indicator change but stopped in `statsforecast_autotbats`/ARIMA optimization before producing a new artifact | `lightgbm_lag` remains the maintained winner at mean RMSE ratio 1.069525; CartoBoost auto is 1.187837 in the older full-roster artifact. Use the scalable-roster artifact for the current external-tree/functime read. |
| `forecasting_m5_full_roster_sample.json` | Refreshed after strict best-validation selection in the Rust classical expert bank | CartoBoost auto wins point quality: RMSE 2.511292, MAE 1.135585, WAPE 0.916059. `statsforecast_autoets` is second by RMSE at 2.525734. `statsforecast_autotbats` still leads official WRMSSE at 0.618397; CartoBoost auto WRMSSE is 0.669928. |
| `forecasting_m5_full.json` | Maintained full-corpus coverage artifact | CartoBoost lag completed the 30,490-series fast run with RMSE 2.634879; this is coverage evidence, not an external bakeoff. |
| `forecasting_m6_full.json` | Refreshed after M6 point-quality selector update | CartoBoost auto wins point quality: RMSE 0.013392, MAE 0.007357, WAPE 1.000000. `statsforecast_autoarima` is second by RMSE at 0.013402. Seasonal-naive baselines still lead calibrated RPS at 0.192195; CartoBoost auto RPS is 0.206007. |

### Current Committed-Run Latency Check

The June 20, 2026 reruns refreshed the CartoBoost-only committed artifacts after
native auto-fit and benchmark selector changes. The maintained non-M synthetic
artifacts now use no-selector scoring because raw native auto and the lag spine
produce identical quality on these route-demand checks, and this avoids
unneeded calibration work. The selector machinery remains available for
explicit experiments and for M paths where the maintained artifacts still need
candidate selection. M5 and M6 latency improved after removing redundant nested
auto calibration from the M5/M6 selector paths.

| Artifact | Previous runtime | Current runtime | Runtime delta | Quality read |
| --- | ---: | ---: | ---: | --- |
| `forecasting_overhaul_committed_suite.json` | 69.808s | 87.170s | +24.9% | Auto and lag still tie with mean RMSE ratio 1.000000. Event-flag route interactions improved mean problem RMSE to 0.687513 and WAPE to 0.016942; no-selector non-M scoring avoids calibration work that did not change quality on this artifact. |
| `forecasting_overhaul_m4_committed.json` | 1624.469s | 648.057s | -60.1% | No metric regression; auto remains 1.000000 mean RMSE ratio. |
| `forecasting_overhaul_m5_committed.json` | 180.293s | 81.262s | -54.9% | No metric regression; auto remains RMSE 2.415225 and WRMSSE 0.568942 with two selector origins. |
| `forecasting_overhaul_m6_committed.json` | 127.041s | 22.316s | -82.4% | Auto keeps RMSE 0.013439 and WAPE 1.000000 after selecting the market-neutral return candidate; the selected route skips the unused outer raw-auto fit, and RPS is audit-only at 0.208171. |

## Competition Results Snapshot

| Competition | Artifact | Details | Result |
| --- | --- | --- | --- |
| M5 committed CartoBoost sample | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` | Committed M5-style sample, CartoBoost-only roster with auto and lag | `cartoboost_auto_forecast` beat `cartoboost_lag`: RMSE 2.415225, MAE 1.139285, WAPE 0.910615, WRMSSE 0.568942. |
| M5 Forecasting Accuracy comparison sample | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` | 100 item-store sample, 90 recent days, 28-day holdout, full 14-model roster with fixed no-hyperopt candidate selection | CartoBoost auto won point quality: RMSE 2.511292, MAE 1.135585, WAPE 0.916059. `statsforecast_autoets` was second by RMSE at 2.525734. `statsforecast_autotbats` won WRMSSE at 0.618397; CartoBoost auto WRMSSE was 0.669928. |
| M5 Forecasting Accuracy full-corpus check | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` | 30,490 item-store full-corpus check, 90 recent days, 28-day holdout, lag-only fast roster | CartoBoost lag RMSE 2.634879, MAE 1.332997, WAPE 0.923884. |
| M6 committed CartoBoost sample | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` | Committed M6-style point proxy sample, CartoBoost-only roster with auto and lag | `cartoboost_auto_forecast` beat `cartoboost_lag` on point RMSE at 0.013439; calibrated RPS is audit-only at 0.208171 versus lag 0.200754. |
| M6 financial assets | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` | 100-symbol daily-return proxy, 28-day holdout, full 15-model roster with fixed no-hyperopt candidate selection | CartoBoost auto won point quality: RMSE 0.013392, MAE 0.007357, WAPE 1.000000. `statsforecast_autoarima` was second by RMSE at 0.013402. Seasonal-naive baselines won calibrated RPS at 0.192195; CartoBoost auto RPS was 0.206007. |

## Bottom Line

The forecasting benchmark has maintained views for real taxi demand, synthetic
taxi-shaped diagnostics, committed M4/M5/M6 samples, and larger M5/M6
competition-style proxy runs:

- Real NYC taxi lane demand: CartoBoost-only plots include
  `cartoboost_auto_forecast`, which beats `cartoboost_lag`.
- Synthetic taxi-shaped committed suite: `cartoboost_auto_forecast` ties
  `cartoboost_lag` in the CartoBoost-only artifact, with both at mean RMSE
  ratio 1.000000. The full
  external-roster artifact ranks `lightgbm_lag` first.
- Non-M scalable generalization check: the compact synthetic external-roster
  wrapper keeps quality pressure outside M4/M5/M6. CartoBoost auto and lag tie
  for first by mean RMSE ratio at 1.000000, ahead of `lightgbm_lag` at 1.196396
  and `xgboost_lag` at 1.258816.
- M4 96-series-per-group committed sample: `cartoboost_auto_forecast` wins or
  ties all 6 groups, with mean RMSE ratio 1.000000 versus 12.104570 for
  `cartoboost_lag`. This is still a sample, not a full M4 corpus claim.
- M5 committed sample: `cartoboost_auto_forecast` is present in the committed
  artifact and beats `cartoboost_lag` on both point RMSE and WRMSSE.
- M6 committed sample: `cartoboost_auto_forecast` is present in the committed
  artifact and beats `cartoboost_lag` on point RMSE/MAE/WAPE after selecting
  the market-neutral return candidate; RPS is emitted for audit rather than
  optimized by this point-quality route.
- M5 comparison sample: Kaggle M5 Accuracy files are now a first-class source
  with a full 14-model roster sample over 100 item-store daily unit-sales
  series, the official 28-day holdout shape, and the same model-family table
  style as the M4 sample.
- M5 full-corpus check: The June 19, 2026 run also completed over all 30,490
  bottom-level series using the public M5 mirror, a 90-day recent-history
  window, and a CartoBoost-only fast roster. That artifact is coverage and
  throughput evidence, not a model bakeoff.
- M6 full-run protocol: M6 assets are now a first-class source for a daily
  return point-forecast proxy over the public M6 asset panel. The June 20, 2026
  run completed over 100 symbols and 38,219 daily-return rows; CartoBoost auto
  now leads the maintained full-roster artifact on RMSE, MAE, and WAPE after
  selecting the market-neutral return candidate, while seasonal-naive baselines
  still lead calibrated RPS.

Forecasting claims should stay split-specific. The taxi run, committed
CartoBoost-only samples, and M5/M6 full-roster artifacts are separate pieces of
evidence. M5 and M6 are real competition panels, but the M5 model comparison
below is a bounded 100-series sample, not a full external-library bakeoff across
all 30,490 bottom-level series.

## Real NYC Taxi Lane Demand

| Field | Value |
| --- | --- |
| Source | NYC TLC yellow taxi trips |
| Period | January 2024 |
| Raw rows | 2,964,624 |
| Clean rows | 2,827,685 |
| Aggregated rows | 744 |
| Series | 24 pickup/dropoff lanes |
| Days | 31 |
| Horizon | 7 days |
| Static covariates | pickup zone, dropoff zone, distance, airport-lane flag, pickup borough code |

The native CartoBoost lag path can use these route static covariates plus rich
calendar features for taxi-shaped sources. That route-context enrichment is not
enabled for the maintained M4/M5/M6 competition artifacts unless a targeted
rerun proves a quality win without an unacceptable latency cost.

Reproduce:

```sh
uv run --group dev python scripts/forecasting_library_benchmark.py \
  --source nyc-taxi \
  --year 2024 \
  --months 1 \
  --taxi-type yellow \
  --lanes 24 \
  --horizon 7 \
  --no-download \
  --no-hyperopt \
  --model-roster cartoboost \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots
```

### Real Taxi Result

| Model | Library | RMSE | MAE | WAPE | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `cartoboost_auto_forecast` | CartoBoost | 39.034 | 29.173 | 0.0937 | Selected the validated seasonal-base route and skipped the unused raw-auto outer fit. |
| `cartoboost_lag` | CartoBoost | 126.861 | 85.685 | 0.2753 | Lag spine baseline with taxi-default partial rolling mean features on the same split. |

The taxi plots include `cartoboost_auto_forecast`. This table is CartoBoost-only;
use the full-roster artifact for external-library comparisons.

### Real Taxi Plots

![Forecasting metric comparison](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_tool_metric_comparison.png)

![Horizon RMSE by tool](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_horizon_rmse_by_tool.png)

![Forecast lines for top lanes](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_forecast_lines.png)

![Actual vs predicted demand](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_actual_vs_predicted.png)

## Synthetic Taxi-Shaped Suite

The synthetic suite uses four taxi-shaped forecasting problems: weekly lane
demand, airport calendar events, route-mix shifts, and borough monthly pulses.
It uses 12 series, 180 days, horizon 14, and 3 rolling-origin folds.
Like the real taxi path, the synthetic route-demand path can enable native
route static covariates and rich deterministic calendar features. M4/M5/M6 use
their compact competition-specific paths.

| Model | Mean RMSE ratio to problem best | Wins/ties | Top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_auto_forecast` | 1.000 | 4 | 4 |
| `cartoboost_lag` | 1.000 | 4 | 4 |

The table above is the fixed CartoBoost-only committed artifact. The current
scalable external-roster committed artifact ranks CartoBoost auto and lag first
at 1.013744, ahead of `lightgbm_lag` at 1.279238. The older full external-roster
artifact still ranks `lightgbm_lag` first at 1.069525, but the attempted refresh
after the gated-indicator change did not complete inside the local run window
because `statsforecast_autotbats` entered slow ARIMA optimization. This suite is
a stress test and wiring check. It is not real TLC evidence.

## Non-M Scalable Generalization Check

This compact external-roster check exists to keep the forecasting stack from
over-indexing on M4, M5, and M6. It uses the same taxi-shaped synthetic problem
families as the committed suite, but with 18 series, 150 days, horizon 14, two
rolling-origin folds, seed 177, and the `scalable` roster. That roster includes
CartoBoost auto and lag plus functime, LightGBM lag, and XGBoost lag baselines.

Reproduce:

```sh
uv run --group dev python scripts/forecasting_generalization.py \
  --compact \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_generalization_scalable_synthetic.json
```

| Model | Mean RMSE ratio to problem best | Wins/ties | Top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_auto_forecast` | 1.000000 | 4 | 4 |
| `cartoboost_lag` | 1.000000 | 4 | 4 |
| `lightgbm_lag` | 1.196396 | 0 | 3 |
| `xgboost_lag` | 1.258816 | 0 | 0 |

The run completed in 25.115 seconds with peak RSS 400.688 MB. This is not a
replacement for the M competition samples; it is a separate external-baseline
guardrail for general route-demand behavior.

## M4 Sample Suite

The maintained M4 artifact scores the first 96 series from each M4 group. It is
included to check non-taxi behavior and library interoperability.

### M4 Model RMSE

Committed CartoBoost-only rows, sorted by mean RMSE.

| Model | Mean RMSE | Mean RMSE ratio to group best |
| --- | ---: | ---: |
| `cartoboost_auto_forecast` | 295.092741 | 1.000000 |
| `cartoboost_lag` | 369.256653 | 12.104570 |

This is a committed CartoBoost-only M4 sample, not a full M4 claim. Use
`--m4-series-limit 0` and include the external roster before making full-corpus
or cross-library M4 statements.

The committed M4 wrapper runs the six groups serially for the clearest timing
provenance, with canonical group order recorded in `dataset.groups`.

## M5 Full Competition Run

The M5 loader consumes Kaggle M5 Forecasting Accuracy files directly when they
exist locally. When downloads are allowed and those files are absent, it uses
Nixtla's public M5 mirror and records the mirror URL in the artifact. It
hard-fails under `--no-download` when the required files are missing.

| Field | Value |
| --- | --- |
| Source | Kaggle M5 Forecasting Accuracy files |
| Required files | `calendar.csv` and `sales_train_evaluation.csv` or `sales_train_validation.csv`; public mirror download is used when local files are absent and downloads are allowed |
| Series | Full bottom-level item-store corpus when `--m5-series-limit 0` is used |
| Split | Last 28 days from the supplied training/evaluation file |
| Horizon | 28 daily steps |
| Scoring | Shared point metrics plus `official_metrics.m5` |
| Maintained comparison artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` |
| Maintained full-corpus artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |

Reproduce the maintained full-roster comparison sample:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster full \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 100 \
  --m5-history-days 90 \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots
```

Reproduce the maintained full-corpus CartoBoost fast run:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster cartoboost \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 0 \
  --m5-history-days 90 \
  --cartoboost-n-estimators 1 \
  --cartoboost-max-depth 3 \
  --cartoboost-min-samples-leaf 20 \
  --no-candidate-selection \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_plots
```

Use a positive `--m5-series-limit` only for smoke tests. Results from limited
runs must be labeled as samples, not as M5 full-corpus evidence.

### M5 Full-Roster Sample Result

Command run on June 20, 2026:

```sh
uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster full \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 100 \
  --m5-history-days 90 \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots
```

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` |
| Plots | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots/` |
| Source files | `data/forecasting_benchmarks/m5/datasets/sales_train_evaluation.csv`, `calendar.csv` |
| Mirror URL | `https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip` |
| Series | 100 item-store series from 30,490 available |
| Rows | 9,000 daily unit-sales rows |
| Days materialized | 90 recent days from 1,941 available days |
| Horizon | 28 daily steps |
| Roster | `full`: CartoBoost, functime, StatsForecast, Prophet, XGBoost lag, LightGBM lag |
| Candidate selection | Enabled, RMSE-first shared one-origin calibration |
| Total runtime | 62.356 seconds |
| Winner | `cartoboost_auto_forecast` |
| Best RMSE | 2.511292 |
| CartoBoost RMSE | 2.511292 |
| CartoBoost MAE | 1.135585 |
| CartoBoost WAPE | 0.916059 |
| Best WRMSSE | `statsforecast_autotbats`, 0.618397 |
| CartoBoost WRMSSE | 0.669928 |

### M5 Model RMSE

Committed CartoBoost-only rows, sorted by RMSE. Use `official_metrics.m5` for
WRMSSE.

| Model | RMSE | MAE | WAPE | Read |
| --- | ---: | ---: | ---: | --- |
| `cartoboost_auto_forecast` | 2.415225 | 1.139285 | 0.910615 | Best point metrics and WRMSSE, 0.568942. |
| `cartoboost_lag` | 2.540625 | 1.219927 | 0.975071 | WRMSSE 0.743721. |

The all-series M5 artifact is a lag-only CartoBoost full-corpus check:

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |
| Run | 30,490 item-store series, 2,744,100 rows, 90 recent days, 28-day holdout |
| Roster | Lag-only CartoBoost fast run |
| Candidate selection | Disabled |
| CartoBoost settings | 1 estimator, learning rate 0.06, max depth 3, min samples leaf 20 |
| Total runtime | 21.941 seconds |
| CartoBoost RMSE | 2.634879 |
| CartoBoost MAE | 1.332997 |
| CartoBoost WAPE | 0.923884 |

## M6 Full Competition Proxy Run

The M6 loader consumes the public M6 methods `assets_m6.csv` file with
`symbol`, `date`, and `price` columns. If the file is missing, the runner fetches
it from the M6 methods repository unless `--no-download` is set.

| Field | Value |
| --- | --- |
| Source | M6 methods asset price panel |
| Required file | `assets_m6.csv` |
| Series | Every symbol in the assets file when `--m6-series-limit 0` is used |
| Transform | Calendar-daily adjusted-price panel with forward-filled non-trading days and daily returns as the target |
| Split | Last `--m6-horizon` calendar days from the daily return panel |
| Default horizon | 28 daily steps |
| Scoring | Shared point metrics plus `official_metrics.m6` |
| Maintained artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |

Reproduce:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m6 \
  --m6-assets-path data/forecasting_benchmarks/m6/assets_m6.csv \
  --m6-series-limit 100 \
  --m6-horizon 28 \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m6_full_plots
```

### M6 Result

Command run on June 20, 2026:

```sh
uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
  --source m6 \
  --model-roster full \
  --m6-assets-path data/forecasting_benchmarks/m6/assets_m6.csv \
  --m6-series-limit 0 \
  --m6-horizon 28 \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m6_full_plots
```

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |
| Plots | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full_plots/` |
| Source URL | `https://raw.githubusercontent.com/Mcompetitions/M6-methods/main/assets_m6.csv` |
| Series | 100 symbols |
| Rows | 38,219 daily-return rows |
| Days | 383 calendar days |
| Horizon | 28 calendar days |
| Roster | `full`: CartoBoost, functime, StatsForecast, Prophet, XGBoost lag, LightGBM lag |
| Candidate selection | Enabled, RMSE-first one-origin calibration |
| Total runtime | 103.262 seconds |
| Winner | `cartoboost_auto_forecast` |
| Best RMSE | 0.013392 |
| CartoBoost RMSE | 0.013392 |
| CartoBoost MAE | 0.007357 |
| CartoBoost WAPE | 1.000000 |
| CartoBoost RMSE ratio vs best forecasting library | 0.999218 |
| Best calibrated RPS | `functime_snaive` and `statsforecast_seasonal_naive`, 0.192195 |
| CartoBoost calibrated RPS | 0.206007 |

| Rank | Model | Read |
| ---: | --- | --- |
| 1 | `cartoboost_auto_forecast` | Best RMSE, MAE, and WAPE after selecting the market-neutral return candidate. |
| 2 | `statsforecast_autoarima` | Best forecasting-library model by RMSE. |
| 3 | `statsforecast_autoets` | Close second forecasting-library baseline. |
| 4 | `functime_ridge` | Best functime model by RMSE. |
| 5 | `statsforecast_autotbats` | Strong but slower than simpler baselines. |

### M6 Calibrated RPS

The RPS artifact uses a pre-holdout validation window to build deterministic
Dirichlet-smoothed rank-bucket calibration. Lower is better.

| Rank | Model | Calibrated RPS | Decision return |
| ---: | --- | ---: | ---: |
| 1 | `functime_snaive` | 0.192195 | -0.014937 |
| 1 | `statsforecast_seasonal_naive` | 0.192195 | -0.014937 |
| 3 | `statsforecast_autotheta` | 0.197417 | -0.002552 |
| 4 | `prophet_additive` | 0.197646 | 0.029679 |
| 5 | `functime_ridge` | 0.198198 | 0.005945 |
| 12 | `cartoboost_auto_forecast` | 0.206007 | -0.006657 |

### M6 Model RMSE

Committed CartoBoost-only rows, sorted by RMSE. Use `official_metrics.m6` for
RPS.

| Model | RMSE | MAE | WAPE | Read |
| --- | ---: | ---: | ---: | --- |
| `cartoboost_auto_forecast` | 0.013439 | 0.007342 | 1.000000 | Best point metrics; calibrated RPS 0.208171. |
| `cartoboost_lag` | 0.014440 | 0.009290 | 1.265338 | Calibrated RPS 0.200754. |

This is intentionally named a proxy run. The official M6 competition combined
rank-probability forecasts and investment decisions, with RPS and investment
return as official scoring dimensions. The artifact includes deterministic
rank-probability scoring and decision rows calibrated on a pre-holdout window,
but those rows are audit evidence rather than an official M6 submission file.

## Limitations

- Interval coverage is only benchmark evidence when actually computed.
- Real taxi panel is short: 31 days and a 7-day holdout.
- Synthetic suite is diagnostic.
- M4 artifact is a 96-series-per-group sample.
- M5 full-roster evidence is a 100-series sample with a 90-day materialized
  history window; the full 30,490-series artifact is a lag-only CartoBoost fast
  run.
- M6 artifacts include deterministic RPS and decision rows on top of the daily
  point-forecast proxy, but they are not official leaderboard submission
  payloads.
- External model availability depends on optional benchmark dependencies.
