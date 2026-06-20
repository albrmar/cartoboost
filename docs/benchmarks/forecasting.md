# Forecasting Tool Benchmark

For the deterministic forecasting architecture, see
[Forecasting Overhaul](../forecasting_overhaul.md). The benchmark command
surface accepts fixed no-hyperopt runs:

```sh
python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt
python scripts/forecasting_m4.py --committed --no-hyperopt
python scripts/forecasting_m5.py --committed --official-wrmsse --no-hyperopt
python scripts/forecasting_m6.py --committed --official-style --no-hyperopt
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
| Real NYC taxi lane demand | `uv run --group dev python scripts/forecasting_library_benchmark.py --source nyc-taxi --year 2024 --months 1 --taxi-type yellow --lanes 24 --horizon 7 --no-download --model-roster cartoboost --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots` | `forecasting_library_benchmark_real.json` | `cartoboost_lag` and `cartoboost_auto_forecast` tied: RMSE 67.846536, MAE 48.685209, WAPE 0.156442. |
| Synthetic committed suite | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --model-roster cartoboost --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite.json` | `forecasting_overhaul_committed_suite.json` | `cartoboost_auto_forecast` and `cartoboost_lag` tied with mean RMSE ratio 1.000000 and 4/4 wins-or-ties. |
| Synthetic committed suite, full external roster | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite_full_roster.json` | `forecasting_overhaul_committed_suite_full_roster.json` | `lightgbm_lag` ranked first by mean RMSE ratio at 1.069525. |
| M4 committed sample | `uv run --group dev python scripts/forecasting_m4.py --committed --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m4_committed.json` | `forecasting_overhaul_m4_committed.json` | `cartoboost_auto_forecast` ranked first with mean RMSE ratio 1.000000 and 6/6 wins-or-ties; `cartoboost_lag` had mean RMSE ratio 12.104570. |
| M5 committed sample | `uv run --group dev --group bench python scripts/forecasting_m5.py --committed --official-wrmsse --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` | `forecasting_overhaul_m5_committed.json` | `cartoboost_auto_forecast` beat `cartoboost_lag`: RMSE 2.415225, MAE 1.139285, WAPE 0.910615, WRMSSE 0.568942. |
| M6 committed sample | `uv run --group dev --group bench python scripts/forecasting_m6.py --committed --official-style --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` | `forecasting_overhaul_m6_committed.json` | `cartoboost_auto_forecast` beat `cartoboost_lag` on point metrics and training-calibrated RPS: RMSE 0.013727, MAE 0.007484, WAPE 1.019331, RPS 0.198766. |

Interpretation: the protected auto route now uses the lag spine when validation
does not justify a riskier candidate. On the committed CartoBoost-only samples,
`cartoboost_auto_forecast` ties `cartoboost_lag` on the synthetic suite, beats
or ties every M4 group, beats lag on M5 WRMSSE, and beats lag on M6 point
metrics and training-calibrated RPS. This is not a full
external-library win claim. The maintained full-roster and competition-style
comparison artifacts still show external baselines ahead on important broader
checks.

## Current Benchmark Status

| Artifact | Refresh status | Current read |
| --- | --- | --- |
| `forecasting_overhaul_committed_suite.json` | Refreshed with provenance fields | Protected `cartoboost_auto_forecast` and `cartoboost_lag` tie the CartoBoost-only synthetic suite at mean RMSE ratio 1.000000. |
| `forecasting_overhaul_m4_committed.json` | Refreshed with provenance fields | Protected auto wins/ties all 6 M4 groups, with mean RMSE ratio 1.000000 versus 12.104570 for lag. |
| `forecasting_overhaul_m5_committed.json` | Refreshed after M5 WRMSSE selector changes | Protected auto beats lag on point metrics and WRMSSE 0.568942 versus 0.743721. |
| `forecasting_overhaul_m6_committed.json` | Refreshed after M6 training-window RPS calibration | Protected auto selects the phase-14 calendar candidate, beats lag on point metrics, and beats lag on calibrated RPS 0.198766 versus 0.200754. |
| `forecasting_overhaul_committed_suite_full_roster.json` | Legacy maintained artifact; refresh attempted but `statsforecast_autotbats` did not complete in the local run window | `lightgbm_lag` remains the maintained winner at mean RMSE ratio 1.069525; CartoBoost auto is 1.187837 in that artifact. |
| `forecasting_m5_full_roster_sample.json` | Refreshed after M5 WRMSSE selector changes | `statsforecast_autoets` remains the point-metric winner at RMSE 2.525734; CartoBoost auto is third by RMSE at 2.566068. `statsforecast_autotbats` leads official WRMSSE at 0.618397; CartoBoost auto is second at 0.623464. |
| `forecasting_m5_full.json` | Legacy maintained full-corpus coverage artifact | CartoBoost lag completed the 30,490-series fast run with RMSE 2.634879; this is coverage evidence, not an external bakeoff. |
| `forecasting_m6_full.json` | Refreshed after M6 training-window RPS calibration | `statsforecast_autoarima` remains the maintained winner at RMSE 0.013402; CartoBoost auto is seventh by RMSE at 0.013674 and third by calibrated RPS at 0.197282. |

## Competition Results Snapshot

| Competition | Artifact | Details | Result |
| --- | --- | --- | --- |
| M5 committed CartoBoost sample | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` | Committed M5-style sample, CartoBoost-only roster with auto and lag | `cartoboost_auto_forecast` beat `cartoboost_lag`: RMSE 2.415225, MAE 1.139285, WAPE 0.910615, WRMSSE 0.568942. |
| M5 Forecasting Accuracy comparison sample | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` | 100 item-store sample, 90 recent days, 28-day holdout, full 14-model roster with fixed no-hyperopt candidate selection | `statsforecast_autoets` won RMSE at 2.525734; CartoBoost auto was third at RMSE 2.566068. `statsforecast_autotbats` won WRMSSE at 0.618397; CartoBoost auto was second at 0.623464. |
| M5 Forecasting Accuracy full-corpus check | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` | 30,490 item-store full-corpus check, 90 recent days, 28-day holdout, lag-only fast roster | CartoBoost lag RMSE 2.634879, MAE 1.332997, WAPE 0.923884. |
| M6 committed CartoBoost sample | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` | Committed M6-style point proxy sample, CartoBoost-only roster with auto and lag | `cartoboost_auto_forecast` beat `cartoboost_lag` on point RMSE at 0.013727 and calibrated RPS 0.198766 versus 0.200754. |
| M6 financial assets | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` | 100-symbol daily-return proxy, 28-day holdout, full 15-model roster with fixed no-hyperopt candidate selection | `statsforecast_autoarima` won RMSE at 0.013402; CartoBoost auto RMSE was 0.013674. `functime_snaive` and `statsforecast_seasonal_naive` tied the calibrated-RPS lead at 0.192195; CartoBoost auto ranked third at 0.197282. |

## Bottom Line

The forecasting benchmark has maintained views for real taxi demand, synthetic
taxi-shaped diagnostics, committed M4/M5/M6 samples, and larger M5/M6
competition-style proxy runs:

- Real NYC taxi lane demand: CartoBoost-only plots include
  `cartoboost_auto_forecast`, which ties `cartoboost_lag`.
- Synthetic taxi-shaped committed suite: `cartoboost_auto_forecast` and
  `cartoboost_lag` tie in the CartoBoost-only artifact, with mean RMSE
  ratio 1.000000. The full
  external-roster artifact ranks `lightgbm_lag` first.
- M4 96-series-per-group committed sample: `cartoboost_auto_forecast` wins or
  ties all 6 groups, with mean RMSE ratio 1.000000 versus 12.104570 for
  `cartoboost_lag`. This is still a sample, not a full M4 corpus claim.
- M5 committed sample: `cartoboost_auto_forecast` is present in the committed
  artifact and beats `cartoboost_lag` on both point RMSE and WRMSSE.
- M6 committed sample: `cartoboost_auto_forecast` is present in the committed
  artifact and beats `cartoboost_lag` on point RMSE/MAE/WAPE and
  training-calibrated RPS.
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
  run completed over 100 symbols and 38,219 daily-return rows.

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

Reproduce:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source nyc-taxi \
  --year 2024 \
  --months 1 \
  --taxi-type yellow \
  --lanes 24 \
  --horizon 7 \
  --no-download \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots
```

### Real Taxi Result

| Model | Library | RMSE | MAE | WAPE | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `cartoboost_lag` | CartoBoost | 67.847 | 48.685 | 0.1564 | Tied best on the CartoBoost-only taxi run. |
| `cartoboost_auto_forecast` | CartoBoost | 67.847 | 48.685 | 0.1564 | Tied `cartoboost_lag`; present in the taxi plots. |

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

| Model | Mean RMSE ratio to problem best | Wins/ties | Top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_auto_forecast` | 1.000 | 4 | 4 |
| `cartoboost_lag` | 1.000 | 4 | 4 |

The table above is the fixed CartoBoost-only committed artifact. The full
external-roster committed artifact ranks `lightgbm_lag` first at 1.069525. This
suite is a stress test and wiring check. It is not real TLC evidence.

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
| Candidate selection | Enabled, shared one-origin calibration |
| Total runtime | 79.757 seconds |
| Winner | `statsforecast_autoets` |
| Best RMSE | 2.525734 |
| CartoBoost RMSE | 2.566068 |
| CartoBoost MAE | 1.159957 |
| CartoBoost WAPE | 0.935718 |
| Best WRMSSE | `statsforecast_autotbats`, 0.618397 |
| CartoBoost WRMSSE | 0.623464 |

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
| Roster | Legacy lag-only CartoBoost fast run |
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
  --m6-series-limit 0 \
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
| Candidate selection | Enabled, shared one-origin calibration |
| Total runtime | 154.577 seconds |
| Winner | `statsforecast_autoarima` |
| Best RMSE | 0.013402 |
| CartoBoost RMSE | 0.013674 |
| CartoBoost MAE | 0.007500 |
| CartoBoost WAPE | 1.019396 |
| CartoBoost RMSE ratio vs best | 1.020307 |
| Best calibrated RPS | `functime_snaive` and `statsforecast_seasonal_naive`, 0.192195 |
| CartoBoost calibrated RPS | 0.197282, third by calibrated RPS |

| Rank | Model | Read |
| ---: | --- | --- |
| 1 | `statsforecast_autoarima` | Best RMSE on this daily-return proxy. |
| 2 | `statsforecast_autoets` | Second by RMSE. |
| 3 | `functime_ridge` | Best non-StatsForecast model by RMSE. |
| 4 | `statsforecast_autotbats` | Strong but slower than simpler baselines. |
| 7 | `cartoboost_auto_forecast` | 2.0% higher RMSE than the best forecasting-library model after selecting the phase-14 calendar candidate. |

### M6 Calibrated RPS

The RPS artifact uses a pre-holdout validation window to build deterministic
Dirichlet-smoothed rank-bucket calibration. Lower is better.

| Rank | Model | Calibrated RPS | Decision return |
| ---: | --- | ---: | ---: |
| 1 | `functime_snaive` | 0.192195 | -0.014937 |
| 1 | `statsforecast_seasonal_naive` | 0.192195 | -0.014937 |
| 3 | `cartoboost_auto_forecast` | 0.197282 | -0.010734 |
| 4 | `statsforecast_autotheta` | 0.197417 | -0.002552 |
| 5 | `prophet_additive` | 0.197646 | 0.029679 |

### M6 Model RMSE

Committed CartoBoost-only rows, sorted by RMSE. Use `official_metrics.m6` for
RPS.

| Model | RMSE | MAE | WAPE | Read |
| --- | ---: | ---: | ---: | --- |
| `cartoboost_auto_forecast` | 0.013727 | 0.007484 | 1.019331 | Best point metrics; calibrated RPS 0.198766. |
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
