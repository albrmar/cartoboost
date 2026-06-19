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

The commands above do not imply a new benchmark win by themselves. Public claims
must cite the resulting artifact, split, seed, model roster, and metric table.

## Forecasting Overhaul Committed Run

Run date: June 19, 2026. These runs used the deterministic CartoBoost-only
roster, fixed seed 42, no hyperopt, and the committed benchmark sample settings.
Artifacts are committed under `docs/assets/nyc_taxi_benchmarks/`.

| Suite | Command | Artifact | Result |
| --- | --- | --- | --- |
| Synthetic committed suite | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --model-roster cartoboost --output target/forecasting_library_benchmark_committed.json` | `forecasting_overhaul_committed_suite.json` | `cartoboost_auto_forecast` ranked first by mean RMSE ratio, 1.006163 vs `cartoboost_lag` 1.008250. |
| Synthetic committed suite, full external roster | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite_full_roster.json` | `forecasting_overhaul_committed_suite_full_roster.json` | `lightgbm_lag` ranked first by mean RMSE ratio at 1.069525. `cartoboost_auto_forecast` ranked second at 1.187837, slightly ahead of `cartoboost_lag` at 1.189963. |
| M4 committed sample | `uv run --group dev python scripts/forecasting_m4.py --committed --no-hyperopt --output target/forecasting_m4_committed.json` | `forecasting_overhaul_m4_committed.json` | `cartoboost_auto_forecast` and `cartoboost_lag` tied: both mean RMSE ratio 1.000000 and 6/6 wins-or-ties. |
| M5 committed sample | `uv run --group dev python scripts/forecasting_m5.py --committed --official-wrmsse --no-hyperopt --output target/forecasting_m5_committed.json` | `forecasting_overhaul_m5_committed.json` | Tie on the current point-metric harness: RMSE 2.455906, MAE 1.149696, WAPE 0.918936 for both methods. |
| M6 committed sample | `uv run --group dev python scripts/forecasting_m6.py --committed --official-style --no-hyperopt --output target/forecasting_m6_committed.json` | `forecasting_overhaul_m6_committed.json` | Tie on the current point-metric proxy harness: RMSE 0.014440, MAE 0.009290, WAPE 1.265338 for both methods. |

Interpretation: the deterministic auto route improved the committed synthetic
suite average versus `cartoboost_lag`, and it did not regress the M4/M5/M6
committed samples. The full external-roster synthetic run is not a CartoBoost
win: LightGBM was the best method on that committed suite. The M5 and M6 wrapper
commands enforce the official-style flags, but this benchmark payload is still
the existing point-metric/proxy harness; use the WRMSSE and RPS helper tests for
scorer correctness until level-wise WRMSSE and M6 RPS are fully emitted in
benchmark artifacts.

## Competition Results Snapshot

| Competition | Artifact | Scope | Result |
| --- | --- | --- | --- |
| M5 committed CartoBoost sample | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` | Committed M5-style sample, CartoBoost-only roster with auto and lag | `cartoboost_auto_forecast` tied `cartoboost_lag`: RMSE 2.455906, MAE 1.149696, WAPE 0.918936. |
| M5 Forecasting Accuracy comparison sample | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` | Older 100 item-store sample, 90 recent days, 28-day holdout, full 14-model roster without the auto alias | `statsforecast_autoets` won RMSE at 2.525734; CartoBoost lag RMSE was 2.678097. |
| M5 Forecasting Accuracy full-corpus check | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` | Older 30,490 item-store full-corpus check, 90 recent days, 28-day holdout, lag-only fast roster | CartoBoost lag RMSE 2.634879, MAE 1.332997, WAPE 0.923884. |
| M6 committed CartoBoost sample | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` | Committed M6-style point proxy sample, CartoBoost-only roster with auto and lag | `cartoboost_auto_forecast` tied `cartoboost_lag`: RMSE 0.014440, MAE 0.009290, WAPE 1.265338. |
| M6 financial assets | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` | Older 100-symbol daily-return proxy, 28-day holdout, full 14-model roster without the auto alias | `statsforecast_autoarima` won RMSE at 0.013402; CartoBoost lag RMSE was 0.014348. |

## Bottom Line

The forecasting benchmark has maintained views for real taxi demand, synthetic
taxi-shaped diagnostics, committed M4/M5/M6 samples, and larger M5/M6
competition-style proxy runs:

- Real NYC taxi lane demand: refreshed plots include `cartoboost_auto_forecast`;
  StatsForecast AutoTBATS wins RMSE, while auto ties `cartoboost_lag`.
- Synthetic taxi-shaped committed suite: `cartoboost_auto_forecast` is the best
  CartoBoost method in the refreshed CartoBoost-only artifact, with mean RMSE
  ratio 1.006163 versus `cartoboost_lag` at 1.008250. In the full external
  roster artifact, `lightgbm_lag` wins and `cartoboost_auto_forecast` ranks
  ahead of `cartoboost_lag`.
- M4 24-series-per-group committed sample: `cartoboost_auto_forecast` and
  `cartoboost_lag` tie by mean RMSE ratio, but this is still a sample, not a
  full M4 corpus claim.
- M5 committed sample: `cartoboost_auto_forecast` is present in the committed
  artifact and ties `cartoboost_lag` on the current point-metric harness.
- M6 committed sample: `cartoboost_auto_forecast` is present in the committed
  artifact and ties `cartoboost_lag` on the current point-metric proxy harness.
- M5 comparison sample: Kaggle M5 Accuracy files are now a first-class source
  with a full 14-model roster sample over 100 item-store daily unit-sales
  series, the official 28-day holdout shape, and the same model-family table
  style as the M4 sample.
- M5 full-corpus check: The June 19, 2026 run also completed over all 30,490
  bottom-level series using the public M5 mirror, a 90-day recent-history
  window, and a CartoBoost-only fast roster. That artifact is coverage and
  throughput evidence, not a model bakeoff.
- M6 full-run protocol: M6 assets are now a first-class source for a daily
  return point-forecast proxy over the public M6 asset panel. The June 19, 2026
  run completed over 100 symbols and 38,219 daily-return rows.

Forecasting claims should stay split-specific. The refreshed taxi run, the
committed CartoBoost-only samples, and the older M5/M6 full-roster artifacts are
separate pieces of evidence. M5 and M6 are real competition panels, but the M5
model comparison below is a bounded 100-series sample, not a full
external-library bakeoff across all 30,490 bottom-level series.

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
| `statsforecast_autotbats` | StatsForecast | 40.179 | 31.878 | 0.1024 | Best RMSE on the refreshed real taxi run. |
| `statsforecast_autoces` | StatsForecast | 47.357 | 36.921 | 0.1186 | Second by RMSE. |
| `statsforecast_autoarima` | StatsForecast | 56.986 | 42.836 | 0.1376 | Third by RMSE. |
| `cartoboost_auto_forecast` | CartoBoost | 67.847 | 48.685 | 0.1564 | Ties `cartoboost_lag`; present in the refreshed taxi plots. |
| `cartoboost_lag` | CartoBoost | 67.847 | 48.685 | 0.1564 | Baseline lag route. |

The refreshed taxi plots include `cartoboost_auto_forecast`. This run is not a
CartoBoost win: StatsForecast AutoTBATS has the best RMSE on the short January
lane panel, while the auto route ties the direct lag baseline.

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
| `cartoboost_auto_forecast` | 1.006 | 2 | 4 |
| `cartoboost_lag` | 1.008 | 3 | 4 |

The table above is the fixed CartoBoost-only committed artifact. The full
external-roster committed artifact ranks `lightgbm_lag` first at 1.069525,
`cartoboost_auto_forecast` second at 1.187837, and `cartoboost_lag` third at
1.189963. This suite is a stress test and wiring check. It is not real TLC
evidence.

## M4 Sample Suite

The maintained M4 artifact scores the first 24 series from each M4 group. It is
included to check non-taxi behavior and library interoperability.

| Model | Mean RMSE ratio to group best | Wins/ties | Top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_auto_forecast` | 1.000 | 6 | 6 |
| `cartoboost_lag` | 1.000 | 6 | 6 |

### M4 Model RMSE

M4 groups have different scales, so the ratio table above is the better
cross-group comparison. The table below reports the arithmetic mean of each
model's group RMSE from the same committed CartoBoost-only artifact.

| Model | Mean RMSE | Mean RMSE ratio to group best |
| --- | ---: | ---: |
| `cartoboost_auto_forecast` | 385.707610 | 1.000000 |
| `cartoboost_lag` | 385.707610 | 1.000000 |

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
| Scoring | Shared harness: RMSE, MAE, MASE, WAPE, SMAPE, bias |
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

Command run on June 19, 2026:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster full \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 100 \
  --m5-history-days 90 \
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
| Total runtime | 93.734 seconds |
| Winner | `statsforecast_autoets` |
| Best RMSE | 2.525734 |
| CartoBoost RMSE | 2.678097 |
| CartoBoost MAE | 1.187821 |
| CartoBoost WAPE | 0.958196 |

### M5 Model RMSE

The table below reports every model present in the older M5 full-roster sample
artifact, ranked by RMSE. That older artifact predates the
`cartoboost_auto_forecast` alias; the committed CartoBoost-only table above is
the M5 artifact that includes auto.

The newer committed CartoBoost-only M5 overhaul artifact also contains the
auto route. That run is separate from the older full external-roster sample
below:

| Committed CartoBoost model | Artifact | RMSE | MAE | WAPE | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `cartoboost_auto_forecast` | `forecasting_overhaul_m5_committed.json` | 2.455906 | 1.149696 | 0.918936 | Tied `cartoboost_lag` on the current point-metric harness. |
| `cartoboost_lag` | `forecasting_overhaul_m5_committed.json` | 2.455906 | 1.149696 | 0.918936 | Baseline direct lag route. |

| Model | RMSE | MAE | WAPE |
| --- | ---: | ---: | ---: |
| `statsforecast_autoets` | 2.525734 | 1.141999 | 0.921232 |
| `statsforecast_dynamic_optimized_theta` | 2.556517 | 1.163750 | 0.938779 |
| `statsforecast_autotbats` | 2.602055 | 1.156588 | 0.933001 |
| `functime_ridge` | 2.606775 | 1.207878 | 0.974376 |
| `statsforecast_autotheta` | 2.607077 | 1.196042 | 0.964828 |
| `statsforecast_autoarima` | 2.655754 | 1.194312 | 0.963433 |
| `cartoboost_lag` | 2.678097 | 1.187821 | 0.958196 |
| `functime_snaive` | 2.678097 | 1.187821 | 0.958196 |
| `functime_lightgbm` | 2.678097 | 1.187821 | 0.958196 |
| `statsforecast_seasonal_naive` | 2.678097 | 1.187821 | 0.958196 |
| `statsforecast_autoces` | 2.678097 | 1.187821 | 0.958196 |
| `prophet_additive` | 2.678097 | 1.187821 | 0.958196 |
| `xgboost_lag` | 2.793477 | 1.500446 | 1.210386 |
| `lightgbm_lag` | 2.825295 | 1.253991 | 1.011575 |

This older sample does not support a claim that CartoBoost beats the M5 model
families; `cartoboost_lag` ranks seventh by RMSE. The all-series M5 artifact is
also older and remains useful as a lag-only CartoBoost full-corpus check:

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |
| Scope | 30,490 item-store series, 2,744,100 rows, 90 recent days, 28-day holdout |
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
| Scoring | Shared library harness: RMSE, MAE, MASE, WAPE, SMAPE, bias |
| Maintained artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |

Reproduce:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m6 \
  --m6-assets-path data/forecasting_benchmarks/m6/assets_m6.csv \
  --m6-series-limit 0 \
  --m6-horizon 28 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots
```

### M6 Result

Command run on June 19, 2026:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m6 \
  --m6-assets-path data/forecasting_benchmarks/m6/assets_m6.csv \
  --m6-series-limit 0 \
  --m6-horizon 28 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots
```

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |
| Plots | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots/` |
| Source URL | `https://raw.githubusercontent.com/Mcompetitions/M6-methods/main/assets_m6.csv` |
| Series | 100 symbols |
| Rows | 38,219 daily-return rows |
| Days | 383 calendar days |
| Horizon | 28 calendar days |
| Roster | `full`: CartoBoost, functime, StatsForecast, Prophet, XGBoost lag, LightGBM lag |
| Candidate selection | Enabled, shared one-origin calibration |
| Total runtime | 173.591 seconds |
| Winner | `statsforecast_autoarima` |
| Best RMSE | 0.013402 |
| CartoBoost RMSE | 0.014348 |
| CartoBoost MAE | 0.009357 |
| CartoBoost WAPE | 1.271868 |
| CartoBoost RMSE ratio vs best | 1.070585 |

| Rank | Model | Read |
| ---: | --- | --- |
| 1 | `statsforecast_autoarima` | Best RMSE on this daily-return proxy. |
| 2 | `statsforecast_autoets` | Second by RMSE. |
| 3 | `functime_ridge` | Best non-StatsForecast model by RMSE. |
| 4 | `statsforecast_autotbats` | Strong but slower than simpler baselines. |
| 14 | `cartoboost_lag` | 7.1% higher RMSE than the best forecasting-library model. |

### M6 Model RMSE

The table below reports every model present in the older M6 full-run proxy
artifact, ranked by RMSE. That older artifact predates the
`cartoboost_auto_forecast` alias; the committed CartoBoost-only table above is
the M6 artifact that includes auto.

The newer committed CartoBoost-only M6 overhaul artifact contains the auto route
on the point-forecast proxy harness:

| Committed CartoBoost model | Artifact | RMSE | MAE | WAPE | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `cartoboost_auto_forecast` | `forecasting_overhaul_m6_committed.json` | 0.014440 | 0.009290 | 1.265338 | Tied `cartoboost_lag` on the current point-metric proxy harness. |
| `cartoboost_lag` | `forecasting_overhaul_m6_committed.json` | 0.014440 | 0.009290 | 1.265338 | Baseline direct lag route. |

| Model | RMSE | MAE | WAPE |
| --- | ---: | ---: | ---: |
| `statsforecast_autoarima` | 0.013402 | 0.007400 | 1.005844 |
| `statsforecast_autoets` | 0.013408 | 0.007456 | 1.013524 |
| `functime_ridge` | 0.013474 | 0.007670 | 1.042553 |
| `statsforecast_autotbats` | 0.013477 | 0.007663 | 1.041580 |
| `statsforecast_autoces` | 0.013522 | 0.007617 | 1.035289 |
| `statsforecast_dynamic_optimized_theta` | 0.013669 | 0.008204 | 1.115187 |
| `functime_snaive` | 0.013674 | 0.007500 | 1.019396 |
| `functime_lightgbm` | 0.013674 | 0.007500 | 1.019396 |
| `statsforecast_seasonal_naive` | 0.013674 | 0.007500 | 1.019396 |
| `prophet_additive` | 0.013674 | 0.007500 | 1.019396 |
| `lightgbm_lag` | 0.013674 | 0.007500 | 1.019396 |
| `statsforecast_autotheta` | 0.013683 | 0.008228 | 1.118462 |
| `xgboost_lag` | 0.014246 | 0.008896 | 1.209160 |
| `cartoboost_lag` | 0.014348 | 0.009357 | 1.271868 |

This is intentionally named a proxy run. The official M6 competition combined
rank-probability forecasts and investment decisions, with RPS and investment
return as official scoring dimensions. The CartoBoost forecasting-library
harness is a point-forecast comparison, so it is useful for interoperability and
daily return forecasting behavior, not for claiming official M6 leaderboard
equivalence.

## Limitations

- Interval coverage is only benchmark evidence when actually computed.
- Real taxi panel is short: 31 days and a 7-day holdout.
- Synthetic suite is diagnostic.
- M4 artifact is a 24-series-per-group sample.
- M5 full-roster evidence is a 100-series sample with a 90-day materialized
  history window; the full 30,490-series artifact is an older lag-only
  CartoBoost fast run.
- M6 uses a daily point-forecast proxy, not the official RPS/investment-return
  competition scorer.
- External model availability depends on optional benchmark dependencies.
