# Forecasting Benchmark

This page is the maintained forecasting benchmark write-up. It shows what runs
where, which artifact backs each result, and what claim is allowed. The detailed
metric tables live in the JSON artifacts under
`docs/assets/nyc_taxi_benchmarks/`.

The benchmark commands use fixed model rosters, fixed seeds, and `--no-hyperopt`.
A result is only a quality claim when it cites the artifact, split, seed, model
roster, and metric table. M5 and M6 claims must also cite `official_metrics`.

## Metric Reading

RMSE, MAE, WAPE, WRMSSE, and RPS are raw error metrics, so lower is better.
Their meaning depends on target scale and task. `Mean RMSE ratio to problem
best` is different: `1.000000` means the model tied the best RMSE observed for
that artifact and split.

A raw WAPE of `1.000000` is not a badge of accuracy. It means total absolute
error equals total absolute actual value. On signed-return tasks such as the M6
proxy, WAPE is especially easy to misread because market-neutral or near-zero
forecasts can have competitive RMSE/MAE while WAPE remains near 1.

## What Runs Where

| Run | Artifact | Roster | Current read |
| --- | --- | --- | --- |
| Real NYC taxi lane demand | `forecasting_library_benchmark_real.json` | CartoBoost auto and lag | Auto beats lag: RMSE 39.033944 vs 126.861. |
| Synthetic taxi-shaped suite | `forecasting_overhaul_committed_suite.json` | CartoBoost auto and lag | Auto and lag tie at mean RMSE ratio 1.000000. |
| Synthetic scalable external roster | `forecasting_overhaul_committed_suite_scalable_roster.json` | CartoBoost, LightGBM, XGBoost, functime | Auto and lag rank first at 1.013744; `lightgbm_lag` is 1.279238. |
| Synthetic full external roster | `forecasting_overhaul_committed_suite_full_roster.json` | Full external roster | Older artifact: `lightgbm_lag` remains first at mean RMSE ratio 1.069525. |
| Generalization guardrail | `forecasting_generalization_scalable_synthetic.json` | Scalable external roster | Auto and lag tie at 1.000000; `lightgbm_lag` is 1.196396 and `xgboost_lag` is 1.258816. |
| M4 committed sample | `forecasting_overhaul_m4_committed.json` | CartoBoost auto and lag | Auto wins/ties all 6 groups at mean RMSE ratio 1.000000; lag is 12.104570. |
| M5 committed sample | `forecasting_overhaul_m5_committed.json` | CartoBoost auto and lag | Auto beats lag: RMSE 2.415225 vs 2.540625; WRMSSE 0.568942 vs 0.743721. |
| M5 100-series comparison | `forecasting_m5_full_roster_sample.json` | Full 14-model roster | Auto has best RMSE 2.511292; `statsforecast_autotbats` has best WRMSSE 0.618397. |
| M5 full-corpus fast check | `forecasting_m5_full.json` | CartoBoost lag fast roster | Lag completed 30,490 series: RMSE 2.634879. This is coverage evidence, not a bakeoff. |
| M6 committed sample | `forecasting_overhaul_m6_committed.json` | CartoBoost auto and lag | Auto RMSE 0.013439 vs lag 0.014440; lag RPS 0.200754 vs auto 0.208171. WAPE is diagnostic only. |
| M6 100-symbol proxy | `forecasting_m6_full.json` | Full 15-model roster | Auto has best RMSE 0.013392; seasonal-naive baselines have best calibrated RPS 0.192195. |

## Bottom Line

CartoBoost auto is now guarded by the lag spine. When validation does not justify
a riskier candidate, auto falls back instead of losing casually to
`cartoboost_lag`.

On the committed CartoBoost-only samples, auto ties lag on the synthetic suite,
wins/ties every M4 group, beats lag on M5 RMSE and WRMSSE, and beats lag on M6
RMSE/MAE. That is not a blanket external-library win. The older full external
synthetic artifact is still led by LightGBM. The M5 full-roster sample is led by
CartoBoost on point RMSE, but AutoTBATS still leads WRMSSE. The M6 proxy is led
by CartoBoost on point RMSE, but seasonal-naive baselines still lead calibrated
RPS.

## Reproduce The Maintained Runs

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

uv run --group dev python scripts/forecasting_library_benchmark.py \
  --suite committed \
  --no-hyperopt \
  --model-roster cartoboost \
  --no-candidate-selection \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite.json

uv run --group dev python scripts/forecasting_generalization.py \
  --compact \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_generalization_scalable_synthetic.json

uv run --group dev python scripts/forecasting_m4.py \
  --committed \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m4_committed.json

uv run --group dev --group bench python scripts/forecasting_m5.py \
  --committed \
  --official-wrmsse \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json

uv run --group dev --group bench python scripts/forecasting_m6.py \
  --committed \
  --official-style \
  --no-hyperopt \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json
```

Larger comparison runs:

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

## Notes By Run

Real taxi uses January 2024 NYC TLC yellow taxi trips, 24 pickup/dropoff lanes,
31 days, and a 7-day holdout. The run is CartoBoost-only and shows that auto
selected the validated seasonal-base route while lag stayed as the supervised
lag baseline. The plots in `forecasting_plots/` show the metric comparison,
horizon RMSE, forecast lines, and actual-vs-predicted demand.

The synthetic suite is a route-demand diagnostic, not real TLC evidence. The
generalization guardrail keeps a non-M external-baseline check in the loop so
forecasting changes are not judged only on M4/M5/M6.

The M4 artifact scores the first 96 series from each M4 group. It is a committed
sample, not a full M4 corpus or cross-library claim.

The M5 committed sample reports both shared point metrics and
`official_metrics.m5`. The separate 100-series M5 comparison uses a full
external roster, but it is still a sample. The all-series M5 artifact is a
lag-only fast coverage run over 30,490 bottom-level item-store series.

The M6 artifacts are point-forecast proxies over daily returns. They include RPS
and decision rows for audit, but they are not official M6 submission payloads.
For M6, prefer the RMSE/MAE and RPS reads over WAPE.

## Limitations

- Real taxi panel is short: 31 days and a 7-day holdout.
- Synthetic results are diagnostics, not real-world superiority claims.
- M4 is a 96-series-per-group sample.
- M5 full-roster evidence is a 100-series sample; the full-corpus artifact is
  a lag-only coverage run.
- M6 is a daily-return point proxy with audit RPS, not an official leaderboard
  submission.
- External model availability depends on optional benchmark dependencies.
