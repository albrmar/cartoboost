# Forecasting Tool Benchmark

## What It Tests

This benchmark compares `cartoboost_lag` with forecasting baselines on taxi
lane-demand panels and selected public forecasting datasets. The key question is
whether a shared lag model satisfies the rolling-origin target relative to
simple seasonal/statistical baselines.

## Data

Each run should record source, extraction date, series count, frequency,
training length, horizon, missing timestamp policy, known-future feature policy,
and whether the data is a smoke sample or full benchmark input.

Small M4 or Monash samples are smoke artifacts. Do not describe them as
full-corpus results.

## Splits

Forecasting tasks use rolling-origin evaluation with fixed horizons. Model
selection uses only data before each cutoff. Reports should include
horizon-wise metrics so one-step accuracy cannot hide long-horizon failure.

The same cutoffs, horizons, feature availability, and missing-data rules apply
to every model family.

## Baselines

Report at least:

- `cartoboost_lag`;
- at least one statistical forecasting baseline;
- at least one global or neural forecasting baseline when dependencies and data
  shape allow it.

If a required baseline fails, the benchmark is incomplete unless the report is
specifically auditing that failure.

## Metrics

- MASE;
- RMSE;
- MAE;
- WAPE;
- sMAPE;
- horizon-wise RMSE or MAE.

- coverage at reported levels, such as 80% and 95%;
- interval width;
- interval score or CRPS when intervals or distributions are produced.

Do not write placeholder coverage values. If intervals are unavailable, omit
coverage claims or wrap point forecasts in a documented conformal protocol.

## Reproduce

Real taxi panels:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source nyc-taxi \
  --year 2024 \
  --months 1 \
  --taxi-type yellow \
  --lanes 24 \
  --horizon 7 \
  --no-download \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json
```

M4 smoke sample:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m4 \
  --m4-suite \
  --m4-series-limit 24 \
  --output target/forecasting_m4_suite_sample.json
```

These commands produce benchmark artifacts. Do not turn their output into a
claim unless the report also shows the split, budget, baseline set, and interval
handling.

## Reporting Rule

Report ties plainly. If seasonal naive is the selected model on a short taxi
panel, say so. Claim CartoBoost forecasting evidence only when
`cartoboost_lag` satisfies the primary metric threshold under the same
rolling-origin split and complete baseline set.
