# NYC Taxi Benchmarks

## Bottom Line

A bounded current-code run on real January 2024 NYC TLC yellow taxi data compares
one primary `cartoboost` row with XGBoost, scikit-learn HistGradientBoosting,
RandomForest, ExtraTrees, Ridge, and a mean baseline under fixed settings. In
the maintained single-seed artifact, CartoBoost has lower RMSE than the best
external baseline on duration random, duration spatial holdout, and
pickup-demand random splits. Ridge is slightly lower on fare random and fare
spatial holdout.

A three-seed repeated run preserves the same direction for duration and
pickup-demand random, shows Ridge ahead on fare random, and leaves fare spatial
holdout mixed because the paired RMSE interval crosses zero.

This is bounded evidence, not a broad winner claim: it is one month, a 30,000
trip-row sample for row-level tasks, fixed hyperparameters, and local-hardware
timing.

## Data

| Field | Value |
| --- | --- |
| Source | NYC TLC trip records |
| Source URL | [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Taxi type | Yellow |
| Period | January 2024 |
| Trip sample size | 30,000 rows |
| Duration rows | 30,000 |
| Fare rows | 30,000 |
| Pickup-demand rows | 24,650 |
| Dataset hash | `741a94b7345cd469a8dc6261b116910f39131f6e1ca0e824dd319e53ef6bd8c8` |
| Zone treatment | Train-only smoothed target-mean zone features for all eligible models |

Raw TLC files stay under `data/nyc_taxi/` and are not committed. The run used
`--no-download`, so missing local real inputs would hard-fail.

## Reproduce

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-download \
  --no-plots \
  --sample-size 30000 \
  --output-dir docs/assets/nyc_taxi_benchmarks \
  --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,mean \
  --n-estimators 24 \
  --cartoboost-n-estimators 24 \
  --tasks duration,fare,pickup_demand \
  --model-workers 1
```

Generated artifacts:

- `docs/assets/nyc_taxi_benchmarks/results.json`
- `docs/assets/nyc_taxi_benchmarks/results.jsonl`
- `docs/assets/nyc_taxi_benchmarks/results.md`

The JSON and Markdown artifacts record the runtime resource snapshot, baseline
dependency status, and output artifact sizes. LightGBM and CatBoost are
requested and are included in comparisons only when their optional estimators
are available.

| Baseline | Package | Import | Version | Importable | Required class | Class available |
| --- | --- | --- | --- | --- | --- | --- |
| sklearn | scikit-learn | sklearn | `1.9.0` | true |  |  |
| xgboost | xgboost | xgboost | `3.3.0` | true | XGBRegressor | true |
| lightgbm | lightgbm | lightgbm | `None` | true | LGBMRegressor | false |
| catboost | catboost | catboost | `None` | false | CatBoostRegressor | false |

The command uses `--no-plots`, so it refreshes the maintained tabular evidence
without changing generated PNG assets.

Repeated-run confidence intervals use:

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_repeated_nyc_taxi_benchmarks.py \
  --runs 3 \
  --run-dir target/repeated-nyc-expanded-real-30k-maintained \
  --summary-json docs/assets/nyc_taxi_benchmarks/repeated_results.json \
  --summary-md docs/assets/nyc_taxi_benchmarks/repeated_results.md \
  --no-download \
  --no-plots \
  --sample-size 30000 \
  --tasks duration,fare,pickup_demand \
  --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,mean \
  --n-estimators 24 \
  --cartoboost-n-estimators 24 \
  --cartoboost-splitters axis_histogram:512,diagonal_2d,gaussian_2d,periodic:24,periodic:7,sparse_set \
  --cartoboost-min-samples-leaf 20 \
  --model-workers 1 \
  --n-threads 1
```

Repeated artifacts:

- `docs/assets/nyc_taxi_benchmarks/repeated_results.json`
- `docs/assets/nyc_taxi_benchmarks/repeated_results.md`

The repeated summary artifacts include output artifact sizes for the summary
JSON and Markdown files.

## Comparison Summary

For each runnable learned-model split, the table compares the single primary
`cartoboost` row with the lowest-RMSE external baseline that finished under the
same task, split, sample, target transformation, and global settings.

| Task / split | CartoBoost RMSE | CartoBoost WAPE | Best external baseline | External RMSE | External WAPE | RMSE delta | R2 delta | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Duration / random | 0.321631 | 0.037386 | HistGradientBoosting | 0.337427 | 0.039421 | -0.015796 | +0.021386 | CartoBoost lower RMSE |
| Duration / spatial holdout | 0.317218 | 0.037839 | HistGradientBoosting | 0.330881 | 0.039590 | -0.013663 | +0.021329 | CartoBoost lower RMSE |
| Fare / random | 0.174649 | 0.040675 | Ridge | 0.169843 | 0.038907 | +0.004807 | -0.006014 | External lower RMSE |
| Fare / spatial holdout | 0.159090 | 0.039392 | Ridge | 0.158739 | 0.039154 | +0.000350 | -0.000692 | External lower RMSE |
| Pickup demand / random | 0.598105 | 0.179037 | HistGradientBoosting | 0.631339 | 0.191566 | -0.033234 | +0.009805 | CartoBoost lower RMSE |

The pickup-demand spatial holdout skips learned models because held-out pickup
zones have no training-side demand history. Reporting learned-model scores there
would mostly measure fallback priors.

## Selection and Leakage Policy

- Global hyperparameters are fixed before holdout scoring; no model family uses
  test labels for tuning.
- The public CartoBoost comparison uses one configured `cartoboost` row; no
  internal CartoBoost candidate is selected from test metrics.
- Target-mean zone encodings are fit on training rows for each outer split
  before transforming holdout rows.
- Graph and neural feature gates use deterministic inner train/validation rows
  inside the training split only.
- Pickup-zone segment diagnostics are computed after prediction and are excluded
  from fitting, tuning, and model selection.

## Repeated Confidence Intervals

The repeated run uses seeds 11, 29, and 47. Each row compares the primary
`cartoboost` result with the lowest-RMSE external baseline that finished in that
run. Negative RMSE and WAPE deltas favor CartoBoost; positive R2 deltas favor
CartoBoost.

| Task / split | Best external baseline counts | RMSE delta mean | RMSE delta 95% CI | WAPE delta mean | R2 delta mean | R2 delta 95% CI | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Duration / random | HistGradientBoosting: 3 | -0.014505 | -0.016121 to -0.012639 | -0.001843 | +0.019311 | +0.017008 to +0.021278 | CartoBoost lower RMSE |
| Duration / spatial holdout | HistGradientBoosting: 1, RandomForest: 1, XGBoost: 1 | -0.014260 | -0.017035 to -0.010027 | -0.002072 | +0.019047 | +0.015947 to +0.022415 | CartoBoost lower RMSE |
| Fare / random | Ridge: 3 | +0.004348 | +0.001972 to +0.005972 | +0.001684 | -0.005239 | -0.007013 to -0.002473 | External lower RMSE |
| Fare / spatial holdout | HistGradientBoosting: 1, RandomForest: 1, Ridge: 1 | +0.021544 | -0.005706 to +0.073071 | +0.003694 | -0.020020 | -0.074373 to +0.009312 | Mixed interval |
| Pickup demand / random | HistGradientBoosting: 2, RandomForest: 1 | -0.033195 | -0.036340 to -0.029866 | -0.007023 | +0.009827 | +0.008833 to +0.010901 | CartoBoost lower RMSE |

## Pickup-Zone Segment Diagnostics

These diagnostics are computed after prediction on each holdout split. They
summarize pickup-zone error distribution and are not used for training, model
selection, or tuning.

| Task / split | Model | Pickup zones | Zone rows min-max | Zone RMSE p50 | Zone RMSE p90 | Worst zone RMSE |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Duration / random | cartoboost | 126 | 1-302 | 0.322226 | 0.692416 | 2.272663 |
| Duration / random | HistGradientBoosting | 126 | 1-302 | 0.342045 | 0.698395 | 2.226535 |
| Duration / spatial holdout | cartoboost | 39 | 1-1525 | 0.358977 | 0.700164 | 1.641809 |
| Duration / spatial holdout | HistGradientBoosting | 39 | 1-1525 | 0.367757 | 0.718512 | 1.740194 |
| Fare / random | cartoboost | 126 | 1-302 | 0.161276 | 0.420249 | 1.238979 |
| Fare / random | Ridge | 126 | 1-302 | 0.169932 | 0.453957 | 1.461132 |
| Fare / spatial holdout | cartoboost | 39 | 1-1525 | 0.176255 | 0.481167 | 0.812295 |
| Fare / spatial holdout | Ridge | 39 | 1-1525 | 0.206107 | 0.488287 | 0.811945 |
| Pickup demand / random | cartoboost | 239 | 1-47 | 0.505235 | 0.833421 | 1.955022 |
| Pickup demand / random | HistGradientBoosting | 239 | 1-47 | 0.555335 | 0.899397 | 2.060286 |

## Full Model Metrics

### Duration Random

| Model | Status | RMSE | MAE | R2 | WAPE | Train s | Predict s | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.321631 | 0.244797 | 0.787495 | 0.037386 | 50.407 | 0.01150 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.340162 | 0.260461 | 0.762302 | 0.039779 | 0.041 | 0.00044 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.337427 | 0.258119 | 0.766108 | 0.039421 | 0.084 | 0.00098 | n_estimators=24 |
| random_forest | ok | 0.346062 | 0.266391 | 0.753984 | 0.040684 | 0.157 | 0.01747 | n_estimators=24 |
| extra_trees | ok | 0.360183 | 0.279085 | 0.733498 | 0.042623 | 0.046 | 0.01502 | n_estimators=24 |
| ridge | ok | 0.361002 | 0.277595 | 0.732285 | 0.042396 | 0.002 | 0.00009 |  |
| mean | ok | 0.697752 | 0.557510 | -0.000130 | 0.085145 | 0.000 | 0.00001 |  |

### Duration Spatial Holdout

| Model | Status | RMSE | MAE | R2 | WAPE | Train s | Predict s | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.317218 | 0.244041 | 0.757616 | 0.037839 | 51.411 | 0.01224 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.334581 | 0.258639 | 0.730355 | 0.040102 | 0.045 | 0.00048 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.330881 | 0.255337 | 0.736287 | 0.039590 | 0.241 | 0.00309 | n_estimators=24 |
| random_forest | ok | 0.339897 | 0.264278 | 0.721719 | 0.040977 | 0.197 | 0.01242 | n_estimators=24 |
| extra_trees | ok | 0.353708 | 0.273389 | 0.698645 | 0.042389 | 0.048 | 0.01643 | n_estimators=24 |
| ridge | ok | 0.354533 | 0.274377 | 0.697238 | 0.042542 | 0.002 | 0.00009 |  |
| mean | ok | 0.657256 | 0.519352 | -0.040539 | 0.080526 | 0.000 | 0.00001 |  |

### Fare Random

| Model | Status | RMSE | MAE | R2 | WAPE | Train s | Predict s | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.174649 | 0.128754 | 0.889225 | 0.040675 | 51.120 | 0.01328 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.182710 | 0.136996 | 0.878764 | 0.043279 | 0.048 | 0.00057 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.180789 | 0.134599 | 0.881300 | 0.042522 | 0.447 | 0.01610 | n_estimators=24 |
| random_forest | ok | 0.177191 | 0.129255 | 0.885978 | 0.040834 | 0.283 | 0.01479 | n_estimators=24 |
| extra_trees | ok | 0.181030 | 0.133740 | 0.880984 | 0.042251 | 0.057 | 0.01618 | n_estimators=24 |
| ridge | ok | 0.169843 | 0.123155 | 0.895239 | 0.038907 | 0.002 | 0.00010 |  |
| mean | ok | 0.524746 | 0.399845 | -0.000010 | 0.126318 | 0.000 | 0.00001 |  |

### Fare Spatial Holdout

| Model | Status | RMSE | MAE | R2 | WAPE | Train s | Predict s | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.159090 | 0.120233 | 0.842719 | 0.039392 | 46.511 | 0.01229 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.167078 | 0.127587 | 0.826527 | 0.041801 | 0.043 | 0.00043 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.163738 | 0.124610 | 0.833393 | 0.040826 | 0.147 | 0.00121 | n_estimators=24 |
| random_forest | ok | 0.166072 | 0.124522 | 0.828609 | 0.040797 | 0.173 | 0.01542 | n_estimators=24 |
| extra_trees | ok | 0.170498 | 0.129226 | 0.819354 | 0.042338 | 0.055 | 0.01658 | n_estimators=24 |
| ridge | ok | 0.158739 | 0.119507 | 0.843411 | 0.039154 | 0.002 | 0.00009 |  |
| mean | ok | 0.425543 | 0.344144 | -0.125328 | 0.112751 | 0.000 | 0.00001 |  |

### Pickup Demand Random

| Model | Status | RMSE | MAE | R2 | WAPE | Train s | Predict s | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.598105 | 0.481325 | 0.914156 | 0.179037 | 10.946 | 0.00838 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.647344 | 0.521361 | 0.899440 | 0.193929 | 0.036 | 0.00047 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.631339 | 0.515010 | 0.904351 | 0.191566 | 0.224 | 0.00330 | n_estimators=24 |
| random_forest | ok | 0.645452 | 0.486077 | 0.900027 | 0.180804 | 0.064 | 0.01719 | n_estimators=24 |
| extra_trees | ok | 0.697081 | 0.534807 | 0.883394 | 0.198930 | 0.033 | 0.01663 | n_estimators=24 |
| ridge | ok | 0.800820 | 0.606266 | 0.846105 | 0.225510 | 0.001 | 0.00008 |  |
| mean | ok | 2.041944 | 1.752775 | -0.000560 | 0.651973 | 0.000 | 0.00001 |  |

### Pickup Demand Spatial Holdout

| Model | Status | RMSE | MAE | R2 | WAPE | Train s | Predict s | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout |
| lightgbm | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout |
| xgboost | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout |
| catboost | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout |
| hist_gradient_boosting | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout |
| random_forest | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout |
| extra_trees | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout |
| ridge | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout |
| mean | ok | 2.088607 | 1.807484 | -0.002958 | 0.659779 | 0.000 | 0.00000 |  |

## Interpretation

The bounded evidence is mixed. CartoBoost is strongest on duration and
observed-zone pickup demand. Fare is close, but Ridge is lower on the random
split in both the single run and repeated summary, so these artifacts do not
support a fare winner claim for CartoBoost. CartoBoost also trains materially
slower than the external tree and linear baselines at these settings, though
prediction latency remains low.

## Limitations

- One January 2024 sample.
- Fixed settings, no equal-budget hyperparameter search.
- Transformed targets, not raw dollars or seconds.
- Repeated confidence intervals cover three seeds, not a full cross-month study.
- LightGBM was requested but skipped because `LGBMRegressor` was not available
  from the installed `lightgbm` import in the validated local environment.
- CatBoost was not included because it was not installed in the validated local environment.
- Timing is local-hardware specific.
