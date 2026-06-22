# Model Benchmark Suite

## Bottom Line

The standard model suite is a bounded regression benchmark for checking
ordinary tabular behavior, public graph-regression plumbing, validation-search
discipline, timing, and artifact reporting before making broader claims.

The current maintained public run uses seed 42, a 5,000-row deterministic sample
for California housing, and a bounded inner-validation search with three
candidates per tunable model before final holdout scoring. It requests
LightGBM, XGBoost, CatBoost, scikit-learn HistGradientBoosting, RandomForest,
ExtraTrees, Ridge, mean, and graph-specific diagnostic rows where applicable.
The artifact also repeats the same protocol with seeds 42, 43, and 44 for
comparison intervals.

This run does not support a CartoBoost winner claim. The best completed
external baseline has lower RMSE than the single current-code `cartoboost` row
on diabetes, California housing, karate random, and karate group holdout.
CartoBoost is close to XGBoost on the California housing sample, but
HistGradientBoosting is clearly lower RMSE in this maintained run.

The command requests LightGBM and CatBoost. This artifact records both as
skipped in the local Python 3.13 benchmark environment: LightGBM imports only a
namespace package without `LGBMRegressor`, and CatBoost is not installed. The
completed external-baseline comparison therefore uses XGBoost,
HistGradientBoosting, RandomForest, ExtraTrees, Ridge, and mean.

## Reproduce

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_model_benchmark_suite.py \
  --output-dir docs/assets/model_benchmarks_public \
  --datasets diabetes,california_housing,karate \
  --n-rows 5000 \
  --models mean,cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,node2vec_regressor,graphsage_regressor \
  --n-estimators 24 \
  --graph-dim 4 \
  --graph-epochs 2 \
  --selection-mode validation_search \
  --validation-trials 3 \
  --repeat-seeds 42,43,44 \
  --no-plots
```

Artifacts:

- `docs/assets/model_benchmarks_public/results.json`
- `docs/assets/model_benchmarks_public/results.jsonl`
- `docs/assets/model_benchmarks_public/results_aggregate.json`
- `docs/assets/model_benchmarks_public/results.md`

`results.json` and `results.md` include the runtime resource snapshot and output
artifact sizes for this run: `results.json` 255,706 bytes, `results.jsonl`
110,167 bytes, and `results.md` 16,317 bytes.

## Baseline Environment

| Key | Package | Import | Version | Required class available |
| --- | --- | --- | --- | ---: |
| scikit-learn | `scikit-learn` | `sklearn` | `1.9.0` |  |
| XGBoost | `xgboost` | `xgboost` | `3.3.0` | true |
| LightGBM | `lightgbm` | `lightgbm` | unavailable metadata | false |
| CatBoost | `catboost` | `catboost` | not installed | false |

## Selection and Leakage Policy

- Tunable model families choose from a three-candidate grid on deterministic
  inner validation rows drawn only from the outer training split.
- The public CartoBoost comparison uses one validation-selected `cartoboost`
  row retrained on the full outer training split; graph, neural, and
  link-prediction rows are diagnostics.
- Neural and graph feature gates use deterministic inner train/validation rows
  inside the training split only.
- The best external baseline is selected only for reporting after every model
  has already been scored on the same held-out split.

## Dataset Sources

| Workload | Source | Rows | Features | SHA-256 fingerprint |
| --- | --- | ---: | ---: | --- |
| Diabetes | `sklearn.datasets.load_diabetes` bundled public regression dataset. | 442 | 10 | `d0e115e7bf84c3d7f4c1b43e7e1cb0bf35cd01ad1e0fd239320748b66f1f3888` |
| California housing | `sklearn.datasets.fetch_california_housing` deterministic 5,000-row seed-42 sample from the 20,640-row public California housing dataset. | 5,000 | 8 | `d0f75cd29b2fa35166c72d168c78cd2f206ab5b1c2d6a29e38437c55d3fa77ad` |
| Karate | Embedded Zachary karate club edge list and post-split labels from the benchmark harness constants. | 78 | 5 | `069058a0030b0e4859fbfb8254bc70c9f73eceb83c0fad5e2f1eba22352a6824` |

## Comparison Summary

For each regression split, this table compares the single primary `cartoboost`
row with the lowest-RMSE external baseline that finished under the same split
and global benchmark settings.

| Workload / split | CartoBoost RMSE | CartoBoost WAPE | Best external baseline | External RMSE | External WAPE | RMSE delta | R2 delta | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Diabetes / random | 52.9608 | 0.2834 | Ridge | 51.5180 | 0.2657 | +1.4429 | -0.0237 | External lower RMSE |
| California housing / random | 0.6301 | 0.2264 | HistGradientBoosting | 0.5958 | 0.2138 | +0.0342 | -0.0307 | External lower RMSE |
| Karate / random | 0.2708 | 0.2233 | XGBoost | 0.0488 | 0.0409 | +0.2220 | -1.2108 | External lower RMSE |
| Karate / group holdout | 0.3019 | 0.2641 | XGBoost | 0.2584 | 0.1157 | +0.0435 | -0.2387 | External lower RMSE |

## Repeated Comparison

The repeated comparison uses seeds 42, 43, and 44 with the same model roster,
validation-search budget, split policy, and dataset definitions. Negative RMSE
and WAPE deltas favor CartoBoost; positive R2 deltas favor CartoBoost.

| Workload / split | Best external baseline counts | RMSE delta mean | RMSE delta 95% CI | WAPE delta mean | R2 delta mean | R2 delta 95% CI | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| California housing / random | HistGradientBoosting: 3 | +0.030574 | +0.026971 to +0.034176 | +0.012214 | -0.029786 | -0.030655 to -0.028918 | External lower RMSE |
| Diabetes / random | HistGradientBoosting: 1, Ridge: 2 | +1.777059 | +0.255655 to +3.298462 | +0.014558 | -0.031123 | -0.057544 to -0.004703 | External lower RMSE |
| Karate / group holdout | RandomForest: 1, Ridge: 1, XGBoost: 1 | +0.099368 | +0.011556 to +0.187179 | +0.120028 | -0.176090 | -0.351045 to -0.001136 | External lower RMSE |
| Karate / random | ExtraTrees: 1, XGBoost: 2 | +0.110431 | +0.001041 to +0.219821 | +0.135488 | -0.575543 | -1.199597 to +0.048511 | External lower RMSE |

## Validation Search Selections

The table records the selected inner-validation candidate for the primary
CartoBoost row and the best external baseline on each split. Full candidate
tables are in `docs/assets/model_benchmarks_public/results.md`.

| Workload / split | Model | Selected trial | Validation RMSE | Inner train rows | Inner validation rows | Selected config |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Diabetes / random | CartoBoost | 2 | 59.4120 | 283 | 70 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 18}` |
| Diabetes / random | Ridge | 1 | 56.2885 | 283 | 70 | `{"ridge_alpha": 0.1}` |
| California housing / random | CartoBoost | 1 | 0.6444 | 3,200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| California housing / random | HistGradientBoosting | 1 | 0.6286 | 3,200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| Karate / random | CartoBoost | 2 | 0.2465 | 50 | 12 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 18}` |
| Karate / random | XGBoost | 2 | 0.2817 | 50 | 12 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 18}` |
| Karate / group holdout | CartoBoost | 1 | 0.3821 | 42 | 10 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| Karate / group holdout | XGBoost | 1 | 0.0765 | 42 | 10 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |

## Full Model Metrics

### Diabetes Random

| Model | RMSE | MAE | R2 | WAPE | Train s | Predict rows/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge | 51.5180 | 42.0061 | 0.5834 | 0.2657 | 0.0004 | 1,634,258 |
| cartoboost | 52.9608 | 44.7950 | 0.5597 | 0.2834 | 0.1255 | 242,838 |
| hist_gradient_boosting | 53.0690 | 44.2315 | 0.5579 | 0.2798 | 0.1946 | 14,394 |
| random_forest | 54.1006 | 44.6537 | 0.5405 | 0.2825 | 0.0377 | 5,210 |
| xgboost | 54.4905 | 45.8271 | 0.5339 | 0.2899 | 0.0329 | 153,669 |
| extra_trees | 55.2261 | 45.7889 | 0.5212 | 0.2897 | 0.0197 | 5,360 |
| mean | 80.1589 | 66.0195 | -0.0087 | 0.4177 | 0.0000 | 13,185,085 |

### California Housing Random

| Model | RMSE | MAE | R2 | WAPE | Train s | Predict rows/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hist_gradient_boosting | 0.5958 | 0.4446 | 0.7407 | 0.2138 | 0.3416 | 185,441 |
| cartoboost | 0.6301 | 0.4707 | 0.7100 | 0.2264 | 1.8348 | 194,373 |
| xgboost | 0.6304 | 0.4727 | 0.7098 | 0.2273 | 0.0433 | 2,042,901 |
| random_forest | 0.6891 | 0.5103 | 0.6531 | 0.2454 | 0.0538 | 58,617 |
| ridge | 0.7151 | 0.5250 | 0.6265 | 0.2525 | 0.0005 | 14,678,906 |
| extra_trees | 0.7891 | 0.5896 | 0.5452 | 0.2836 | 0.0198 | 60,567 |
| mean | 1.1702 | 0.9268 | -0.0002 | 0.4457 | 0.0000 | 134,825,496 |

### Karate Random

| Model | RMSE | MAE | R2 | WAPE | Train s | Predict rows/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.0488 | 0.0383 | 0.9593 | 0.0409 | 0.0293 | 36,564 |
| random_forest | 0.1079 | 0.0799 | 0.8012 | 0.0852 | 0.0207 | 941 |
| extra_trees | 0.1847 | 0.1500 | 0.4180 | 0.1601 | 0.0152 | 1,085 |
| ridge | 0.2281 | 0.1728 | 0.1121 | 0.1843 | 0.0004 | 279,072 |
| hist_gradient_boosting | 0.2547 | 0.2023 | -0.1069 | 0.2158 | 0.0413 | 5,807 |
| graphsage_regressor | 0.2547 | 0.2023 | -0.1069 | 0.2158 | 0.0110 | 29,286 |
| node2vec_regressor | 0.2588 | 0.1902 | -0.1431 | 0.2029 | 0.0136 | 46,366 |
| mean | 0.2614 | 0.2036 | -0.1666 | 0.2172 | 0.0000 | 4,625,627 |
| cartoboost | 0.2708 | 0.2093 | -0.2515 | 0.2233 | 0.0111 | 92,731 |

### Karate Group Holdout

| Model | RMSE | MAE | R2 | WAPE | Train s | Predict rows/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost | 0.2584 | 0.1024 | 0.3458 | 0.1157 | 0.0391 | 47,084 |
| extra_trees | 0.2720 | 0.1351 | 0.2752 | 0.1527 | 0.0176 | 1,578 |
| ridge | 0.2988 | 0.2015 | 0.1252 | 0.2278 | 0.0003 | 483,343 |
| cartoboost | 0.3019 | 0.2337 | 0.1071 | 0.2641 | 0.0098 | 153,316 |
| random_forest | 0.3075 | 0.1351 | 0.0738 | 0.1528 | 0.0333 | 1,502 |
| hist_gradient_boosting | 0.3149 | 0.2314 | 0.0288 | 0.2616 | 0.0628 | 3,883 |
| graphsage_regressor | 0.3149 | 0.2314 | 0.0288 | 0.2616 | 0.0077 | 40,245 |
| node2vec_regressor | 0.3184 | 0.2355 | 0.0069 | 0.2663 | 0.0108 | 64,750 |
| mean | 0.3218 | 0.2337 | -0.0145 | 0.2642 | 0.0000 | 7,009,991 |

## Interpretation

Use this page to diagnose benchmark plumbing, leakage-safe selection, and
model-family behavior. The maintained run now includes a larger public tabular
workload in addition to the small diabetes and karate fixtures, but it is still
a bounded single-seed benchmark. External baselines are stronger on every
maintained split, so any reusable CartoBoost model improvement should be
justified by larger real-data evidence and then rerun through this fixed
protocol before public claims change.
