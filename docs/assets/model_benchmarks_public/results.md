# Model Benchmark Suite

This generated report compares the primary CartoBoost regressor against optional external baselines on deterministic public tabular workloads and embedded graph diagnostics.

## Command

`PYTHONPATH=python uv run --group dev --group bench python scripts/run_model_benchmark_suite.py ...`

Command arguments:

`scripts/run_model_benchmark_suite.py --output-dir docs/assets/model_benchmarks_public --datasets diabetes,california_housing,karate --n-rows 5000 --models mean,cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,node2vec_regressor,graphsage_regressor --n-estimators 24 --graph-dim 4 --graph-epochs 2 --selection-mode validation_search --validation-trials 3 --repeat-seeds 42,43,44 --no-plots`

## Configuration

- Seed: `42`
- Datasets requested: `diabetes, california_housing, karate`
- Rows per workload: `5000`
- Train fraction: `0.8`
- Selection mode: `validation_search`
- Validation trials per tunable model: `3`
- Models requested: `mean, cartoboost, lightgbm, xgboost, catboost, hist_gradient_boosting, random_forest, extra_trees, ridge, node2vec_regressor, graphsage_regressor`

## Resource Usage

| Field | Value |
| --- | --- |
| cpu | `arm` |
| threads | `10` |
| os | `macOS-26.5.1-arm64-arm-64bit-Mach-O` |
| python | `3.13.12` |
| numpy | `2.4.6` |
| rustc | `rustc 1.94.0 (4a4ef493e 2026-03-02)` |

## Baseline Dependency Status

| Key | Package | Import | Version | Module importable | Required class | Required class available |
| --- | --- | --- | --- | ---: | --- | ---: |
| catboost | catboost | catboost | `None` | False | CatBoostRegressor | False |
| lightgbm | lightgbm | lightgbm | `None` | True | LGBMRegressor | False |
| sklearn | scikit-learn | sklearn | `1.9.0` | True |  |  |
| xgboost | xgboost | xgboost | `3.3.0` | True | XGBRegressor | True |

## Output Artifacts

| Artifact | Size bytes |
| --- | ---: |
| `results.json` | 255706 |
| `results.jsonl` | 110167 |
| `results.md` | 16317 |

## Selection and Leakage Policy

- global hyperparameters: fixed before holdout scoring; no model family uses test labels for tuning
- primary cartoboost row: single configured cartoboost run; no internal candidate is selected on test metrics
- neural feature gate: uses deterministic inner train/validation rows inside the training split only
- graph feature gate: uses deterministic inner train/validation rows inside the training split only
- external baseline selection: best external baseline is selected only for reporting after each model is scored
- diagnostic rows: graph, neural, and link-prediction rows are diagnostics and are not substitutes for the primary cartoboost comparison row

## Split Definitions

| Split | Kind | Train fraction | Purpose |
| --- | --- | --- | --- |
| random | seeded_row_shuffle | configured_by_--train-frac | interpolation across rows drawn from the same workload distribution |
| group_holdout | seeded_group_holdout | configured_by_--train-frac_over_unique_groups | cold-group generalization for workloads with repeated IDs or graph sources |

## Dataset Sources

| Workload | Source | Rows | Features | SHA-256 fingerprint |
| --- | --- | ---: | ---: | --- |
| diabetes | sklearn.datasets.load_diabetes bundled public regression dataset. | 442 | 10 | `d0e115e7bf84c3d7f4c1b43e7e1cb0bf35cd01ad1e0fd239320748b66f1f3888` |
| california_housing | sklearn.datasets.fetch_california_housing deterministic 5000-row seed-42 sample from the 20,640-row public California housing dataset. | 5000 | 8 | `d0f75cd29b2fa35166c72d168c78cd2f206ab5b1c2d6a29e38437c55d3fa77ad` |
| karate | Embedded Zachary karate club edge list and post-split labels from the benchmark harness constants. | 78 | 5 | `069058a0030b0e4859fbfb8254bc70c9f73eceb83c0fad5e2f1eba22352a6824` |

## Results

## CartoBoost vs External Baselines

For each regression split, this table compares the single primary `cartoboost` row with the lowest-RMSE external baseline that finished under the same data split and global benchmark settings.

| Workload | Split | CartoBoost RMSE | CartoBoost WAPE | Best external baseline | External RMSE | External WAPE | RMSE delta | R2 delta | Result |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| diabetes | random | 52.9608 | 0.2834 | ridge | 51.5180 | 0.2657 | 1.4429 | -0.0237 | external_lower_or_tied_rmse |
| california_housing | random | 0.6301 | 0.2264 | hist_gradient_boosting | 0.5958 | 0.2138 | 0.0342 | -0.0307 | external_lower_or_tied_rmse |
| karate | random | 0.2708 | 0.2233 | xgboost | 0.0488 | 0.0409 | 0.2220 | -1.2108 | external_lower_or_tied_rmse |
| karate | group_holdout | 0.3019 | 0.2641 | xgboost | 0.2584 | 0.1157 | 0.0435 | -0.2387 | external_lower_or_tied_rmse |

### Interpretation Notes

- Dense public or synthetic workloads are baseline sanity checks for ordinary tabular regression behavior without graph or neural inputs.
- Neural workloads, when requested, show the difference between repeated-ID and cold-ID claims. Neural and graph rows are diagnostics and are not used as substitutes for the primary `cartoboost` comparison row.
- The graph workload separates two surfaces. Augmented CartoBoost uses graph features as extra columns for the booster, while standalone GraphSAGE-style regressors and link predictors can score graph tasks without a boosted wrapper. The link-predictor rows report AUC/AP because they are ranking candidate source-target edges, not predicting the regression target.
- External baseline rows use the same train/test split and global benchmark settings; no test labels are used for model selection.

## Repeated External Baseline Comparison

Repeated rows use the same model roster, validation-search budget, and split policy with different deterministic seeds. Negative RMSE and WAPE deltas favor CartoBoost; positive R2 deltas favor CartoBoost.

| Workload | Split | Seeds | Best external baseline counts | RMSE delta mean | RMSE delta 95% CI | WAPE delta mean | R2 delta mean | R2 delta 95% CI | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| california_housing | random | 42, 43, 44 | hist_gradient_boosting: 3 | 0.030574 | 0.026971 to 0.034176 | 0.012214 | -0.029786 | -0.030655 to -0.028918 | external_lower_rmse |
| diabetes | random | 42, 43, 44 | hist_gradient_boosting: 1, ridge: 2 | 1.777059 | 0.255655 to 3.298462 | 0.014558 | -0.031123 | -0.057544 to -0.004703 | external_lower_rmse |
| karate | group_holdout | 42, 43, 44 | random_forest: 1, ridge: 1, xgboost: 1 | 0.099368 | 0.011556 to 0.187179 | 0.120028 | -0.176090 | -0.351045 to -0.001136 | external_lower_rmse |
| karate | random | 42, 43, 44 | extra_trees: 1, xgboost: 2 | 0.110431 | 0.001041 to 0.219821 | 0.135488 | -0.575543 | -1.199597 to 0.048511 | external_lower_rmse |

## Validation Search Selections

The table records the inner-validation winner for each tunable model. Final holdout metrics above are computed only after retraining the selected configuration on the full outer training split.

| Workload | Split | Model | Selected trial | Validation RMSE | Inner train rows | Inner validation rows | Selected config |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| diabetes | random | cartoboost | 2 | 59.411968 | 283 | 70 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 18}` |
| diabetes | random | xgboost | 3 | 58.800533 | 283 | 70 | `{"learning_rate": 0.08, "max_depth": 3, "n_estimators": 24}` |
| diabetes | random | hist_gradient_boosting | 3 | 59.861378 | 283 | 70 | `{"learning_rate": 0.08, "max_depth": 3, "n_estimators": 24}` |
| diabetes | random | random_forest | 3 | 59.247501 | 283 | 70 | `{"max_depth": 4, "min_samples_leaf": 5, "n_estimators": 24}` |
| diabetes | random | extra_trees | 3 | 58.247349 | 283 | 70 | `{"max_depth": 4, "min_samples_leaf": 5, "n_estimators": 24}` |
| diabetes | random | ridge | 1 | 56.288485 | 283 | 70 | `{"ridge_alpha": 0.1}` |
| california_housing | random | cartoboost | 1 | 0.644431 | 3200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| california_housing | random | xgboost | 1 | 0.649380 | 3200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| california_housing | random | hist_gradient_boosting | 1 | 0.628613 | 3200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| california_housing | random | random_forest | 3 | 0.709553 | 3200 | 800 | `{"max_depth": 4, "min_samples_leaf": 5, "n_estimators": 24}` |
| california_housing | random | extra_trees | 1 | 0.763658 | 3200 | 800 | `{"max_depth": 4, "min_samples_leaf": 2, "n_estimators": 24}` |
| california_housing | random | ridge | 3 | 0.700590 | 3200 | 800 | `{"ridge_alpha": 10.0}` |
| karate | random | cartoboost | 2 | 0.246528 | 50 | 12 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 18}` |
| karate | random | xgboost | 2 | 0.281668 | 50 | 12 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 18}` |
| karate | random | hist_gradient_boosting | 1 | 0.282902 | 50 | 12 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| karate | random | random_forest | 2 | 0.285372 | 50 | 12 | `{"max_depth": 3, "min_samples_leaf": 2, "n_estimators": 24}` |
| karate | random | extra_trees | 2 | 0.253762 | 50 | 12 | `{"max_depth": 3, "min_samples_leaf": 2, "n_estimators": 24}` |
| karate | random | ridge | 3 | 0.272913 | 50 | 12 | `{"ridge_alpha": 10.0}` |
| karate | group_holdout | cartoboost | 1 | 0.382089 | 42 | 10 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| karate | group_holdout | xgboost | 1 | 0.076476 | 42 | 10 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| karate | group_holdout | hist_gradient_boosting | 1 | 0.348748 | 42 | 10 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| karate | group_holdout | random_forest | 1 | 0.242643 | 42 | 10 | `{"max_depth": 4, "min_samples_leaf": 2, "n_estimators": 24}` |
| karate | group_holdout | extra_trees | 1 | 0.188111 | 42 | 10 | `{"max_depth": 4, "min_samples_leaf": 2, "n_estimators": 24}` |
| karate | group_holdout | ridge | 1 | 0.233270 | 42 | 10 | `{"ridge_alpha": 0.1}` |

### sklearn diabetes

Frozen public scikit-learn diabetes regression workload with 442 rows, 10 numeric features, and disease-progression target.

#### random

Train rows: `353`; test rows: `89`.
Train index SHA-256: `c283a0fd7785bad10b7846411aa608d952724a4167e18f73ef589a64308a908a`; test index SHA-256: `a821cc67f044dc1884b43cca8b91624118475c7270b7c3a61b8296d5b772d654`.

| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 66.0195 | 80.1589 | -0.0087 | 0.4177 | 0.0000 | 13185085 |
| cartoboost | ok | 44.7950 | 52.9608 | 0.5597 | 0.2834 | 0.1255 | 242838 |
| lightgbm | skipped: all validation-search candidates failed: lightgbm is not installed |  |  |  |  |  |  |
| xgboost | ok | 45.8271 | 54.4905 | 0.5339 | 0.2899 | 0.0329 | 153669 |
| catboost | skipped: all validation-search candidates failed: catboost is not installed |  |  |  |  |  |  |
| hist_gradient_boosting | ok | 44.2315 | 53.0690 | 0.5579 | 0.2798 | 0.1946 | 14394 |
| random_forest | ok | 44.6537 | 54.1006 | 0.5405 | 0.2825 | 0.0377 | 5210 |
| extra_trees | ok | 45.7889 | 55.2261 | 0.5212 | 0.2897 | 0.0197 | 5360 |
| ridge | ok | 42.0061 | 51.5180 | 0.5834 | 0.2657 | 0.0004 | 1634258 |
| node2vec_regressor | skipped: workload has no graph topology |  |  |  |  |  |  |
| graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |  |

### California housing

Public scikit-learn California housing regression workload with eight numeric census-block features and median house value target.

#### random

Train rows: `4000`; test rows: `1000`.
Train index SHA-256: `c852c7fee0c7fe3a52c7053306e50afd8983882fa30ceb44aa4ffb9ea599f668`; test index SHA-256: `f9a7e8c6c3732700e2e7f6d168a253178d2dae578216a06e4d988d05bce6ead4`.

| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 0.9268 | 1.1702 | -0.0002 | 0.4457 | 0.0000 | 134825496 |
| cartoboost | ok | 0.4707 | 0.6301 | 0.7100 | 0.2264 | 1.8348 | 194373 |
| lightgbm | skipped: all validation-search candidates failed: lightgbm is not installed |  |  |  |  |  |  |
| xgboost | ok | 0.4727 | 0.6304 | 0.7098 | 0.2273 | 0.0433 | 2042901 |
| catboost | skipped: all validation-search candidates failed: catboost is not installed |  |  |  |  |  |  |
| hist_gradient_boosting | ok | 0.4446 | 0.5958 | 0.7407 | 0.2138 | 0.3416 | 185441 |
| random_forest | ok | 0.5103 | 0.6891 | 0.6531 | 0.2454 | 0.0538 | 58617 |
| extra_trees | ok | 0.5896 | 0.7891 | 0.5452 | 0.2836 | 0.0198 | 60567 |
| ridge | ok | 0.5250 | 0.7151 | 0.6265 | 0.2525 | 0.0005 | 14678906 |
| node2vec_regressor | skipped: workload has no graph topology |  |  |  |  |  |  |
| graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |  |

### Zachary karate club

Frozen public 78-edge Zachary karate club graph workload. Rows are observed edges; the regression target is whether the two endpoints share the same post-split club label.

#### random

Train rows: `62`; test rows: `16`.
Train index SHA-256: `1367cc4e05f89b99b101292f4105c89818e39baaec774ed5d7c5a522158c6832`; test index SHA-256: `cae5be975a23027d748116778eae74072f5097fc8a61653bf0bd63217e68f326`.

| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 0.2036 | 0.2614 | -0.1666 | 0.2172 | 0.0000 | 4625627 |
| cartoboost | ok | 0.2093 | 0.2708 | -0.2515 | 0.2233 | 0.0111 | 92731 |
| lightgbm | skipped: all validation-search candidates failed: lightgbm is not installed |  |  |  |  |  |  |
| xgboost | ok | 0.0383 | 0.0488 | 0.9593 | 0.0409 | 0.0293 | 36564 |
| catboost | skipped: all validation-search candidates failed: catboost is not installed |  |  |  |  |  |  |
| hist_gradient_boosting | ok | 0.2023 | 0.2547 | -0.1069 | 0.2158 | 0.0413 | 5807 |
| random_forest | ok | 0.0799 | 0.1079 | 0.8012 | 0.0852 | 0.0207 | 941 |
| extra_trees | ok | 0.1500 | 0.1847 | 0.4180 | 0.1601 | 0.0152 | 1085 |
| ridge | ok | 0.1728 | 0.2281 | 0.1121 | 0.1843 | 0.0004 | 279072 |
| node2vec_regressor | ok | 0.1902 | 0.2588 | -0.1431 | 0.2029 | 0.0136 | 46366 |
| graphsage_regressor | ok | 0.2023 | 0.2547 | -0.1069 | 0.2158 | 0.0110 | 29286 |

#### group_holdout

Train rows: `52`; test rows: `26`.
Train index SHA-256: `dddac4b5f12ac598f17c4aaea291a98aba3f738d686b682c8cf642a6f3ba1c3a`; test index SHA-256: `b10d872fdad08114783ad1e005843d965e6ef28616371c8395487296ed69d123`.

| Model | Status | MAE | RMSE | R2 | WAPE | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 0.2337 | 0.3218 | -0.0145 | 0.2642 | 0.0000 | 7009991 |
| cartoboost | ok | 0.2337 | 0.3019 | 0.1071 | 0.2641 | 0.0098 | 153316 |
| lightgbm | skipped: all validation-search candidates failed: lightgbm is not installed |  |  |  |  |  |  |
| xgboost | ok | 0.1024 | 0.2584 | 0.3458 | 0.1157 | 0.0391 | 47084 |
| catboost | skipped: all validation-search candidates failed: catboost is not installed |  |  |  |  |  |  |
| hist_gradient_boosting | ok | 0.2314 | 0.3149 | 0.0288 | 0.2616 | 0.0628 | 3883 |
| random_forest | ok | 0.1351 | 0.3075 | 0.0738 | 0.1528 | 0.0333 | 1502 |
| extra_trees | ok | 0.1351 | 0.2720 | 0.2752 | 0.1527 | 0.0176 | 1578 |
| ridge | ok | 0.2015 | 0.2988 | 0.1252 | 0.2278 | 0.0003 | 483343 |
| node2vec_regressor | ok | 0.2355 | 0.3184 | 0.0069 | 0.2663 | 0.0108 | 64750 |
| graphsage_regressor | ok | 0.2314 | 0.3149 | 0.0288 | 0.2616 | 0.0077 | 40245 |

## Interpretation Notes

- Dense workloads check numeric behavior without ID or graph augmentation.
- Neural workloads, when requested, include repeated IDs and a group holdout split, so `cartoboost_neural` should be read as an embedding augmentation check rather than a replacement for external neural networks.
- The graph workload benchmarks node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE feature augmentation from train topology before fitting CartoBoost on augmented source-target rows.
- Optional dependency rows are skipped when the corresponding benchmark dependency is not installed.
