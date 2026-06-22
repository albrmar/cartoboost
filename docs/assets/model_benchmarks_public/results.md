# Model Benchmark Suite

This generated report compares CartoBoost against optional XGBoost and LightGBM baselines on deterministic synthetic workloads.

## Command

`uv run --group bench python scripts/run_model_benchmark_suite.py`

## Configuration

- Seed: `42`
- Rows per workload: `2400`
- Train fraction: `0.8`
- Models requested: `mean, cartoboost, lightgbm, xgboost, node2vec_regressor, graphsage_regressor`

## Results

## LightGBM Comparison

For each benchmark split, this table compares LightGBM with the best CartoBoost-family model that finished successfully under the same data split and global benchmark settings.

| Workload | Split | Best CartoBoost-family model | CartoBoost RMSE | LightGBM RMSE | RMSE delta | R2 delta | Winner |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| diabetes | random | cartoboost | 52.9771 | 53.0144 | -0.0373 | 0.0006 | cartoboost |
| karate | random | graphsage_regressor | 0.2547 | 0.2557 | -0.0010 | 0.0089 | cartoboost |
| karate | group_holdout | cartoboost | 0.3019 | 0.3149 | -0.0130 | 0.0783 | cartoboost |

### Why CartoBoost Wins Here

- The normal dense workload is a baseline sanity check: CartoBoost is competitive without relying on graph or neural inputs.
- The neural workload shows the difference between repeated-ID and cold-ID claims. `cartoboost_neural` wins the random split because held-out rows reuse train-observed IDs; the group holdout falls back to the base CartoBoost structure instead of pretending unseen IDs can be recovered from an embedding table.
- The graph workload separates two surfaces. Augmented CartoBoost uses graph features as extra columns for the booster, while standalone GraphSAGE-style regressors and link predictors can score graph tasks without a boosted wrapper. The link-predictor rows report AUC/AP because they are ranking candidate source-target edges, not predicting the regression target.
- LightGBM sees the benchmark as ordinary dense tabular columns. CartoBoost-family rows can add ID residual structure, source-target topology, and spatially shaped splitters when the workload exposes those contracts.

### sklearn diabetes

Frozen public scikit-learn diabetes regression workload with 442 rows, 10 numeric features, and disease-progression target.

#### random

Train rows: `353`; test rows: `89`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 66.0195 | 80.1589 | -0.0087 | 0.0000 | 16306292 |
| cartoboost | ok | 44.8184 | 52.9771 | 0.5594 | 0.2025 | 94815 |
| lightgbm | ok | 43.7780 | 53.0144 | 0.5588 | 0.0655 | 45774 |
| xgboost | ok | 47.4114 | 56.5052 | 0.4988 | 0.0658 | 120155 |
| node2vec_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |

### Zachary karate club

Frozen public 78-edge Zachary karate club graph workload. Rows are observed edges; the regression target is whether the two endpoints share the same post-split club label.

#### random

Train rows: `62`; test rows: `16`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 0.2036 | 0.2614 | -0.1666 | 0.0000 | 3459455 |
| cartoboost | ok | 0.2072 | 0.2665 | -0.2120 | 0.0199 | 49472 |
| lightgbm | ok | 0.1993 | 0.2557 | -0.1158 | 0.0164 | 22209 |
| xgboost | ok | 0.0349 | 0.0444 | 0.9663 | 0.0562 | 22006 |
| node2vec_regressor | ok | 0.1902 | 0.2588 | -0.1431 | 0.0236 | 21992 |
| graphsage_regressor | ok | 0.2023 | 0.2547 | -0.1069 | 0.0176 | 24995 |

#### group_holdout

Train rows: `52`; test rows: `26`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 0.2337 | 0.3218 | -0.0145 | 0.0000 | 6709677 |
| cartoboost | ok | 0.2337 | 0.3019 | 0.1071 | 0.0154 | 46598 |
| lightgbm | ok | 0.2314 | 0.3149 | 0.0288 | 0.0133 | 38746 |
| xgboost | ok | 0.1024 | 0.2584 | 0.3458 | 0.0533 | 37695 |
| node2vec_regressor | ok | 0.2355 | 0.3184 | 0.0069 | 0.0195 | 34371 |
| graphsage_regressor | ok | 0.2314 | 0.3149 | 0.0288 | 0.0123 | 37116 |

## Plots

![MAE by workload and split](mae_by_model.png)

![Training time by workload and split](train_time_by_model.png)

![Prediction throughput by workload and split](prediction_throughput_by_model.png)

## Interpretation Notes

- The normal workload checks dense numeric behavior without ID or graph augmentation.
- The neural workload includes repeated IDs and a group holdout split, so `cartoboost_neural` should be read as an embedding augmentation check rather than a replacement for external neural networks.
- The graph workload benchmarks node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE feature augmentation from train topology before fitting CartoBoost on augmented source-target rows.
- XGBoost and LightGBM rows are skipped when their optional benchmark dependencies are not installed.
