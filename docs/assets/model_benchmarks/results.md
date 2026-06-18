# Model Benchmark Suite

This generated report compares CartoBoost against optional XGBoost and LightGBM baselines on deterministic synthetic workloads.

## Command

`uv run --group dev --group bench python scripts/run_model_benchmark_suite.py`

## Configuration

- Seed: `42`
- Rows per workload: `2400`
- Train fraction: `0.8`
- Models requested: `mean, cartoboost, cartoboost_neural, cartoboost_graph_node2vec, cartoboost_graph_graphsage, cartoboost_graph_hetero_graphsage, cartoboost_graph_hinsage, xgboost, lightgbm`

## Results

### Normal dense

IID dense numeric regression with nonlinear feature interactions.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.7284 | 2.1533 | -0.0000 | 0.0000 | 159999946 |
| cartoboost | ok | 0.3821 | 0.5082 | 0.9443 | 0.0469 | 1809893 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.3810 | 0.5173 | 0.9423 | 0.0739 | 1119967 |
| lightgbm | ok | 0.3783 | 0.5080 | 0.9443 | 0.1013 | 134953 |

### Neural ID

Dense regression with repeated cell IDs whose residual signal is learnable by embedding features.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.3848 | 1.7144 | -0.0005 | 0.0000 | 230326922 |
| cartoboost | ok | 0.4420 | 0.5523 | 0.8962 | 0.0451 | 1868010 |
| cartoboost_neural | ok | 0.3757 | 0.4796 | 0.9217 | 0.0896 | 816500 |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4492 | 0.5636 | 0.8919 | 0.0707 | 1206285 |
| lightgbm | ok | 0.4291 | 0.5368 | 0.9019 | 0.0975 | 748344 |

#### group_holdout

Train rows: `1945`; test rows: `455`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.3599 | 1.6951 | -0.0034 | 0.0000 | 237473935 |
| cartoboost | ok | 0.4529 | 0.5740 | 0.8849 | 0.0448 | 1693288 |
| cartoboost_neural | ok | 0.6491 | 0.7876 | 0.7834 | 0.0910 | 818653 |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4686 | 0.5985 | 0.8749 | 0.0774 | 1117365 |
| lightgbm | ok | 0.4402 | 0.5677 | 0.8875 | 0.1045 | 686016 |

### Graph source-target

Directed source-target regression where graph topology and node features carry predictive signal.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2321 | 1.5445 | -0.0000 | 0.0000 | 213333089 |
| cartoboost | ok | 0.4123 | 0.5118 | 0.8902 | 0.0363 | 1792162 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | ok | 0.4316 | 0.5400 | 0.8778 | 0.0647 | 1812456 |
| cartoboost_graph_graphsage | ok | 0.4167 | 0.5191 | 0.8870 | 0.0658 | 1953538 |
| cartoboost_graph_hetero_graphsage | ok | 0.4222 | 0.5259 | 0.8840 | 0.0643 | 2029598 |
| cartoboost_graph_hinsage | ok | 0.4071 | 0.5113 | 0.8904 | 0.0623 | 1823078 |
| xgboost | ok | 0.4140 | 0.5154 | 0.8886 | 0.0691 | 1412633 |
| lightgbm | ok | 0.3994 | 0.4987 | 0.8957 | 0.1016 | 783781 |

#### group_holdout

Train rows: `1906`; test rows: `494`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2678 | 1.5773 | -0.0127 | 0.0000 | 269503120 |
| cartoboost | ok | 0.4393 | 0.5481 | 0.8777 | 0.0390 | 1696866 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | ok | 0.4638 | 0.5791 | 0.8635 | 0.0630 | 1219001 |
| cartoboost_graph_graphsage | ok | 0.4644 | 0.5737 | 0.8660 | 0.0722 | 1689611 |
| cartoboost_graph_hetero_graphsage | ok | 0.5137 | 0.6448 | 0.8308 | 0.0682 | 1797998 |
| cartoboost_graph_hinsage | ok | 0.5130 | 0.6432 | 0.8316 | 0.0647 | 1227457 |
| xgboost | ok | 0.4401 | 0.5460 | 0.8786 | 0.0708 | 1226567 |
| lightgbm | ok | 0.4257 | 0.5343 | 0.8838 | 0.1048 | 712929 |

## Plots

![MAE by workload and split](mae_by_model.png)

![Training time by workload and split](train_time_by_model.png)

![Prediction throughput by workload and split](prediction_throughput_by_model.png)

## Interpretation Notes

- The normal workload checks dense numeric behavior without ID or graph augmentation.
- The neural workload includes repeated IDs and a group holdout split, so `cartoboost_neural` should be read as an embedding augmentation check rather than a replacement for external neural networks.
- The graph workload benchmarks node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE feature augmentation from train topology before fitting CartoBoost on augmented source-target rows.
- XGBoost and LightGBM rows are skipped when their optional benchmark dependencies are not installed.
