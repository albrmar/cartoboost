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
| mean | ok | 1.7284 | 2.1533 | -0.0000 | 0.0001 | 25043047 |
| cartoboost | ok | 0.3821 | 0.5082 | 0.9443 | 0.3903 | 73796 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.3810 | 0.5173 | 0.9423 | 0.1402 | 308533 |
| lightgbm | ok | 0.3783 | 0.5080 | 0.9443 | 0.1548 | 127484 |

### Neural ID

Dense regression with repeated cell IDs whose residual signal is learnable by embedding features.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.3848 | 1.7144 | -0.0005 | 0.0000 | 41587257 |
| cartoboost | ok | 0.4420 | 0.5523 | 0.8962 | 0.4143 | 247099 |
| cartoboost_neural | ok | 0.3386 | 0.4360 | 0.9353 | 1.0953 | 111531 |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4492 | 0.5636 | 0.8919 | 0.1649 | 363372 |
| lightgbm | ok | 0.4291 | 0.5368 | 0.9019 | 0.1366 | 123120 |

#### group_holdout

Train rows: `1945`; test rows: `455`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.3599 | 1.6951 | -0.0034 | 0.0000 | 37656231 |
| cartoboost | ok | 0.4529 | 0.5740 | 0.8849 | 0.4214 | 229605 |
| cartoboost_neural | ok | 0.6737 | 0.7955 | 0.7790 | 0.9786 | 114956 |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4686 | 0.5985 | 0.8749 | 0.1500 | 419710 |
| lightgbm | ok | 0.4402 | 0.5677 | 0.8875 | 0.1364 | 290650 |

### Graph source-target

Directed source-target regression where graph topology and node features carry predictive signal.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2321 | 1.5445 | -0.0000 | 0.0000 | 77319590 |
| cartoboost | ok | 0.4123 | 0.5118 | 0.8902 | 0.3482 | 178685 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | ok | 0.4316 | 0.5400 | 0.8778 | 0.5023 | 358098 |
| cartoboost_graph_graphsage | ok | 0.4167 | 0.5191 | 0.8870 | 0.4827 | 314995 |
| cartoboost_graph_hetero_graphsage | ok | 0.4222 | 0.5259 | 0.8840 | 0.4799 | 296663 |
| cartoboost_graph_hinsage | ok | 0.4071 | 0.5113 | 0.8904 | 0.5212 | 377334 |
| xgboost | ok | 0.4140 | 0.5154 | 0.8886 | 0.0786 | 597418 |
| lightgbm | ok | 0.3994 | 0.4987 | 0.8957 | 0.0932 | 244259 |

#### group_holdout

Train rows: `1906`; test rows: `494`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2678 | 1.5773 | -0.0127 | 0.0000 | 110787044 |
| cartoboost | ok | 0.4393 | 0.5481 | 0.8777 | 0.1792 | 396853 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | ok | 0.4638 | 0.5791 | 0.8635 | 0.5289 | 96238 |
| cartoboost_graph_graphsage | ok | 0.4644 | 0.5737 | 0.8660 | 0.5065 | 328849 |
| cartoboost_graph_hetero_graphsage | ok | 0.5137 | 0.6448 | 0.8308 | 0.4896 | 338385 |
| cartoboost_graph_hinsage | ok | 0.5130 | 0.6432 | 0.8316 | 0.4872 | 396893 |
| xgboost | ok | 0.4401 | 0.5460 | 0.8786 | 0.0816 | 661128 |
| lightgbm | ok | 0.4257 | 0.5343 | 0.8838 | 0.1099 | 185578 |

## Plots

![MAE by workload and split](mae_by_model.png)

![Training time by workload and split](train_time_by_model.png)

![Prediction throughput by workload and split](prediction_throughput_by_model.png)

## Interpretation Notes

- The normal workload checks dense numeric behavior without ID or graph augmentation.
- The neural workload includes repeated IDs and a group holdout split, so `cartoboost_neural` should be read as an embedding augmentation check rather than a replacement for external neural networks.
- The graph workload benchmarks node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE feature augmentation from train topology before fitting CartoBoost on augmented source-target rows.
- XGBoost and LightGBM rows are skipped when their optional benchmark dependencies are not installed.
