# Model Benchmark Suite

This generated report compares CartoBoost against optional XGBoost and LightGBM baselines on deterministic synthetic workloads.

## Command

`uv run --group dev --group bench python scripts/run_model_benchmark_suite.py`

## Configuration

- Seed: `42`
- Rows per workload: `2400`
- Train fraction: `0.8`
- Models requested: `mean, cartoboost, cartoboost_neural, cartoboost_graph, xgboost, lightgbm`

## Results

### Normal dense

IID dense numeric regression with nonlinear feature interactions.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.7284 | 2.1533 | -0.0000 | 0.0000 | 114068267 |
| cartoboost | ok | 0.3821 | 0.5082 | 0.9443 | 0.1497 | 846996 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.3810 | 0.5173 | 0.9423 | 0.0812 | 1084645 |
| lightgbm | ok | 0.3783 | 0.5080 | 0.9443 | 0.1158 | 527738 |

### Neural ID

Dense regression with repeated cell IDs whose residual signal is learnable by embedding features.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.3848 | 1.7144 | -0.0005 | 0.0000 | 213333089 |
| cartoboost | ok | 0.4420 | 0.5523 | 0.8962 | 0.1376 | 1049274 |
| cartoboost_neural | ok | 0.3386 | 0.4360 | 0.9353 | 0.3928 | 407125 |
| cartoboost_graph | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4492 | 0.5636 | 0.8919 | 0.0823 | 1107905 |
| lightgbm | ok | 0.4291 | 0.5368 | 0.9019 | 0.1145 | 703426 |

#### group_holdout

Train rows: `1945`; test rows: `455`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.3599 | 1.6951 | -0.0034 | 0.0000 | 222929860 |
| cartoboost | ok | 0.4529 | 0.5740 | 0.8849 | 0.1627 | 533881 |
| cartoboost_neural | ok | 0.6737 | 0.7955 | 0.7790 | 0.6037 | 256615 |
| cartoboost_graph | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4686 | 0.5985 | 0.8749 | 0.1201 | 777944 |
| lightgbm | ok | 0.4402 | 0.5677 | 0.8875 | 0.1663 | 482439 |

### Graph source-target

Directed source-target regression where graph topology and node features carry predictive signal.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2321 | 1.5445 | -0.0000 | 0.0000 | 155692248 |
| cartoboost | ok | 0.4123 | 0.5118 | 0.8902 | 0.1799 | 599594 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph | ok | 0.4167 | 0.5191 | 0.8870 | 0.5091 | 604978 |
| xgboost | ok | 0.4140 | 0.5154 | 0.8886 | 0.1120 | 905804 |
| lightgbm | ok | 0.3994 | 0.4987 | 0.8957 | 0.1635 | 457979 |

#### group_holdout

Train rows: `1906`; test rows: `494`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2678 | 1.5773 | -0.0127 | 0.0000 | 148170195 |
| cartoboost | ok | 0.4393 | 0.5481 | 0.8777 | 0.1704 | 872021 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph | ok | 0.4644 | 0.5737 | 0.8660 | 0.3930 | 527426 |
| xgboost | ok | 0.4401 | 0.5460 | 0.8786 | 0.0897 | 1041645 |
| lightgbm | ok | 0.4257 | 0.5343 | 0.8838 | 0.1409 | 399892 |

## Plots

![MAE by workload and split](mae_by_model.png)

![Training time by workload and split](train_time_by_model.png)

![Prediction throughput by workload and split](prediction_throughput_by_model.png)

## Interpretation Notes

- The normal workload checks dense numeric behavior without ID or graph augmentation.
- The neural workload includes repeated IDs and a group holdout split, so `cartoboost_neural` should be read as an embedding augmentation check rather than a replacement for external neural networks.
- The graph workload can benchmark node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE feature augmentation from train topology before fitting CartoBoost on source-target rows.
- XGBoost and LightGBM rows are skipped when their optional benchmark dependencies are not installed.
