# Model Benchmark Suite

This generated report compares CartoBoost against optional XGBoost and LightGBM baselines on deterministic synthetic workloads.

## Command

`uv run --group dev --group bench python scripts/run_model_benchmark_suite.py`

## Configuration

- Seed: `42`
- Rows per workload: `1200`
- Train fraction: `0.8`
- Models requested: `mean, cartoboost, cartoboost_neural, cartoboost_graph, xgboost, lightgbm`

## Results

### Normal dense

IID dense numeric regression with nonlinear feature interactions.

#### random

Train rows: `960`; test rows: `240`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.7165 | 2.0972 | -0.0000 | 0.0000 | 64707478 |
| cartoboost | ok | 0.4468 | 0.5861 | 0.9219 | 0.2082 | 502529 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4427 | 0.5913 | 0.9205 | 0.1149 | 230059 |
| lightgbm | ok | 0.4511 | 0.5875 | 0.9215 | 0.0937 | 154735 |

### Neural ID

Dense regression with repeated cell IDs whose residual signal is learnable by embedding features.

#### random

Train rows: `960`; test rows: `240`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.4357 | 1.7831 | -0.0016 | 0.0000 | 81135788 |
| cartoboost | ok | 0.4680 | 0.5895 | 0.8905 | 0.2894 | 329595 |
| cartoboost_neural | ok | 0.3977 | 0.5108 | 0.9178 | 0.2983 | 325939 |
| cartoboost_graph | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4625 | 0.5906 | 0.8901 | 0.0783 | 648137 |
| lightgbm | ok | 0.4982 | 0.6151 | 0.8808 | 0.0850 | 392691 |

#### group_holdout

Train rows: `962`; test rows: `238`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.4153 | 1.7390 | -0.0110 | 0.0000 | 102014598 |
| cartoboost | ok | 0.5211 | 0.6455 | 0.8607 | 0.1083 | 750493 |
| cartoboost_neural | ok | 0.7384 | 0.8807 | 0.7407 | 0.3063 | 344055 |
| cartoboost_graph | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.5229 | 0.6591 | 0.8548 | 0.0883 | 616713 |
| lightgbm | ok | 0.5048 | 0.6300 | 0.8673 | 0.1010 | 333236 |

### Graph source-target

Directed source-target regression where graph topology and node features carry predictive signal.

#### random

Train rows: `960`; test rows: `240`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2035 | 1.5113 | -0.0252 | 0.0000 | 101052481 |
| cartoboost | ok | 0.3893 | 0.5001 | 0.8878 | 0.0931 | 744091 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph | ok | 0.3879 | 0.4977 | 0.8888 | 0.2258 | 806834 |
| xgboost | ok | 0.3888 | 0.5067 | 0.8848 | 0.0707 | 672897 |
| lightgbm | ok | 0.3895 | 0.4971 | 0.8891 | 0.0810 | 528004 |

#### group_holdout

Train rows: `964`; test rows: `236`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2119 | 1.5581 | -0.0316 | 0.0000 | 96013004 |
| cartoboost | ok | 0.3953 | 0.5017 | 0.8930 | 0.0841 | 959837 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph | ok | 0.3907 | 0.4950 | 0.8959 | 0.2250 | 856886 |
| xgboost | ok | 0.3946 | 0.5092 | 0.8898 | 0.0735 | 683810 |
| lightgbm | ok | 0.3900 | 0.4910 | 0.8976 | 0.0820 | 472276 |

## Plots

![MAE by workload and split](mae_by_model.png)

![Training time by workload and split](train_time_by_model.png)

![Prediction throughput by workload and split](prediction_throughput_by_model.png)

## Interpretation Notes

- The normal workload checks dense numeric behavior without ID or graph augmentation.
- The neural workload includes repeated IDs and a group holdout split, so `cartoboost_neural` should be read as an embedding augmentation check rather than a replacement for external neural networks.
- The graph workload fits GraphSAGE features from train topology and node features, then trains CartoBoost on augmented source-target rows.
- XGBoost and LightGBM rows are skipped when their optional benchmark dependencies are not installed.
