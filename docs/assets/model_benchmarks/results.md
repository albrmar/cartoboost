# Model Benchmark Suite

This generated report compares CartoBoost against optional XGBoost and LightGBM baselines on deterministic synthetic workloads.

## Command

`uv run --group dev --group bench python scripts/run_model_benchmark_suite.py`

## Configuration

- Seed: `42`
- Rows per workload: `2400`
- Train fraction: `0.8`
- Models requested: `mean, cartoboost, cartoboost_neural, cartoboost_graph_node2vec, cartoboost_graph_graphsage, cartoboost_graph_hetero_graphsage, cartoboost_graph_hinsage, neural_embedding_regressor, node2vec_regressor, graphsage_regressor, hetero_graphsage_regressor, hinsage_regressor, node2vec_link_predictor, graphsage_link_predictor, hetero_graphsage_link_predictor, hinsage_link_predictor, xgboost, lightgbm`

## Results

## LightGBM Comparison

For each benchmark split, this table compares LightGBM with the best CartoBoost-family model that finished successfully under the same data split and global benchmark settings.

| Workload | Split | Best CartoBoost-family model | CartoBoost RMSE | LightGBM RMSE | RMSE delta | R2 delta | Winner |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| normal | random | cartoboost | 0.4625 | 0.5080 | -0.0455 | 0.0095 | cartoboost |
| neural | random | cartoboost_neural | 0.4584 | 0.5368 | -0.0783 | 0.0265 | cartoboost |
| neural | group_holdout | cartoboost | 0.5387 | 0.5677 | -0.0290 | 0.0112 | cartoboost |
| graph | random | graphsage_regressor | 0.4495 | 0.4987 | -0.0493 | 0.0196 | cartoboost |
| graph | group_holdout | cartoboost | 0.5210 | 0.5343 | -0.0132 | 0.0057 | cartoboost |

### Why CartoBoost Wins Here

- The normal dense workload is a baseline sanity check: CartoBoost is competitive without relying on graph or neural inputs.
- The neural workload shows the difference between repeated-ID and cold-ID claims. `cartoboost_neural` wins the random split because held-out rows reuse train-observed IDs; the group holdout falls back to the base CartoBoost structure instead of pretending unseen IDs can be recovered from an embedding table.
- The graph workload separates two surfaces. Augmented CartoBoost uses graph features as extra columns for the booster, while standalone GraphSAGE-style regressors and link predictors can score graph tasks without a boosted wrapper. The link-predictor rows report AUC/AP because they are ranking candidate source-target edges, not predicting the regression target.
- LightGBM sees the benchmark as ordinary dense tabular columns. CartoBoost-family rows can add ID residual structure, source-target topology, and spatially shaped splitters when the workload exposes those contracts.

### Normal dense

IID dense numeric regression with nonlinear feature interactions.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.7284 | 2.1533 | -0.0000 | 0.0000 | 78150259 |
| cartoboost | ok | 0.3502 | 0.4625 | 0.9539 | 0.2745 | 769222 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| neural_embedding_regressor | skipped: workload has no embedding ids |  |  |  |  |  |
| node2vec_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| hetero_graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| hinsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| node2vec_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| graphsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| hetero_graphsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| hinsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.3810 | 0.5173 | 0.9423 | 0.0922 | 119524 |
| lightgbm | ok | 0.3783 | 0.5080 | 0.9443 | 0.0594 | 100006 |

### Neural ID

Dense regression with repeated cell IDs whose residual signal is learnable by embedding features.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.3848 | 1.7144 | -0.0005 | 0.0000 | 64829695 |
| cartoboost | ok | 0.4084 | 0.5072 | 0.9124 | 0.2981 | 582269 |
| cartoboost_neural | ok | 0.3550 | 0.4584 | 0.9285 | 0.4873 | 236224 |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| neural_embedding_regressor | ok | 0.4148 | 0.5288 | 0.9048 | 0.1095 | 370947 |
| node2vec_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| hetero_graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| hinsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| node2vec_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| graphsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| hetero_graphsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| hinsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4492 | 0.5636 | 0.8919 | 0.0751 | 292041 |
| lightgbm | ok | 0.4291 | 0.5368 | 0.9019 | 0.0419 | 415984 |

#### group_holdout

Train rows: `1945`; test rows: `455`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.3599 | 1.6951 | -0.0034 | 0.0000 | 88693684 |
| cartoboost | ok | 0.4178 | 0.5387 | 0.8987 | 0.3105 | 489437 |
| cartoboost_neural | ok | 0.4178 | 0.5387 | 0.8987 | 0.2790 | 825241 |
| cartoboost_graph_node2vec | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hetero_graphsage | skipped: workload has no graph topology |  |  |  |  |  |
| cartoboost_graph_hinsage | skipped: workload has no graph topology |  |  |  |  |  |
| neural_embedding_regressor | ok | 0.6894 | 0.8185 | 0.7660 | 0.1039 | 317692 |
| node2vec_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| hetero_graphsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| hinsage_regressor | skipped: workload has no graph topology |  |  |  |  |  |
| node2vec_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| graphsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| hetero_graphsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| hinsage_link_predictor | skipped: workload has no graph topology |  |  |  |  |  |
| xgboost | ok | 0.4686 | 0.5985 | 0.8749 | 0.0400 | 567966 |
| lightgbm | ok | 0.4402 | 0.5677 | 0.8875 | 0.0271 | 444430 |

### Graph source-target

Directed source-target regression where graph topology and node features carry predictive signal.

#### random

Train rows: `1920`; test rows: `480`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2321 | 1.5445 | -0.0000 | 0.0000 | 72583968 |
| cartoboost | ok | 0.3772 | 0.4701 | 0.9074 | 0.2767 | 729035 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | ok | 0.3929 | 0.4929 | 0.8981 | 0.3226 | 841350 |
| cartoboost_graph_graphsage | ok | 0.3782 | 0.4764 | 0.9049 | 0.3505 | 804486 |
| cartoboost_graph_hetero_graphsage | ok | 0.3885 | 0.4859 | 0.9010 | 0.3256 | 817641 |
| cartoboost_graph_hinsage | ok | 0.3901 | 0.4880 | 0.9001 | 0.3369 | 562741 |
| neural_embedding_regressor | skipped: workload has no embedding ids |  |  |  |  |  |
| node2vec_regressor | ok | 0.4034 | 0.5214 | 0.8860 | 0.4421 | 376525 |
| graphsage_regressor | ok | 0.3518 | 0.4495 | 0.9153 | 0.1464 | 370932 |
| hetero_graphsage_regressor | ok | 0.3686 | 0.4674 | 0.9084 | 0.1541 | 305238 |
| hinsage_regressor | ok | 0.3686 | 0.4674 | 0.9084 | 0.1610 | 351358 |
| node2vec_link_predictor | ok link: AUC 0.9902, AP 0.9917 |  |  |  | 0.2637 | 696687 |
| graphsage_link_predictor | ok link: AUC 0.9906, AP 0.9917 |  |  |  | 0.0101 | 658605 |
| hetero_graphsage_link_predictor | ok link: AUC 0.9374, AP 0.9472 |  |  |  | 0.0106 | 636349 |
| hinsage_link_predictor | ok link: AUC 0.9374, AP 0.9472 |  |  |  | 0.0105 | 646871 |
| xgboost | ok | 0.4140 | 0.5154 | 0.8886 | 0.1039 | 597275 |
| lightgbm | ok | 0.3994 | 0.4987 | 0.8957 | 0.0658 | 399816 |

#### group_holdout

Train rows: `1906`; test rows: `494`.

| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mean | ok | 1.2678 | 1.5773 | -0.0127 | 0.0000 | 109074329 |
| cartoboost | ok | 0.4230 | 0.5210 | 0.8895 | 0.2809 | 684106 |
| cartoboost_neural | skipped: workload has no embedding ids |  |  |  |  |  |
| cartoboost_graph_node2vec | ok | 0.4859 | 0.5977 | 0.8546 | 0.3658 | 649003 |
| cartoboost_graph_graphsage | ok | 0.4372 | 0.5459 | 0.8787 | 0.3627 | 747036 |
| cartoboost_graph_hetero_graphsage | ok | 0.4886 | 0.6051 | 0.8510 | 0.3512 | 656846 |
| cartoboost_graph_hinsage | ok | 0.4893 | 0.6101 | 0.8485 | 0.3470 | 764077 |
| neural_embedding_regressor | skipped: workload has no embedding ids |  |  |  |  |  |
| node2vec_regressor | ok | 0.4593 | 0.5783 | 0.8639 | 0.2000 | 319513 |
| graphsage_regressor | ok | 0.4570 | 0.5777 | 0.8641 | 0.1636 | 324630 |
| hetero_graphsage_regressor | ok | 0.5504 | 0.6876 | 0.8076 | 0.1605 | 329516 |
| hinsage_regressor | ok | 0.5504 | 0.6876 | 0.8076 | 0.1665 | 324009 |
| node2vec_link_predictor | ok link: AUC 0.6206, AP 0.5992 |  |  |  | 0.0506 | 715846 |
| graphsage_link_predictor | ok link: AUC 0.9973, AP 0.9979 |  |  |  | 0.0090 | 695994 |
| hetero_graphsage_link_predictor | ok link: AUC 0.8570, AP 0.8938 |  |  |  | 0.0094 | 663345 |
| hinsage_link_predictor | ok link: AUC 0.8570, AP 0.8938 |  |  |  | 0.0094 | 659525 |
| xgboost | ok | 0.4401 | 0.5460 | 0.8786 | 0.0357 | 608835 |
| lightgbm | ok | 0.4257 | 0.5343 | 0.8838 | 0.0416 | 522924 |

## Plots

![MAE by workload and split](mae_by_model.png)

![Training time by workload and split](train_time_by_model.png)

![Prediction throughput by workload and split](prediction_throughput_by_model.png)

## Interpretation Notes

- The normal workload checks dense numeric behavior without ID or graph augmentation.
- The neural workload includes repeated IDs and a group holdout split, so `cartoboost_neural` should be read as an embedding augmentation check rather than a replacement for external neural networks.
- The graph workload benchmarks node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE feature augmentation from train topology before fitting CartoBoost on augmented source-target rows.
- XGBoost and LightGBM rows are skipped when their optional benchmark dependencies are not installed.
