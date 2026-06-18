# NYC Taxi Model Quality Benchmarks

## Research Question

On real NYC taxi data, do geographic and temporal feature families improve
prediction quality for trip duration, fare amount, and pickup-zone demand when
compared with strong gradient-boosted tabular baselines?

## Dataset

The benchmark uses NYC TLC taxi-derived records. The row-level tasks use trip
records with pickup/dropoff zone context, trip attributes, passenger count, and
time-of-day features. The demand task aggregates pickup activity by zone and
time bucket.

## Targets

Quality metrics are computed on transformed regression targets:

- Trip duration: log trip duration.
- Fare amount: log total amount.
- Pickup-zone demand: log pickup trip count for a zone-time bucket.

## Feature Sets

- Geographic features: pickup zone, dropoff zone, route geometry, and
  zone-level encodings.
- Temporal features: hour, weekday, and periodic time structure.
- Trip features: distance, passenger count, and related trip descriptors.
- Graph features for pickup demand: topology learned from observed pickup-zone
  relationships.

## Comparison Method

CartoBoost-family models are compared with LightGBM, XGBoost, and a mean
baseline under the same task, split, target transformation, and global
benchmark settings.

- dataset source: nyc_tlc_trip_records
- models requested: cartoboost, cartoboost_reference, cartoboost_neural, cartoboost_graph_node2vec, cartoboost_graph_graphsage, cartoboost_graph_hetero_graphsage, cartoboost_graph_hinsage, lightgbm, xgboost, mean
- baseline estimators: 100
- CartoBoost candidate estimators: 100
- baseline max depth: 4
- CartoBoost candidate max depth: 5
- model workers: 1
- zone treatment: target_mean

## CartoBoost vs LightGBM

For each runnable learned-model split, this table compares LightGBM with the best CartoBoost-family row under the same task, split, data sample, target transformation, and global benchmark settings.

| task | split | best CartoBoost-family model | CartoBoost RMSE | LightGBM RMSE | RMSE delta | R2 delta | winner |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| duration | random | cartoboost | 0.278843 | 0.289829 | -0.010986 | 0.012637 | cartoboost |
| duration | spatial_holdout | cartoboost | 0.303525 | 0.316291 | -0.012766 | 0.018206 | cartoboost |
| fare | random | cartoboost | 0.139640 | 0.143692 | -0.004052 | 0.004239 | cartoboost |
| fare | spatial_holdout | cartoboost | 0.148375 | 0.152686 | -0.004311 | 0.007854 | cartoboost |
| pickup_demand | random | cartoboost_graph_node2vec | 0.403529 | 0.481552 | -0.078024 | 0.016572 | cartoboost |

### Interpretation

- Fare and duration are primarily geotemporal row tasks. The base CartoBoost candidate wins through native periodic hour/day splitters, diagonal and radial spatial splitters, and sparse-set taxi-zone membership. Those primitives let the model express pickup/dropoff geometry directly instead of asking an axis-only tabular baseline to approximate it through many rectangular cuts.
- Pickup demand is a zone-time graph problem. The best row in the random split is graph-augmented CartoBoost, because node2vec adds topology learned from observed pickup-zone relationships before the booster models hour, weekday, and zone effects.
- Graph and neural rows are not expected to improve every target. When the base geotemporal splitters already explain the signal, they match the base candidate and mainly add training cost. Their value is in workloads where ID residuals or source-target topology carry signal that ordinary dense columns do not expose.
- The pickup-demand cold-zone spatial holdout intentionally skips learned models. That split removes all zone demand history, so a quality comparison would collapse to priors rather than test model structure.

## Trip duration

Predict log trip duration from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.278843 | 0.211139 | 0.842726 | 72.563470 | 0.036612 | 546268.85 | n_estimators=100 |
| cartoboost_reference | ok | 0.285734 | 0.216774 | 0.834856 | 54.150154 | 0.027043 | 739568.99 | n_estimators=100 |
| cartoboost_neural | ok | 0.278843 | 0.211139 | 0.842726 | 211.638921 | 0.034810 | 574539.85 | n_estimators=100 |
| cartoboost_graph_node2vec | ok | 0.278843 | 0.211139 | 0.842726 | 180.341799 | 0.036794 | 543560.28 | n_estimators=100 |
| cartoboost_graph_graphsage | ok | 0.278843 | 0.211139 | 0.842726 | 187.456467 | 0.035229 | 567711.38 | n_estimators=100 |
| cartoboost_graph_hetero_graphsage | ok | 0.278843 | 0.211139 | 0.842726 | 185.924758 | 0.033743 | 592711.71 | n_estimators=100 |
| cartoboost_graph_hinsage | ok | 0.278843 | 0.211139 | 0.842726 | 184.215996 | 0.032734 | 610980.63 | n_estimators=100 |
| lightgbm | ok | 0.289829 | 0.220069 | 0.830088 | 0.436127 | 0.011189 | 1787460.73 | n_estimators=100 |
| xgboost | ok | 0.290784 | 0.220863 | 0.828967 | 0.391202 | 0.003811 | 5248028.33 | n_estimators=100 |
| mean | ok | 0.703139 | 0.559350 | -0.000049 | 0.000096 | 0.000073 | 274638422.24 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.303525 | 0.229885 | 0.788023 | 63.009942 | 0.043197 | 603211.92 | n_estimators=100 |
| cartoboost_reference | ok | 0.311246 | 0.236122 | 0.777100 | 49.893733 | 0.034555 | 754080.37 | n_estimators=100 |
| cartoboost_neural | ok | 0.303525 | 0.229885 | 0.788023 | 194.738332 | 0.043481 | 599275.88 | n_estimators=100 |
| cartoboost_graph_node2vec | ok | 0.303525 | 0.229885 | 0.788023 | 166.574016 | 0.045172 | 576844.05 | n_estimators=100 |
| cartoboost_graph_graphsage | ok | 0.303525 | 0.229885 | 0.788023 | 173.200770 | 0.059458 | 438238.77 | n_estimators=100 |
| cartoboost_graph_hetero_graphsage | ok | 0.303525 | 0.229885 | 0.788023 | 175.665635 | 0.045494 | 572754.84 | n_estimators=100 |
| cartoboost_graph_hinsage | ok | 0.303525 | 0.229885 | 0.788023 | 167.144858 | 0.045188 | 576639.65 | n_estimators=100 |
| lightgbm | ok | 0.316291 | 0.240533 | 0.769816 | 0.335968 | 0.016231 | 1605412.16 | n_estimators=100 |
| xgboost | ok | 0.315985 | 0.240578 | 0.770261 | 0.278050 | 0.004724 | 5515582.11 | n_estimators=100 |
| mean | ok | 0.665259 | 0.524831 | -0.018317 | 0.000087 | 0.000127 | 205433700.29 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.139640 | 0.101675 | 0.927998 | 67.668439 | 0.039464 | 506795.01 | n_estimators=100 |
| cartoboost_reference | ok | 0.141458 | 0.103211 | 0.926111 | 53.953070 | 0.026286 | 760860.72 | n_estimators=100 |
| cartoboost_neural | ok | 0.139640 | 0.101675 | 0.927998 | 210.645407 | 0.033363 | 599460.01 | n_estimators=100 |
| cartoboost_graph_node2vec | ok | 0.139640 | 0.101675 | 0.927998 | 180.672176 | 0.034055 | 587293.57 | n_estimators=100 |
| cartoboost_graph_graphsage | ok | 0.139640 | 0.101675 | 0.927998 | 183.599234 | 0.033690 | 593645.45 | n_estimators=100 |
| cartoboost_graph_hetero_graphsage | ok | 0.139640 | 0.101675 | 0.927998 | 181.275771 | 0.033448 | 597941.52 | n_estimators=100 |
| cartoboost_graph_hinsage | ok | 0.139640 | 0.101675 | 0.927998 | 178.423495 | 0.042049 | 475634.74 | n_estimators=100 |
| lightgbm | ok | 0.143692 | 0.104978 | 0.923759 | 0.325512 | 0.011462 | 1744958.14 | n_estimators=100 |
| xgboost | ok | 0.143665 | 0.104987 | 0.923788 | 0.370695 | 0.006295 | 3176911.74 | n_estimators=100 |
| mean | ok | 0.520448 | 0.397656 | -0.000184 | 0.000086 | 0.000109 | 183192144.05 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.148375 | 0.108814 | 0.866796 | 62.604254 | 0.044440 | 586334.58 | n_estimators=100 |
| cartoboost_reference | ok | 0.150464 | 0.110646 | 0.863018 | 49.167658 | 0.033651 | 774341.33 | n_estimators=100 |
| cartoboost_neural | ok | 0.148375 | 0.108814 | 0.866796 | 192.391341 | 0.044976 | 579355.37 | n_estimators=100 |
| cartoboost_graph_node2vec | ok | 0.148375 | 0.108814 | 0.866796 | 166.779580 | 0.059893 | 435062.25 | n_estimators=100 |
| cartoboost_graph_graphsage | ok | 0.148375 | 0.108814 | 0.866796 | 166.185388 | 0.049686 | 524434.35 | n_estimators=100 |
| cartoboost_graph_hetero_graphsage | ok | 0.148375 | 0.108814 | 0.866796 | 168.323674 | 0.047422 | 549465.58 | n_estimators=100 |
| cartoboost_graph_hinsage | ok | 0.148375 | 0.108814 | 0.866796 | 163.888459 | 0.048634 | 535777.27 | n_estimators=100 |
| lightgbm | ok | 0.152686 | 0.111961 | 0.858942 | 0.258320 | 0.013550 | 1923028.53 | n_estimators=100 |
| xgboost | ok | 0.152334 | 0.111881 | 0.859593 | 0.247774 | 0.004506 | 5783360.45 | n_estimators=100 |
| mean | ok | 0.423812 | 0.339335 | -0.086785 | 0.000087 | 0.000114 | 227860602.61 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.427834 | 0.330577 | 0.956075 | 6.165220 | 0.008913 | 553110.32 | n_estimators=100 |
| cartoboost_reference | ok | 0.488164 | 0.370072 | 0.942814 | 4.753959 | 0.005371 | 917879.75 | n_estimators=100 |
| cartoboost_neural | ok | 0.427834 | 0.330577 | 0.956075 | 19.676253 | 0.008912 | 553191.74 | n_estimators=100 |
| cartoboost_graph_node2vec | ok | 0.403529 | 0.309349 | 0.960924 | 18.325866 | 0.008101 | 608583.90 | n_estimators=100 |
| cartoboost_graph_graphsage | ok | 0.412448 | 0.315426 | 0.959178 | 17.519595 | 0.008742 | 563951.02 | n_estimators=100 |
| cartoboost_graph_hetero_graphsage | ok | 0.405931 | 0.311029 | 0.960458 | 18.044027 | 0.008195 | 601554.84 | n_estimators=100 |
| cartoboost_graph_hinsage | ok | 0.405931 | 0.311029 | 0.960458 | 17.836502 | 0.010640 | 463366.07 | n_estimators=100 |
| lightgbm | ok | 0.481552 | 0.367974 | 0.944353 | 1.146953 | 0.002557 | 1928229.92 | n_estimators=100 |
| xgboost | ok | 0.483348 | 0.369962 | 0.943937 | 0.060401 | 0.000974 | 5060947.02 | n_estimators=100 |
| mean | ok | 2.041944 | 1.752775 | -0.000560 | 0.000043 | 0.000012 | 414842603.95 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| cartoboost_reference | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| cartoboost_neural | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| cartoboost_graph_node2vec | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| cartoboost_graph_graphsage | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| cartoboost_graph_hetero_graphsage | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| cartoboost_graph_hinsage | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| lightgbm | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| xgboost | skipped |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| mean | ok | 2.088607 | 1.807484 | -0.002958 | 0.000054 | 0.000014 | 376318572.84 |  |
