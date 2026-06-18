# NYC Taxi Model Quality Benchmarks

These artifacts compare predictive quality and speed on NYC TLC taxi-derived tasks.
Quality metrics are computed on transformed regression targets.

- dataset source: nyc_tlc_trip_records
- models requested: cartoboost, cartoboost_reference, cartoboost_neural, cartoboost_graph_node2vec, cartoboost_graph_graphsage, cartoboost_graph_hetero_graphsage, cartoboost_graph_hinsage, lightgbm, xgboost, mean
- baseline estimators: 20
- CartoBoost candidate estimators: 20
- baseline max depth: 3
- CartoBoost candidate max depth: 3
- model workers: 4
- zone treatment: target_mean

## Trip duration

Predict log trip duration from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.360522 | 0.280265 | 0.733712 | 0.760908 | 0.001782 | 1122466.10 | n_estimators=20 |
| cartoboost_reference | ok | 0.360522 | 0.280265 | 0.733712 | 0.729666 | 0.001486 | 1345442.32 | n_estimators=20 |
| cartoboost_neural | ok | 0.375043 | 0.289923 | 0.711829 | 0.402476 | 0.747613 | 2675.18 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.360661 | 0.280017 | 0.733508 | 2.424576 | 0.001667 | 1200030.00 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.360601 | 0.280430 | 0.733596 | 1.731300 | 0.001667 | 1199970.24 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.360595 | 0.280426 | 0.733604 | 0.913983 | 0.001773 | 1127793.05 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.360601 | 0.280430 | 0.733596 | 1.703167 | 0.001540 | 1298314.33 | n_estimators=20 |
| lightgbm | ok | 0.365159 | 0.283737 | 0.726819 | 0.037752 | 0.000555 | 3601170.37 | n_estimators=20 |
| xgboost | ok | 0.364763 | 0.284134 | 0.727412 | 0.028892 | 0.000328 | 6100667.17 | n_estimators=20 |
| mean | ok | 0.698645 | 0.558907 | -0.000001 | 0.000006 | 0.000011 | 180440198.50 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.365853 | 0.283224 | 0.691871 | 19.059232 | 0.002797 | 932427.60 | n_estimators=20 |
| cartoboost_reference | ok | 0.365853 | 0.283224 | 0.691871 | 32.536208 | 0.020959 | 124430.45 | n_estimators=20 |
| cartoboost_neural | ok | 0.423961 | 0.332386 | 0.586218 | 1.042468 | 0.173934 | 14994.20 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.365868 | 0.283218 | 0.691844 | 3.002615 | 0.007671 | 339972.53 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.365868 | 0.283218 | 0.691844 | 3.695323 | 0.006952 | 375159.60 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.365868 | 0.283218 | 0.691844 | 5.248997 | 0.005724 | 455655.29 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.365868 | 0.283218 | 0.691844 | 2.068994 | 0.006064 | 430091.00 | n_estimators=20 |
| lightgbm | ok | 0.371379 | 0.288710 | 0.682493 | 1.602867 | 0.001119 | 2329956.93 | n_estimators=20 |
| xgboost | ok | 0.371903 | 0.289009 | 0.681596 | 1.598768 | 0.000482 | 5412191.93 | n_estimators=20 |
| mean | ok | 0.667082 | 0.528604 | -0.024420 | 0.000377 | 0.000026 | 101447009.35 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.200199 | 0.153503 | 0.850829 | 1.515160 | 0.004762 | 419998.92 | n_estimators=20 |
| cartoboost_reference | ok | 0.200199 | 0.153503 | 0.850829 | 1.410791 | 0.005760 | 347199.62 | n_estimators=20 |
| cartoboost_neural | ok | 0.207919 | 0.159386 | 0.839103 | 2.002780 | 0.023007 | 86929.91 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.199955 | 0.153359 | 0.851193 | 3.662401 | 0.006384 | 313275.01 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.200028 | 0.153017 | 0.851083 | 3.474290 | 0.004176 | 478941.54 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.199794 | 0.153061 | 0.851433 | 5.062900 | 0.004220 | 473975.77 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.199625 | 0.152911 | 0.851683 | 2.194089 | 0.004163 | 480403.50 | n_estimators=20 |
| lightgbm | ok | 0.201223 | 0.154882 | 0.849300 | 1.693160 | 0.000841 | 2377414.55 | n_estimators=20 |
| xgboost | ok | 0.201669 | 0.155100 | 0.848630 | 1.689080 | 0.000548 | 3647698.04 | n_estimators=20 |
| mean | ok | 0.518414 | 0.398975 | -0.000263 | 0.000129 | 0.000012 | 163265323.58 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.185314 | 0.144115 | 0.810903 | 58.301057 | 0.004561 | 571867.12 | n_estimators=20 |
| cartoboost_reference | ok | 0.185314 | 0.144115 | 0.810903 | 51.612174 | 0.011324 | 230298.00 | n_estimators=20 |
| cartoboost_neural | ok | 0.193923 | 0.149893 | 0.792926 | 1.265020 | 0.008174 | 319078.35 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.186392 | 0.144458 | 0.808697 | 61.036009 | 0.005443 | 479180.54 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.188300 | 0.144515 | 0.804759 | 3.348016 | 0.005026 | 518871.67 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.183799 | 0.142815 | 0.813983 | 3.283562 | 0.002167 | 1203553.81 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.186023 | 0.144213 | 0.809453 | 2.269661 | 0.002324 | 1122343.63 | n_estimators=20 |
| lightgbm | ok | 0.185852 | 0.145070 | 0.809804 | 2.323890 | 0.000500 | 5212528.45 | n_estimators=20 |
| xgboost | ok | 0.185979 | 0.144976 | 0.809542 | 0.861627 | 0.000264 | 9885003.41 | n_estimators=20 |
| mean | ok | 0.445019 | 0.356651 | -0.090500 | 0.000005 | 0.000005 | 508877786.20 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.727320 | 0.592363 | 0.873058 | 25.420644 | 0.004246 | 1161047.13 | n_estimators=20 |
| cartoboost_reference | ok | 0.727320 | 0.592363 | 0.873058 | 34.471663 | 0.017103 | 288252.83 | n_estimators=20 |
| cartoboost_neural | ok | 0.726528 | 0.590406 | 0.873334 | 8.453740 | 0.014447 | 341257.14 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.719462 | 0.585209 | 0.875786 | 10.050736 | 0.004097 | 1203319.50 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.719462 | 0.585209 | 0.875786 | 5.079840 | 0.021572 | 228536.11 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.719462 | 0.585209 | 0.875786 | 5.138256 | 0.018408 | 267816.52 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.719462 | 0.585209 | 0.875786 | 5.258818 | 0.012574 | 392077.58 | n_estimators=20 |
| lightgbm | ok | 0.740537 | 0.602401 | 0.868402 | 5.319374 | 0.002963 | 1663643.65 | n_estimators=20 |
| xgboost | ok | 0.740934 | 0.602332 | 0.868261 | 4.532286 | 0.001016 | 4851765.27 | n_estimators=20 |
| mean | ok | 2.041944 | 1.752775 | -0.000560 | 0.000150 | 0.000014 | 340964075.93 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 2.082132 | 1.843798 | 0.003252 | 22.158905 | 0.003919 | 1311401.62 | n_estimators=20 |
| cartoboost_reference | ok | 2.082132 | 1.843798 | 0.003252 | 16.943972 | 0.011481 | 447615.60 | n_estimators=20 |
| cartoboost_neural | ok | 2.216042 | 1.710216 | -0.129081 | 0.761300 | 0.367947 | 13966.68 | n_estimators=20 |
| cartoboost_graph_node2vec | skipped |  |  |  |  |  |  | graph embeddings are skipped for pickup_demand cold-zone spatial holdout; use contextual tabular baselines for this split |
| cartoboost_graph_graphsage | skipped |  |  |  |  |  |  | graph embeddings are skipped for pickup_demand cold-zone spatial holdout; use contextual tabular baselines for this split |
| cartoboost_graph_hetero_graphsage | skipped |  |  |  |  |  |  | graph embeddings are skipped for pickup_demand cold-zone spatial holdout; use contextual tabular baselines for this split |
| cartoboost_graph_hinsage | skipped |  |  |  |  |  |  | graph embeddings are skipped for pickup_demand cold-zone spatial holdout; use contextual tabular baselines for this split |
| lightgbm | ok | 2.083912 | 1.813077 | 0.001546 | 0.205090 | 0.001500 | 3425808.16 | n_estimators=20 |
| xgboost | ok | 2.083900 | 1.813071 | 0.001558 | 0.029237 | 0.000662 | 7757953.01 | n_estimators=20 |
| mean | ok | 2.088607 | 1.807484 | -0.002958 | 0.000031 | 0.000012 | 435841083.97 |  |

