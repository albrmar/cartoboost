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
| cartoboost | ok | 0.360522 | 0.280265 | 0.733712 | 0.677868 | 0.001542 | 1297297.86 | n_estimators=20 |
| cartoboost_reference | ok | 0.360522 | 0.280265 | 0.733712 | 1.355425 | 0.001668 | 1199190.31 | n_estimators=20 |
| cartoboost_neural | ok | 0.375043 | 0.289923 | 0.711829 | 0.885723 | 0.046916 | 42629.42 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.360661 | 0.280017 | 0.733508 | 1.670992 | 0.002060 | 970932.70 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.360601 | 0.280430 | 0.733596 | 1.707881 | 0.001659 | 1205454.68 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.360595 | 0.280426 | 0.733604 | 0.956169 | 0.001569 | 1274697.26 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.360601 | 0.280430 | 0.733596 | 3.918762 | 0.003923 | 509813.92 | n_estimators=20 |
| lightgbm | ok | 0.365159 | 0.283737 | 0.726819 | 0.070755 | 0.003936 | 508130.08 | n_estimators=20 |
| xgboost | ok | 0.364763 | 0.284134 | 0.727412 | 0.050652 | 0.001976 | 1012252.30 | n_estimators=20 |
| mean | ok | 0.698645 | 0.558907 | -0.000001 | 0.000115 | 0.000016 | 126318445.93 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.365853 | 0.283224 | 0.691871 | 17.438573 | 0.005996 | 434974.77 | n_estimators=20 |
| cartoboost_reference | ok | 0.365853 | 0.283224 | 0.691871 | 20.968136 | 0.461835 | 5647.04 | n_estimators=20 |
| cartoboost_neural | ok | 0.423961 | 0.332386 | 0.586218 | 0.766671 | 0.008577 | 304070.51 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.365868 | 0.283218 | 0.691844 | 23.794582 | 0.007365 | 354127.27 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.365868 | 0.283218 | 0.691844 | 3.676793 | 0.007339 | 355349.66 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.365868 | 0.283218 | 0.691844 | 4.531680 | 0.028775 | 90633.84 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.365868 | 0.283218 | 0.691844 | 2.671934 | 0.002421 | 1077259.50 | n_estimators=20 |
| lightgbm | ok | 0.371379 | 0.288710 | 0.682493 | 1.485792 | 0.000607 | 4300082.40 | n_estimators=20 |
| xgboost | ok | 0.371903 | 0.289009 | 0.681596 | 0.025087 | 0.000266 | 9798359.66 | n_estimators=20 |
| mean | ok | 0.667082 | 0.528604 | -0.024420 | 0.000010 | 0.000007 | 386370343.69 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.200199 | 0.153503 | 0.850829 | 0.702606 | 0.001822 | 1097820.77 | n_estimators=20 |
| cartoboost_reference | ok | 0.200199 | 0.153503 | 0.850829 | 0.683901 | 0.001650 | 1212121.21 | n_estimators=20 |
| cartoboost_neural | ok | 0.207919 | 0.159386 | 0.839103 | 0.919599 | 0.032055 | 62392.68 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.199955 | 0.153359 | 0.851193 | 1.775922 | 0.008489 | 235589.77 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.200028 | 0.153017 | 0.851083 | 2.155699 | 0.007279 | 274778.72 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.199794 | 0.153061 | 0.851433 | 4.030600 | 0.006090 | 328431.92 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.199625 | 0.152911 | 0.851683 | 5.549434 | 0.006073 | 329342.36 | n_estimators=20 |
| lightgbm | ok | 0.201223 | 0.154882 | 0.849300 | 6.009285 | 0.003178 | 629334.74 | n_estimators=20 |
| xgboost | ok | 0.201669 | 0.155100 | 0.848630 | 0.047266 | 0.000398 | 5019349.55 | n_estimators=20 |
| mean | ok | 0.518414 | 0.398975 | -0.000263 | 0.000058 | 0.000021 | 95616025.74 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.185314 | 0.144115 | 0.810903 | 43.455885 | 0.002168 | 1202813.33 | n_estimators=20 |
| cartoboost_reference | ok | 0.185314 | 0.144115 | 0.810903 | 43.265647 | 0.007936 | 328608.33 | n_estimators=20 |
| cartoboost_neural | ok | 0.193923 | 0.149893 | 0.792926 | 44.180186 | 0.185964 | 14024.19 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.186392 | 0.144458 | 0.808697 | 3.980107 | 0.010953 | 238100.13 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.188300 | 0.144515 | 0.804759 | 7.066973 | 0.008085 | 322559.38 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.183799 | 0.142815 | 0.813983 | 8.917916 | 0.002044 | 1275903.96 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.186023 | 0.144213 | 0.809453 | 6.439278 | 0.002044 | 1275747.30 | n_estimators=20 |
| lightgbm | ok | 0.185852 | 0.145070 | 0.809804 | 0.106150 | 0.000507 | 5146105.64 | n_estimators=20 |
| xgboost | ok | 0.185979 | 0.144976 | 0.809542 | 0.026876 | 0.000238 | 10963741.48 | n_estimators=20 |
| mean | ok | 0.445019 | 0.356651 | -0.090500 | 0.000016 | 0.000008 | 347733046.94 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.727320 | 0.592363 | 0.873058 | 54.941074 | 0.009906 | 497657.28 | n_estimators=20 |
| cartoboost_reference | ok | 0.727320 | 0.592363 | 0.873058 | 28.115108 | 0.010813 | 455937.94 | n_estimators=20 |
| cartoboost_neural | ok | 0.726528 | 0.590406 | 0.873334 | 32.810084 | 0.017003 | 289951.68 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.719462 | 0.585209 | 0.875786 | 7.748321 | 0.011581 | 425712.59 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.719462 | 0.585209 | 0.875786 | 6.673883 | 0.009983 | 493831.32 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.719462 | 0.585209 | 0.875786 | 9.852296 | 0.014699 | 335394.11 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.719462 | 0.585209 | 0.875786 | 9.332324 | 0.010361 | 475811.32 | n_estimators=20 |
| lightgbm | ok | 0.740537 | 0.602401 | 0.868402 | 6.381165 | 0.001926 | 2559210.95 | n_estimators=20 |
| xgboost | ok | 0.740934 | 0.602332 | 0.868261 | 0.231149 | 0.000463 | 10651790.20 | n_estimators=20 |
| mean | ok | 2.041944 | 1.752775 | -0.000560 | 0.000070 | 0.000022 | 219111143.22 |  |

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
| mean | ok | 2.088607 | 1.807484 | -0.002958 | 0.000017 | 0.000007 | 721263558.20 |  |

