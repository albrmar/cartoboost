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
| cartoboost | ok | 0.360522 | 0.280265 | 0.733712 | 0.157954 | 0.000304 | 6572634.19 | n_estimators=20 |
| cartoboost_reference | ok | 0.360522 | 0.280265 | 0.733712 | 0.102317 | 0.000416 | 4803362.29 | n_estimators=20 |
| cartoboost_neural | ok | 0.364934 | 0.283791 | 0.727156 | 0.886310 | 0.003113 | 642467.07 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.360661 | 0.280017 | 0.733508 | 0.434497 | 0.000231 | 8645508.19 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.360601 | 0.280430 | 0.733596 | 0.250881 | 0.000326 | 6140469.33 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.360595 | 0.280426 | 0.733604 | 0.165802 | 0.000214 | 9365882.76 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.360601 | 0.280430 | 0.733596 | 0.529095 | 0.000254 | 7866273.34 | n_estimators=20 |
| lightgbm | ok | 0.365159 | 0.283737 | 0.726819 | 0.040194 | 0.000518 | 3859454.13 | n_estimators=20 |
| xgboost | ok | 0.364763 | 0.284134 | 0.727412 | 0.030280 | 0.000407 | 4916009.99 | n_estimators=20 |
| mean | ok | 0.698645 | 0.558907 | -0.000001 | 0.000040 | 0.000008 | 265181799.62 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.365853 | 0.283224 | 0.691871 | 1.227037 | 0.000402 | 6490258.39 | n_estimators=20 |
| cartoboost_reference | ok | 0.365853 | 0.283224 | 0.691871 | 1.201944 | 0.000326 | 8004100.35 | n_estimators=20 |
| cartoboost_neural | ok | 0.371984 | 0.289306 | 0.681456 | 0.683845 | 0.016898 | 154340.07 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.365868 | 0.283218 | 0.691844 | 0.203499 | 0.000376 | 6933865.12 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.365868 | 0.283218 | 0.691844 | 0.271656 | 0.000435 | 5994244.85 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.365868 | 0.283218 | 0.691844 | 0.260409 | 0.000450 | 5791797.34 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.365868 | 0.283218 | 0.691844 | 0.158198 | 0.000444 | 5868917.01 | n_estimators=20 |
| lightgbm | ok | 0.371379 | 0.288710 | 0.682493 | 0.251114 | 0.000514 | 5072291.82 | n_estimators=20 |
| xgboost | ok | 0.371903 | 0.289009 | 0.681596 | 0.255802 | 0.000265 | 9852365.49 | n_estimators=20 |
| mean | ok | 0.667082 | 0.528604 | -0.024420 | 0.000013 | 0.000008 | 319333885.68 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.200199 | 0.153503 | 0.850829 | 0.102151 | 0.000268 | 7448789.57 | n_estimators=20 |
| cartoboost_reference | ok | 0.200199 | 0.153503 | 0.850829 | 0.101575 | 0.000287 | 6977758.40 | n_estimators=20 |
| cartoboost_neural | ok | 0.201290 | 0.155061 | 0.849199 | 0.721847 | 0.003860 | 518157.13 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.199955 | 0.153359 | 0.851193 | 0.231386 | 0.000216 | 9246758.74 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.200028 | 0.153017 | 0.851083 | 0.227083 | 0.000295 | 6787322.66 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.199794 | 0.153061 | 0.851433 | 0.330642 | 0.000307 | 6511116.11 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.199625 | 0.152911 | 0.851683 | 0.119382 | 0.000292 | 6840530.14 | n_estimators=20 |
| lightgbm | ok | 0.201223 | 0.154882 | 0.849300 | 0.274525 | 0.000491 | 4072971.39 | n_estimators=20 |
| xgboost | ok | 0.201669 | 0.155100 | 0.848630 | 0.267866 | 0.000286 | 6988926.05 | n_estimators=20 |
| mean | ok | 0.518414 | 0.398975 | -0.000263 | 0.000177 | 0.000004 | 465983215.36 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.185314 | 0.144115 | 0.810903 | 3.785752 | 0.000322 | 8099378.89 | n_estimators=20 |
| cartoboost_reference | ok | 0.185314 | 0.144115 | 0.810903 | 3.659503 | 0.001396 | 1867916.53 | n_estimators=20 |
| cartoboost_neural | ok | 0.185545 | 0.144870 | 0.810430 | 1.252604 | 0.048421 | 53860.97 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.186392 | 0.144458 | 0.808697 | 7.586801 | 0.000456 | 5719812.60 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.188300 | 0.144515 | 0.804759 | 0.254067 | 0.000656 | 3974349.59 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.183799 | 0.142815 | 0.813983 | 0.420318 | 0.000718 | 3633364.55 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.186023 | 0.144213 | 0.809453 | 0.743743 | 0.000590 | 4422835.28 | n_estimators=20 |
| lightgbm | ok | 0.185852 | 0.145070 | 0.809804 | 0.316807 | 0.000988 | 2640456.49 | n_estimators=20 |
| xgboost | ok | 0.185979 | 0.144976 | 0.809542 | 0.240412 | 0.000509 | 5120834.38 | n_estimators=20 |
| mean | ok | 0.445019 | 0.356651 | -0.090500 | 0.000036 | 0.000013 | 194990654.49 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.727320 | 0.592363 | 0.873058 | 3.828866 | 0.006331 | 778728.49 | n_estimators=20 |
| cartoboost_reference | ok | 0.727320 | 0.592363 | 0.873058 | 3.813791 | 0.001046 | 4713945.73 | n_estimators=20 |
| cartoboost_neural | ok | 0.702889 | 0.576462 | 0.881442 | 1.129303 | 0.005668 | 869833.71 | n_estimators=20 |
| cartoboost_graph_node2vec | ok | 0.719462 | 0.585209 | 0.875786 | 0.903050 | 0.000575 | 8576403.92 | n_estimators=20 |
| cartoboost_graph_graphsage | ok | 0.719462 | 0.585209 | 0.875786 | 0.350738 | 0.000627 | 7861798.13 | n_estimators=20 |
| cartoboost_graph_hetero_graphsage | ok | 0.719462 | 0.585209 | 0.875786 | 0.505996 | 0.000709 | 6951416.18 | n_estimators=20 |
| cartoboost_graph_hinsage | ok | 0.719462 | 0.585209 | 0.875786 | 0.769274 | 0.000622 | 7924987.55 | n_estimators=20 |
| lightgbm | ok | 0.740537 | 0.602401 | 0.868402 | 0.112001 | 0.000583 | 8461093.63 | n_estimators=20 |
| xgboost | ok | 0.740934 | 0.602332 | 0.868261 | 0.109008 | 0.000433 | 11379111.29 | n_estimators=20 |
| mean | ok | 2.041944 | 1.752775 | -0.000560 | 0.000037 | 0.000008 | 613031619.58 |  |

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
| mean | ok | 2.088607 | 1.807484 | -0.002958 | 0.000007 | 0.000005 | 934363938.08 |  |

