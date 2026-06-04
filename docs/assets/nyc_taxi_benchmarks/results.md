# NYC Taxi Model Quality Benchmarks

These artifacts compare predictive quality and speed on NYC TLC taxi-derived tasks.
Quality metrics are computed on transformed regression targets.

- dataset source: nyc_tlc_trip_records
- models requested: geoboost, geoboost_reference, lightgbm, xgboost, mean
- baseline estimators: 100
- GeoBoost candidate estimators: 100
- baseline max depth: 4
- GeoBoost candidate max depth: 4
- zone treatment: target_mean

## Trip duration

Predict log trip duration from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.318385 | 0.241976 | 0.794259 | 0.878132 | 0.007103 | 703915.53 | n_estimators=100 |
| geoboost_reference | ok | 0.318385 | 0.241976 | 0.794259 | 0.708692 | 0.009526 | 524879.28 | n_estimators=100 |
| lightgbm | ok | 0.299960 | 0.227121 | 0.817383 | 0.219078 | 0.002540 | 1968794.57 | n_estimators=100 |
| xgboost | ok | 0.300267 | 0.227814 | 0.817009 | 0.114830 | 0.001289 | 3879851.86 | n_estimators=100 |
| mean | ok | 0.701976 | 0.559078 | -0.000136 | 0.000014 | 0.000005 | 1008468688.16 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.339032 | 0.260560 | 0.762805 | 0.566307 | 0.006561 | 905591.41 | n_estimators=100 |
| geoboost_reference | ok | 0.339032 | 0.260560 | 0.762805 | 0.555587 | 0.005953 | 998117.31 | n_estimators=100 |
| lightgbm | ok | 0.323047 | 0.247877 | 0.784645 | 0.207989 | 0.002349 | 2529721.69 | n_estimators=100 |
| xgboost | ok | 0.323207 | 0.247685 | 0.784431 | 0.111166 | 0.000865 | 6873337.15 | n_estimators=100 |
| mean | ok | 0.697192 | 0.562186 | -0.003066 | 0.000013 | 0.000004 | 1345265509.95 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.157605 | 0.115509 | 0.908959 | 0.570269 | 0.004328 | 1155223.44 | n_estimators=100 |
| geoboost_reference | ok | 0.157605 | 0.115509 | 0.908959 | 0.556145 | 0.005757 | 868583.34 | n_estimators=100 |
| lightgbm | ok | 0.149526 | 0.108812 | 0.918054 | 0.234001 | 0.002330 | 2145462.35 | n_estimators=100 |
| xgboost | ok | 0.149721 | 0.108946 | 0.917840 | 0.116308 | 0.000852 | 5867394.37 | n_estimators=100 |
| mean | ok | 0.522368 | 0.400237 | -0.000112 | 0.000011 | 0.000004 | 1142856043.21 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.174823 | 0.132057 | 0.899358 | 0.499529 | 0.018588 | 319675.05 | n_estimators=100 |
| geoboost_reference | ok | 0.174823 | 0.132057 | 0.899358 | 0.483134 | 0.004123 | 1441052.54 | n_estimators=100 |
| lightgbm | ok | 0.170288 | 0.127065 | 0.904512 | 0.188252 | 0.002453 | 2422422.94 | n_estimators=100 |
| xgboost | ok | 0.171819 | 0.127741 | 0.902787 | 0.127555 | 0.001074 | 5531301.05 | n_estimators=100 |
| mean | ok | 0.555986 | 0.428471 | -0.017905 | 0.000014 | 0.000006 | 1011401564.33 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.398324 | 0.317630 | 0.576531 | 0.129874 | 0.001420 | 917363.36 | n_estimators=100 |
| geoboost_reference | ok | 0.398324 | 0.317630 | 0.576531 | 0.116880 | 0.001351 | 964410.77 | n_estimators=100 |
| lightgbm | ok | 0.395695 | 0.315427 | 0.582101 | 0.175687 | 0.001332 | 978534.56 | n_estimators=100 |
| xgboost | ok | 0.392644 | 0.313509 | 0.588522 | 0.103073 | 0.000607 | 2145594.11 | n_estimators=100 |
| mean | ok | 0.612315 | 0.521487 | -0.000689 | 0.000006 | 0.000004 | 332650303.84 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.603939 | 0.504713 | 0.045199 | 0.109860 | 0.001347 | 1182226.10 | n_estimators=100 |
| geoboost_reference | ok | 0.603939 | 0.504713 | 0.045199 | 0.104872 | 0.001536 | 1036996.63 | n_estimators=100 |
| lightgbm | ok | 0.592859 | 0.494358 | 0.079910 | 0.173041 | 0.001092 | 1458513.36 | n_estimators=100 |
| xgboost | ok | 0.598158 | 0.489593 | 0.063390 | 0.105929 | 0.000602 | 2645629.99 | n_estimators=100 |
| mean | ok | 0.618753 | 0.524348 | -0.002218 | 0.000005 | 0.000004 | 364113935.37 |  |
