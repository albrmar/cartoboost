# NYC Taxi Model Quality Benchmarks

These artifacts compare predictive quality and speed on NYC TLC taxi-derived tasks.
Quality metrics are computed on transformed regression targets.

- dataset source: nyc_tlc_trip_records
- models requested: geoboost, lightgbm, xgboost, mean

## Trip duration

Predict log trip duration from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.326123 | 0.254579 | 0.774539 | 46.897852 | 0.012295 | 81334.70 |  |
| lightgbm | ok | 0.333662 | 0.260228 | 0.763994 | 0.010757 | 0.003048 | 328058.26 |  |
| xgboost | ok | 0.338152 | 0.264688 | 0.757600 | 0.013855 | 0.001036 | 965619.13 |  |
| mean | ok | 0.686834 | 0.552704 | -0.000027 | 0.000041 | 0.000009 | 110399472.96 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.349664 | 0.268514 | 0.799682 | 44.810052 | 0.013152 | 82422.31 |  |
| lightgbm | ok | 0.354319 | 0.272841 | 0.794313 | 0.066968 | 0.000924 | 1172911.36 |  |
| xgboost | ok | 0.364518 | 0.281228 | 0.782301 | 0.015407 | 0.000786 | 1379638.63 |  |
| mean | ok | 0.853364 | 0.702886 | -0.193128 | 0.000040 | 0.000009 | 114733562.49 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.162875 | 0.125421 | 0.905062 | 46.849530 | 0.011531 | 86722.79 |  |
| lightgbm | ok | 0.167747 | 0.130096 | 0.899297 | 0.074128 | 0.001410 | 709084.08 |  |
| xgboost | ok | 0.171794 | 0.133952 | 0.894380 | 0.010821 | 0.000741 | 1349134.44 |  |
| mean | ok | 0.528609 | 0.406263 | -0.000001 | 0.000049 | 0.000009 | 109075071.80 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.203373 | 0.157985 | 0.915161 | 44.201134 | 0.013067 | 82954.44 |  |
| lightgbm | ok | 0.206964 | 0.161577 | 0.912139 | 0.084999 | 0.000990 | 1094710.64 |  |
| xgboost | ok | 0.232830 | 0.184341 | 0.888805 | 0.011213 | 0.000750 | 1444553.24 |  |
| mean | ok | 0.802193 | 0.618637 | -0.319969 | 0.000055 | 0.000010 | 104825468.15 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.278787 | 0.238052 | 0.079470 | 19.824430 | 0.005968 | 103720.45 |  |
| lightgbm | ok | 0.279479 | 0.236981 | 0.074896 | 0.008834 | 0.000963 | 642627.48 |  |
| xgboost | ok | 0.279568 | 0.237751 | 0.074303 | 0.009390 | 0.000717 | 862923.46 |  |
| mean | ok | 0.291315 | 0.256008 | -0.005124 | 0.000039 | 0.000008 | 74614373.89 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.333580 | 0.268546 | -0.049927 | 18.545814 | 0.006190 | 101770.29 |  |
| lightgbm | ok | 0.329526 | 0.264805 | -0.024560 | 0.008210 | 0.000942 | 669030.59 |  |
| xgboost | ok | 0.328253 | 0.263268 | -0.016661 | 0.009566 | 0.000785 | 802268.76 |  |
| mean | ok | 0.328596 | 0.276084 | -0.018787 | 0.000040 | 0.000022 | 28007467.02 |  |

