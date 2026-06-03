# NYC Taxi Model Quality Benchmarks

These artifacts compare predictive quality on NYC TLC taxi-derived tasks.
Metrics are computed on transformed regression targets and are not runtime claims.

- dataset source: nyc_tlc_trip_records
- models requested: geoboost, lightgbm, xgboost, mean

## Trip duration

Predict log trip duration from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | note |
| --- | --- | ---: | ---: | ---: | --- |
| geoboost | ok | 0.326123 | 0.254579 | 0.774539 |  |
| lightgbm | ok | 0.333662 | 0.260228 | 0.763994 |  |
| xgboost | ok | 0.338152 | 0.264688 | 0.757600 |  |
| mean | ok | 0.686834 | 0.552704 | -0.000027 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | note |
| --- | --- | ---: | ---: | ---: | --- |
| geoboost | ok | 0.349664 | 0.268514 | 0.799682 |  |
| lightgbm | ok | 0.354319 | 0.272841 | 0.794313 |  |
| xgboost | ok | 0.364518 | 0.281228 | 0.782301 |  |
| mean | ok | 0.853364 | 0.702886 | -0.193128 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | note |
| --- | --- | ---: | ---: | ---: | --- |
| geoboost | ok | 0.162875 | 0.125421 | 0.905062 |  |
| lightgbm | ok | 0.167747 | 0.130096 | 0.899297 |  |
| xgboost | ok | 0.171794 | 0.133952 | 0.894380 |  |
| mean | ok | 0.528609 | 0.406263 | -0.000001 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | note |
| --- | --- | ---: | ---: | ---: | --- |
| geoboost | ok | 0.203373 | 0.157985 | 0.915161 |  |
| lightgbm | ok | 0.206964 | 0.161577 | 0.912139 |  |
| xgboost | ok | 0.232830 | 0.184341 | 0.888805 |  |
| mean | ok | 0.802193 | 0.618637 | -0.319969 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | note |
| --- | --- | ---: | ---: | ---: | --- |
| geoboost | ok | 0.278787 | 0.238052 | 0.079470 |  |
| lightgbm | ok | 0.279479 | 0.236981 | 0.074896 |  |
| xgboost | ok | 0.279568 | 0.237751 | 0.074303 |  |
| mean | ok | 0.291315 | 0.256008 | -0.005124 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | note |
| --- | --- | ---: | ---: | ---: | --- |
| geoboost | ok | 0.333580 | 0.268546 | -0.049927 |  |
| lightgbm | ok | 0.329526 | 0.264805 | -0.024560 |  |
| xgboost | ok | 0.328253 | 0.263268 | -0.016661 |  |
| mean | ok | 0.328596 | 0.276084 | -0.018787 |  |

