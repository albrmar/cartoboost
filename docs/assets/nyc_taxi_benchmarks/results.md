# NYC Taxi Model Quality Benchmarks

These artifacts compare predictive quality and speed on NYC TLC taxi-derived tasks.
Quality metrics are computed on transformed regression targets.

- dataset source: nyc_tlc_trip_records
- models requested: geoboost, geoboost_reference, xgboost, mean
- baseline estimators: 100
- GeoBoost candidate estimators: 100
- baseline max depth: 4
- GeoBoost candidate max depth: 5
- zone treatment: target_mean

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 0.402378 | 0.307331 | 0.847020 | 7.495296 | 0.016674 | 116825.09 | n_estimators=100 |
| geoboost_reference | ok | 0.432521 | 0.332787 | 0.823241 | 5.951209 | 0.014182 | 137361.25 | n_estimators=100 |
| xgboost | ok | 0.412935 | 0.319783 | 0.838887 | 0.161172 | 0.000616 | 3164263.95 | n_estimators=100 |
| mean | ok | 1.028850 | 0.905170 | -0.000163 | 0.000021 | 0.000006 | 303567876.30 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| geoboost | ok | 1.025998 | 0.875732 | 0.046688 | 6.725872 | 0.021500 | 107304.62 | n_estimators=100 |
| geoboost_reference | ok | 1.004021 | 0.871453 | 0.087090 | 5.410095 | 0.060090 | 38392.62 | n_estimators=100 |
| xgboost | ok | 1.029376 | 0.888706 | 0.040400 | 0.199823 | 0.000902 | 2558123.28 | n_estimators=100 |
| mean | ok | 1.061005 | 0.951875 | -0.019476 | 0.000015 | 0.000010 | 220575276.15 |  |

