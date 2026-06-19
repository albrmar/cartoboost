# Repeated NYC Taxi Quality And Speed Benchmark

## Research Question

Across repeated NYC taxi runs, are the quality gains stable, and what operating
cost do they carry relative to XGBoost?

## Dataset

The repeated benchmark uses the same NYC TLC taxi-derived tasks as the
single-run report:

- Trip duration from pickup/dropoff, trip, passenger, and time features.
- Fare amount from pickup/dropoff, trip, passenger, and time features.
- Pickup-zone demand from zone-time demand features.

## Targets

The targets are transformed continuous regression targets. RMSE and R2 are the
quality metrics used for the repeated comparison.

## Features

The comparison keeps the feature construction fixed across runs: geographic
zone features, trip descriptors, hour/day features, and the configured
CartoBoost split families. XGBoost receives the same tabular inputs under its
histogram tree method.

## Comparison Method

- baseline estimators: 100; CartoBoost candidate estimators: 100
- baseline max depth: 4; CartoBoost candidate max depth: 5
- CartoBoost splitters: axis_histogram:512; XGBoost tree_method: hist
- zone treatment: target_mean
- gate requires train <= XGBoost, predict rows/sec >= XGBoost, lower RMSE than
  XGBoost, and R2 no worse than XGBoost.

## Results

| task/split | train ratio vs XGBoost median | train ratio min-max | predict rps ratio vs XGBoost median | predict rps ratio min-max | RMSE delta vs Carto ref | R2 delta vs Carto ref | RMSE delta vs XGB | R2 delta vs XGB | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| duration/random | 18.78x | 16.01x-20.79x | 0.287x | 0.076x-0.357x | -0.005611 | 0.006801 | -0.004507 | 0.005453 | miss |
| duration/spatial_holdout | 18.08x | 17.85x-18.35x | 0.361x | 0.128x-0.419x | -0.004058 | 0.005427 | -0.001197 | 0.001594 | miss |
| fare/random | 17.45x | 14.54x-18.69x | 0.234x | 0.134x-0.292x | -0.001132 | 0.001246 | -0.000148 | 0.000163 | miss |
| fare/spatial_holdout | 17.68x | 8.13x-17.91x | 0.319x | 0.077x-0.367x | -0.003043 | 0.003412 | -0.003096 | 0.003472 | miss |
| pickup_demand/random | 17.59x | 16.74x-20.49x | 0.297x | 0.016x-0.385x | -0.013793 | 0.028507 | -0.012370 | 0.025518 | miss |
| pickup_demand/spatial_holdout | 15.83x | 11.00x-16.22x | 0.310x | 0.095x-0.336x | -0.006504 | 0.020389 | -0.002626 | 0.008206 | miss |

## Interpretation

The repeated runs show consistent metric deltas against XGBoost on the measured
taxi tasks: RMSE deltas are negative and R2 deltas are positive in all reported
task/split rows. The combined speed gate still misses because the candidate
trains slower and predicts fewer rows per second than XGBoost on this run. The
correct conclusion is therefore quality-positive but speed-negative, not an
overall deployment recommendation.
