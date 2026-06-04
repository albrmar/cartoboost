# Repeated NYC Taxi Speed Benchmark

- baseline estimators: 100; GeoBoost candidate estimators: 100
- baseline max depth: 4; GeoBoost candidate max depth: 5
- GeoBoost splitters: axis_histogram:512; XGBoost tree_method: hist
- zone treatment: target_mean
- gate requires train <= XGBoost, predict rows/sec >= XGBoost, lower RMSE than XGBoost, and R2 no worse than XGBoost.

| task/split | train ratio vs XGBoost median | train ratio min-max | predict rps ratio vs XGBoost median | predict rps ratio min-max | RMSE delta vs Geo ref | R2 delta vs Geo ref | RMSE delta vs XGB | R2 delta vs XGB | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| duration/random | 1.87x | 1.81x-1.94x | 2.084x | 1.980x-2.192x | -0.005611 | 0.006801 | -0.004507 | 0.005453 | miss |
| duration/spatial_holdout | 1.83x | 1.75x-1.85x | 1.995x | 1.940x-2.198x | -0.004058 | 0.005427 | -0.001197 | 0.001594 | miss |
| fare/random | 1.86x | 1.79x-1.89x | 2.084x | 2.035x-2.442x | -0.001132 | 0.001246 | -0.000148 | 0.000163 | miss |
| fare/spatial_holdout | 1.79x | 1.71x-1.98x | 2.229x | 2.092x-2.629x | -0.003043 | 0.003412 | -0.003096 | 0.003472 | miss |
| pickup_demand/random | 2.13x | 2.01x-2.22x | 1.878x | 1.370x-2.230x | -0.013793 | 0.028507 | -0.012370 | 0.025518 | miss |
| pickup_demand/spatial_holdout | 1.65x | 1.62x-1.72x | 1.926x | 1.031x-2.186x | -0.006504 | 0.020389 | -0.002626 | 0.008206 | miss |
