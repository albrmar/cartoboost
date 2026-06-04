# Repeated NYC Taxi Speed Benchmark

- baseline estimators: 100; GeoBoost candidate estimators: 100
- baseline max depth: 4; GeoBoost candidate max depth: 5
- GeoBoost splitters: axis_histogram:512; XGBoost tree_method: hist
- zone treatment: target_mean
- gate requires train <= XGBoost, predict rows/sec >= XGBoost, lower RMSE than XGBoost, and R2 no worse than XGBoost.

| task/split | train ratio vs XGBoost median | train ratio min-max | predict rps ratio vs XGBoost median | predict rps ratio min-max | RMSE delta vs Geo ref | R2 delta vs Geo ref | RMSE delta vs XGB | R2 delta vs XGB | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| duration/random | 1.76x | 1.66x-1.84x | 1.161x | 0.552x-2.009x | -0.005611 | 0.006801 | -0.004507 | 0.005453 | miss |
| duration/spatial_holdout | 1.73x | 1.70x-1.79x | 1.524x | 1.395x-1.944x | -0.004058 | 0.005427 | -0.001197 | 0.001594 | miss |
| fare/random | 1.70x | 1.67x-1.72x | 1.920x | 0.640x-2.208x | -0.001132 | 0.001246 | -0.000148 | 0.000163 | miss |
| fare/spatial_holdout | 1.73x | 1.56x-1.75x | 1.626x | 0.349x-1.838x | -0.003043 | 0.003412 | -0.003096 | 0.003472 | miss |
| pickup_demand/random | 1.91x | 1.80x-1.99x | 1.515x | 0.144x-2.135x | -0.013793 | 0.028507 | -0.012370 | 0.025518 | miss |
| pickup_demand/spatial_holdout | 1.49x | 0.76x-1.50x | 1.015x | 0.056x-2.112x | -0.006504 | 0.020389 | -0.002626 | 0.008206 | miss |
