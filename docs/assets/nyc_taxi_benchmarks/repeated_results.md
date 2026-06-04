# Repeated NYC Taxi Speed Benchmark

- baseline estimators: 100; GeoBoost candidate estimators: 100
- baseline max depth: 4; GeoBoost candidate max depth: 4
- GeoBoost splitters: axis_histogram:64; XGBoost tree_method: hist
- zone treatment: target_mean
- gate requires train <= XGBoost, predict rows/sec >= XGBoost, and same quality as GeoBoost reference.

| task/split | train ratio vs XGBoost median | train ratio min-max | predict rps ratio vs XGBoost median | predict rps ratio min-max | RMSE delta vs Geo ref | R2 delta vs Geo ref | RMSE delta vs XGB | R2 delta vs XGB | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| duration/random | 13.01x | 13.01x-13.01x | 0.479x | 0.479x-0.479x | 0.000000 | 0.000000 | 0.018119 | -0.022750 | miss |
| duration/spatial_holdout | 12.85x | 12.85x-12.85x | 0.429x | 0.429x-0.429x | 0.000000 | 0.000000 | 0.015825 | -0.021626 | miss |
| fare/random | 12.48x | 12.48x-12.48x | 0.486x | 0.486x-0.486x | 0.000000 | 0.000000 | 0.007884 | -0.008881 | miss |
| fare/spatial_holdout | 12.40x | 12.40x-12.40x | 0.493x | 0.493x-0.493x | 0.000000 | 0.000000 | 0.003004 | -0.003429 | miss |
| pickup_demand/random | 10.06x | 10.06x-10.06x | 0.202x | 0.202x-0.202x | 0.000000 | 0.000000 | 0.005680 | -0.011991 | miss |
| pickup_demand/spatial_holdout | 7.97x | 7.97x-7.97x | 0.541x | 0.541x-0.541x | 0.000000 | 0.000000 | 0.005781 | -0.018191 | miss |
