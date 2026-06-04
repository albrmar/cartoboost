# Repeated NYC Taxi Speed Benchmark

- baseline estimators: 100; GeoBoost speed-preset estimators: 1
- baseline max depth: 4; GeoBoost speed-preset max depth: 0

| task/split | train ratio vs XGBoost median | train ratio min-max | predict rps ratio vs XGBoost median | predict rps ratio min-max | RMSE delta vs Geo ref | R2 delta vs Geo ref | RMSE delta vs XGB | R2 delta vs XGB | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| duration/random | 0.00x | 0.00x-0.00x | 178.146x | 155.889x-195.403x | 0.378605 | -0.787902 | 0.397408 | -0.811866 | pass |
| duration/spatial_holdout | 0.00x | 0.00x-0.00x | 366.285x | 228.562x-417.099x | 0.372197 | -0.785106 | 0.383907 | -0.800530 | pass |
| fare/random | 0.00x | 0.00x-0.00x | 312.769x | 275.533x-353.073x | 0.362154 | -0.906032 | 0.369880 | -0.914886 | pass |
| fare/spatial_holdout | 0.00x | 0.00x-0.00x | 410.109x | 209.014x-463.834x | 0.382081 | -0.918317 | 0.375709 | -0.910886 | pass |
| pickup_demand/random | 0.00x | 0.00x-0.00x | 104.899x | 95.791x-121.890x | 0.135955 | -0.395042 | 0.184356 | -0.511863 | pass |
| pickup_demand/spatial_holdout | 0.00x | 0.00x-0.00x | 119.068x | 108.585x-216.304x | -0.004419 | 0.014367 | -0.018383 | 0.060437 | pass |
