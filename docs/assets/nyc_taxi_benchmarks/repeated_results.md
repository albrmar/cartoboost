# Repeated NYC Taxi Speed Benchmark

| task/split | train ratio median | train ratio min-max | predict rps ratio median | predict rps ratio min-max | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| duration/random | 58.87x | 42.62x-59.65x | 0.325x | 0.318x-0.479x | miss |
| duration/spatial_holdout | 58.90x | 58.58x-61.94x | 0.328x | 0.288x-0.345x | miss |
| fare/random | 62.86x | 55.10x-63.80x | 0.313x | 0.218x-0.337x | miss |
| fare/spatial_holdout | 64.39x | 63.78x-65.87x | 0.342x | 0.334x-0.366x | miss |
| pickup_demand/random | 44.46x | 41.76x-45.67x | 0.482x | 0.445x-0.683x | miss |
| pickup_demand/spatial_holdout | 45.04x | 44.60x-46.35x | 0.515x | 0.476x-0.516x | miss |
