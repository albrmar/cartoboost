# Repeated NYC Taxi Speed Benchmark

| task/split | train ratio median | train ratio min-max | predict rps ratio median | predict rps ratio min-max | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| duration/random | 22.22x | 18.55x-22.95x | 0.339x | 0.301x-0.441x | miss |
| duration/spatial_holdout | 23.99x | 23.67x-24.05x | 0.323x | 0.316x-0.333x | miss |
| fare/random | 24.93x | 24.60x-25.89x | 0.307x | 0.287x-0.311x | miss |
| fare/spatial_holdout | 24.81x | 24.19x-25.01x | 0.322x | 0.310x-0.336x | miss |
| pickup_demand/random | 16.74x | 16.03x-17.33x | 0.447x | 0.383x-0.621x | miss |
| pickup_demand/spatial_holdout | 17.35x | 16.64x-17.55x | 0.508x | 0.485x-0.529x | miss |
