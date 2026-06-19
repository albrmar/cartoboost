# Benchmarks

These reports describe CartoBoost benchmarks as data-science studies. Each one
states the target, dataset, features, split design, comparison methods, metrics,
result, and interpretation. The goal is to explain what was measured and why it
matters.

| Report | Primary question |
| --- | --- |
| [Model Benchmark Suite](model-suite.md) | On controlled synthetic regression tasks, when do dense, ID, and graph features improve predictive quality against XGBoost and LightGBM? |
| [NYC Taxi Benchmarks](nyc-taxi.md) | On real NYC taxi fare, duration, and pickup-demand tasks, which feature families explain the quality gains under random and spatial holdout splits? |
| [Forecasting Tool Benchmark](forecasting.md) | On real NYC taxi pickup/dropoff lane demand panels, how does CartoBoost lag forecasting compare with dedicated forecasting tools? |
| [Taxi Zone Acceptance](taxi-zone.md) | On a controlled taxi-lane fixture, can the model recover lane membership, route geometry, hour-of-day periodicity, and combined geotemporal structure? |
| [Neural Embedding Strategy Assessment](neural-embedding-strategy.md) | Do residual embeddings improve repeated-ID tasks without overstating cold-ID generalization? |
| [Neural Embedding Benchmark](neural-embedding-benchmark-latest.md) | Across random, temporal, geographic, tail, and cold-ID splits, where does neural augmentation reduce MAE? |
