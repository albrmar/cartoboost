# Benchmarks

These reports describe CartoBoost benchmarks as data-science studies. Each one
states the target, dataset, features, split design, comparison methods, metrics,
result, and interpretation. The goal is to explain what was measured and why it
matters.

| Report | Primary question |
| --- | --- |
| [Model Benchmark Suite](model-suite.md) | On controlled synthetic regression tasks, when do dense, ID, and graph features improve predictive quality against XGBoost and LightGBM? |
| [NYC Taxi Benchmarks](nyc-taxi.md) | On real NYC taxi fare, duration, and pickup-demand tasks, which feature families explain the quality gains under random and spatial holdout splits? |
| [Forecasting Tool Benchmark](forecasting.md) | On real NYC TLC pickup/dropoff lane demand, how does CartoBoost lag forecasting compare with functime, StatsForecast, and Prophet? |
| [Taxi Zone Acceptance](taxi-zone.md) | On a controlled taxi-lane fixture, can the model recover lane membership, route geometry, hour-of-day periodicity, and combined geotemporal structure? |
| [Neural Embedding Strategy Assessment](neural-embedding-strategy.md) | Do residual embeddings improve repeated-ID tasks without overstating cold-ID generalization? |
| [Neural Embedding Benchmark](neural-embedding-benchmark-latest.md) | Across random, temporal, geographic, tail, and cold-ID splits, where does neural augmentation reduce MAE? |

## Forecasting Benchmark

The maintained forecasting report is now
[Forecasting Tool Benchmark](forecasting.md). It uses cached real NYC TLC
yellow taxi trip records, aggregates top pickup/dropoff lanes into daily demand,
and commits direct tool-comparison plots for CartoBoost, functime,
StatsForecast, and Prophet.

## Evaluation Rules

- Report the command, data source, target, feature set, split, and metrics.
- Name every comparison library and model.
- Separate quality findings from timing observations.
- Do not generalize synthetic or single-domain evidence beyond the measured
  target and split design.
- Refresh the written report whenever benchmark artifacts or benchmark behavior
  change.

## Commands

Run dependency-light NYC smoke validation:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```

Run the synthetic model suite:

```sh
uv run --group dev --group bench python scripts/run_model_benchmark_suite.py
```

Run the maintained repeated NYC comparison:

```sh
just nyc-quality-benchmark-repeated
```

Refresh the full single-run NYC artifact:

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-download \
  --output-dir docs/assets/nyc_taxi_benchmarks
```
