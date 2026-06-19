# Benchmarks

These pages are evidence for model choice, not a command catalog. Read them as
small studies: each page states the target, data, split design, baselines,
metrics, result, and the decision a scientist can reasonably make from that
result.

CartoBoost is designed for taxi-shaped regression and forecasting problems:
pickup/dropoff zones, lane demand, route geometry, periodic hour/day effects,
and residual structure attached to repeated IDs or graph topology. The
benchmarks ask whether those structures help under honest validation. They also
show cases where the right conclusion is to use a simpler baseline.

## How To Read The Evidence

Use the maintained reports in this order:

| Report | Model-choice question |
| --- | --- |
| [Fair Benchmarking Program](fair-benchmarking.md) | What protocol is required before a CartoBoost comparison can support a public model-quality claim? |
| [NYC Taxi Benchmarks](nyc-taxi.md) | On real January 2024 yellow taxi fare, duration, and pickup-demand tasks, do CartoBoost feature families beat LightGBM and XGBoost under comparable settings and random/spatial splits? |
| [Forecasting Tool Benchmark](forecasting.md) | For pickup/dropoff lane demand panels, when does `cartoboost_lag` beat, tie, or defer to forecasting-library baselines such as seasonal naive, StatsForecast, Prophet, and functime? |
| [Model Benchmark Suite](model-suite.md) | In controlled synthetic regression tasks, which mechanism is being tested: dense numeric modeling, repeated-ID residual learning, or graph topology? |
| [Taxi Zone Acceptance](taxi-zone.md) | Can the implementation express lane membership, route geometry, cyclic hour, and combined geotemporal structure before claiming real-data quality? |
| [Neural Embedding Strategy Assessment](neural-embedding-strategy.md) | Do residual embeddings improve repeated-ID tasks while reducing leakage and avoiding cold-ID overclaims? |
| [Neural Embedding Benchmark](neural-embedding-benchmark-latest.md) | Across random, temporal, geographic, tail, and cold-ID splits, where do neural residual embeddings reduce MAE and where do they fail? |

## Evidence Rules

Trust a benchmark when it includes the exact command, fixed splits, comparable
model settings, serious baselines, and stable artifact paths. Treat a result as
exploratory when it lacks any of those details.

Public quality claims must also satisfy the manifest-driven protocol in the
[Fair Benchmarking Program](fair-benchmarking.md): repeated seeds or outer
folds, equal HPO budgets across required model families, train-only feature
engineering, uncertainty intervals, baseline-completeness checks, subgroup
slices, and compute metadata. A missing required baseline makes a benchmark
incomplete rather than a win.

The strongest CartoBoost evidence is a gain on the split that mirrors
deployment: out-of-time for future demand, spatial holdout for new pickup
zones, grouped holdout for new routes or lanes, and cold-ID splits when IDs will
be unseen. Random splits are still useful, but they mostly measure
interpolation.

The taxi-domain examples are intentionally stable across the docs. Benchmarks
should describe pickup/dropoff zones, `PULocationID`, `DOLocationID`, taxi
trips, fare, duration, trip distance, hour/day features, lanes, and demand
panels unless a page explicitly says it is a synthetic mechanism test.

## What Counts As A Win

A CartoBoost win means the pre-registered CartoBoost row improves the primary
metric under the same split, comparable feature access, equal model-selection
budget, and complete required-baseline set. A tie can still be a good result
when CartoBoost provides useful artifacts or guards against overfitting. A
baseline win should be reported plainly and should influence the
recommendation.

For performance-sensitive deployments, compare training time and prediction
throughput separately from quality. A better RMSE may not justify slower
scoring for every production route.
