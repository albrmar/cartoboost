# Benchmarks

These pages are the benchmark report hub. Each report should answer the same
reader questions:

- What data was used?
- What split was used?
- Which models saw which features?
- What command produced the artifact?
- What did the metric table say?
- Which plots should I inspect?
- What claim is allowed, and what claim is not allowed?

CartoBoost is strongest when the task has taxi-shaped structure: pickup and
dropoff zones, route distance, periodic hour/day effects, repeated IDs,
pickup/dropoff topology, or lane-demand history. The benchmark docs therefore
separate real NYC TLC evidence from synthetic mechanism checks.

## Report Map

| Report | Evidence type | What to inspect first |
| --- | --- | --- |
| [NYC Taxi Benchmarks](nyc-taxi.md) | Real TLC fare, duration, and pickup-demand regression. | Metric summary, predicted-vs-actual plots, throughput plots. |
| [Forecasting Tool Benchmark](forecasting.md) | Real taxi lane demand, synthetic taxi-shaped forecasting, M4 sample, M5 full-roster sample, and M5/M6 full-run protocols. | RMSE/WAPE tables, M5/M6 model rosters, run commands, horizon plot, forecast-line plot. |
| [Model Benchmark Suite](model-suite.md) | Synthetic dense, repeated-ID, and graph diagnostics. | MAE-by-model plot and workload table. |
| [Taxi Zone Acceptance](taxi-zone.md) | Deterministic taxi-lane feature acceptance. | Lane heatmap, hour profile, route midpoint geometry. |
| [Neural Embedding Benchmark](neural-embedding-benchmark-latest.md) | Synthetic repeated-ID/cold-ID diagnostic. | Scenario table showing random/tail wins and cold-origin failure. |

## Current Maintained Artifacts

| Artifact | Path |
| --- | --- |
| NYC regression JSON | `docs/assets/nyc_taxi_benchmarks/results.json` |
| NYC regression report | `docs/assets/nyc_taxi_benchmarks/results.md` |
| NYC repeated speed report | `docs/assets/nyc_taxi_benchmarks/repeated_results.md` |
| NYC forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json` |
| Forecasting overhaul committed suite JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite.json` |
| Forecasting overhaul full-roster committed suite JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite_full_roster.json` |
| Forecasting overhaul M4 JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m4_committed.json` |
| Forecasting overhaul M5 JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` |
| Forecasting overhaul M6 JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` |
| Synthetic forecasting suite JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_library_suite_synthetic.json` |
| M4 sample suite JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_m4_suite_sample.json` |
| M5 full-roster sample JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` |
| M5 full forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |
| M6 full forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |
| Model diagnostic suite JSON | `docs/assets/model_benchmarks/results.json` |
| Lane acceptance JSON | `docs/assets/lane_level_tests/acceptance_metrics.json` |

## Claim Rules

A result is usable as benchmark evidence when it names the dataset, command,
split, feature policy, models, metrics, and artifact path. It is only a public
quality claim when the comparison uses complete required baselines, same rows,
comparable feature access, no test-set peeking, equal tuning budget, and
uncertainty or repeatability evidence.

Random splits show interpolation. Spatial, grouped, cold-ID, or out-of-time
splits are the evidence for deployment risk. Synthetic fixtures are useful for
debugging and feature acceptance, not for real-world superiority claims.
