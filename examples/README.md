# CartoBoost Examples

This directory holds lightweight examples for fitting and evaluating CartoBoost
regression models. The examples use synthetic data so the temporal-spatial
patterns are easy to inspect.

Taxi graph examples:

- `03_taxi_od_graph_regression.py` builds a pickup-dropoff zone graph for trip
  duration regression. It uses synthetic taxi-shaped rows by default and can
  read a local TLC-style CSV or Parquet file with `--input`.
- `04_taxi_pickup_zone_graph.py` builds a pickup-zone adjacency graph for demand
  modeling under a spatial holdout.
- `05_neural_embedding_regression.py` fits dense and neural embedding
  regressors on repeated IDs, then demonstrates a cold-ID guard that falls back
  to dense CartoBoost when embeddings are unsupported.

Suggested workflow:

1. Generate regression-shaped CSV files with
   `scripts/generate_synthetic_data.py`.
2. Train CartoBoost models with the CLI or Python estimator.
3. Write predictions to CSV with `target` and `prediction` columns.
4. Summarize metrics with `scripts/validation_report.py`.
5. Plot benchmark summaries with `scripts/plot_benchmarks.py`.
