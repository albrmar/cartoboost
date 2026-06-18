# Scripts Agent Guide

## Dev environment tips
- This folder contains validation, benchmark, plotting, synthetic data, and report scripts.
- Keep scripts runnable through `uv run --group dev python scripts/<name>.py` unless a benchmark dependency group is required.
- Avoid hard-coded machine paths.
- Benchmark scripts should default to honest, reproducible comparisons. Use the same split, same feature treatment, and comparable model settings across CartoBoost and baselines such as LightGBM/XGBoost.
- Do not add p-hacking or hyperparameter-search behavior to benchmark scripts unless the script explicitly compares equal search budgets for every model.
- Real geo benchmark paths should require real geometry and real data. If a required dataset, geometry archive, graph, or neural input is missing, fail clearly rather than falling back to a weaker proxy.
- For NYC taxi examples and reports, use taxi terminology: pickup/dropoff zones, trips, fares, durations, distance, hour, and day-of-week.

## Testing instructions
- Run the specific script after changes.
- Run validation script tests under `tests/integration` or `tests/python` when script contracts change.
- For benchmark changes, record the exact command, sample size, tasks, models, split names, RMSE, MAE, R2, train time, and prediction time.

## PR instructions
- Explain generated outputs and where they are written.
- Mention whether outputs belong under `target/` or committed asset folders.
- Identify whether benchmark evidence is real data, generated acceptance data, or synthetic smoke data.
