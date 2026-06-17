# CartoBoost Examples

This directory holds lightweight examples for fitting and evaluating CartoBoost
regression models. The examples use synthetic data so the temporal-spatial
patterns are easy to inspect.

Suggested workflow:

1. Generate regression-shaped CSV files with
   `scripts/generate_synthetic_data.py`.
2. Train CartoBoost models with the CLI or Python estimator.
3. Write predictions to CSV with `target` and `prediction` columns.
4. Summarize metrics with `scripts/validation_report.py`.
5. Plot benchmark summaries with `scripts/plot_benchmarks.py`.
