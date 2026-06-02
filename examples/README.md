GeoBoost examples scaffold
==========================

This directory holds lightweight example assets for early GeoBoost development.
The scripts intentionally use synthetic data and Python standard-library code so
they can run before the Rust and Python package APIs are finalized.

Suggested workflow once the project API exists:

1. Generate regression, binary, and ranking-shaped CSV files with
   `scripts/generate_synthetic_data.py`.
2. Train GeoBoost models with the CLI or Python package.
3. Write predictions to CSV with `target` and `prediction` columns.
4. Summarize metrics with `scripts/validation_report.py`.
5. Plot benchmark summaries with `scripts/plot_benchmarks.py`.

