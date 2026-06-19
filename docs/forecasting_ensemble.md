# Forecasting Ensembles and Gating

CartoBoost forecasting ensembles are implemented in Rust under
`crates/cartoboost-core/src/forecasting`.

## Weighted Ensembles

`WeightedEnsembleForecaster` combines existing Rust `Forecaster` implementations by
normalizing non-negative member weights and averaging aligned forecast means. Every
member must produce the same forecast index: series id, timestamp, and horizon.

This is useful for taxi forecasting when combining simple baselines such as naive,
seasonal naive, and lag-based global forecasters across the same pickup/dropoff zone
panel.

## Rule-Based Gating

`RuleBasedGating` converts validation error scores into deterministic expert weights
with inverse-error weighting:

```text
weight(expert) = 1 / max(validation_error, error_floor)
```

Weights are then normalized to sum to one. The gating table supports global scores,
series-specific scores, and horizon-specific scores. For panel taxi frames, series
weights are averaged across pickup/dropoff zone series.

The gating code validates non-empty expert names, non-empty metric names, finite
non-negative scores, positive horizons, and positive error floors.

## Evidence Requirements

Do not describe an ensemble as better than a baseline without a real run. Benchmark
writeups should include the exact command, taxi data source, sample size, split
definition, model roster, comparable settings, RMSE, MAE, R2, training time,
prediction time, output artifacts, and limitations.
