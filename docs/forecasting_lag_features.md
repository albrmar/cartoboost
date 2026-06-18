# Forecasting Lag Features

CartoBoost lag-based forecasting is Rust-owned. The public Python class
`CartoBoostLagForecaster` is a thin wrapper over
`cartoboost._native.CartoBoostLagForecaster`.

Python may still expose configuration objects such as `LagFeatureConfig`,
`RollingFeatureConfig`, and `CalendarFeatureConfig`, but supervised lag matrix
construction, recursive prediction, model fitting, and model prediction must be
owned by Rust.

Taxi-domain lag feature contracts should use columns such as `pickup_hour`,
`pickup_trips`, `PULocationID`, `DOLocationID`, pickup/dropoff lane identifiers,
known-future calendar or dispatch plans, and historical-only observed queue or
trip-distance features.
