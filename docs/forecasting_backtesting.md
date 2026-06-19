# Forecasting Backtesting

Forecasting evidence should mimic the moment a forecast would have been made.
CartoBoost therefore uses rolling-origin validation: each fold trains on
observations at or before an origin timestamp and validates only on future
timestamps. The splitter hard-fails if `max(train timestamp) >= min(validation
timestamp)`.

Use this workflow when comparing taxi-demand models, tuning lag features, or
deciding whether a more complex surface is worth its cost. Random
cross-validation and shuffled validation rows are invalid for forecasting
because they allow future trip information to influence training.

## Rolling-Origin Protocol

The same protocol applies to single-series and panel taxi datasets. For panel
data, pass `series_id_col` so fold metadata records which pickup zones or
pickup/dropoff lanes were covered by each validation window.

```python
from cartoboost.forecasting import ExpandingWindowSplitter, RollingOriginBacktester

splitter = ExpandingWindowSplitter(
    horizon=24,
    step=24,
    min_train_size=24 * 14,
    timestamp_col="pickup_hour",
    series_id_col="PULocationID",
)

backtester = RollingOriginBacktester(
    splitter=splitter,
    target_col="fare",
    timestamp_col="pickup_hour",
    series_id_col="PULocationID",
    feature_cols=["pickup_hour", "day_of_week", "trip_distance", "PULocationID"],
)
result = backtester.run(model, taxi_trips)
```

Use `ExpandingWindowSplitter` when every earlier taxi trip should remain
eligible for training. Use `SlidingWindowSplitter` when old behavior should age
out, for example when a dispatch policy, fare rule, or demand pattern changed
and only the most recent `max_train_size` timestamps should define the training
window.

## Comparable Evidence

A useful backtest holds the question fixed across models:

- same input rows and filtering rules;
- same panel identity, timestamp column, target column, and frequency;
- same horizon, origin spacing, and minimum training window;
- same covariate role assumptions;
- same metrics and row alignment rules.

Changing any of those settings changes the experiment. Record the splitter
settings with the metrics when reporting a result, especially for benchmark
claims.

## Metrics

`ForecastMetricSet` reports MAE, RMSE, zero-safe MAPE, sMAPE, MASE, WAPE, and
bias. Dataframe metric evaluation is keyed, not positional: rows must be unique
by `series_id`, `timestamp`, and `horizon`, and separate actual/prediction
frames are inner-aligned on those three columns before scoring.

When horizon or series identifiers are available, CartoBoost also returns
per-horizon and per-series metric dictionaries. These breakdowns are often more
useful than a single aggregate for taxi work. A model that improves airport
pickup zones but fails outer-borough pickup zones should not be treated as a
uniform win.

Quantile forecasts can be scored with pinball loss. Interval forecasts report
empirical coverage and mean interval width. For scientific reporting, coverage
and width should be read together: a very wide interval may cover well without
being operationally useful.

Backtesting uses the same keyed metric contract for row-level predictions. A
model output must cover exactly the validation rows for the fold, with no
missing, extra, or duplicate `series_id`/`timestamp`/`horizon` targets.

## Native Boundary

Rolling-origin validation rules and metric definitions are Rust-core behavior.
The Python `RollingOriginBacktester` exposes the contract and dataframe
ergonomics, while ForecastFrame evaluation delegates fold construction, model
fitting, prediction, and scoring to Rust forecasting bindings. Python fallback
forecasters are intentionally unavailable.

Use `result.to_json()` for stable structured output or `result.to_pandas()` when
pandas is installed and a row-level prediction table is needed.
