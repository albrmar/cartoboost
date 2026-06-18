# Forecasting Backtesting

CartoBoost forecasting validation uses rolling-origin folds. Each fold trains on
observations at or before an origin timestamp and validates only on future
timestamps. The splitter hard-fails if `max(train timestamp) >= min(validation
timestamp)`, so random cross-validation and shuffled validation rows are not
valid forecasting protocols.

The same rules apply to single-series and panel taxi datasets. For panel data,
pass `series_id_col` so fold metadata records the number of pickup or pickup to
dropoff series covered by each validation window.

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

Use `SlidingWindowSplitter` when old taxi trips should age out of training. Set
`max_train_size` to the fixed number of timestamps to keep. Use
`ExpandingWindowSplitter` when every earlier trip should remain eligible for
training.

`ForecastMetricSet` reports MAE, RMSE, zero-safe MAPE, sMAPE, MASE, WAPE, and
bias. Dataframe metric evaluation is keyed, not positional: rows must be unique
by `series_id`, `timestamp`, and `horizon`, and separate actual/prediction frames
are inner-aligned on those three columns before scoring. When horizon or series
identifiers are available, it also returns per-horizon and per-series metric
dictionaries. Quantile forecasts can be scored with pinball loss, and interval
forecasts report empirical coverage and mean interval width.

Backtesting uses the same keyed metric contract for row-level predictions. A
model output must cover exactly the validation rows for the fold, with no
missing, extra, or duplicate `series_id`/`timestamp`/`horizon` targets.

Rolling-origin validation rules and metric definitions are Rust-core behavior.
The Python `RollingOriginBacktester` exposes the contract and dataframe
ergonomics, while ForecastFrame evaluation delegates fold construction, model
fitting, prediction, and scoring to Rust forecasting bindings. Python fallback
forecasters are intentionally unavailable. Use `result.to_json()` for stable structured output or
`result.to_pandas()` when pandas is installed and a row-level prediction table is
needed.
