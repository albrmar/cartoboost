# Forecasting API

CartoBoost forecasting starts with explicit time-series contracts. The Python
objects in `cartoboost.forecasting` validate pandas inputs, describe forecasting
metadata, and standardize forecast outputs. Model-specific estimators are thin
wrappers over Rust bindings exposed through `cartoboost._native`.

## ForecastFrame

Use `ForecastFrame.from_pandas` for both single-series and panel taxi demand or
fare-duration datasets.

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    taxi_trips,
    timestamp_col="pickup_hour",
    target_col="fare",
    series_id_col="PULocationID",  # omit for one series
    freq="h",
    static_covariates=["DOLocationID"],
    known_future_covariates=["hour", "day_of_week"],
    historical_covariates=["trip_distance"],
)
```

The constructor:

- parses timestamps with pandas and rejects unparseable or null values;
- validates that targets are finite numeric values;
- sorts deterministically by `series_id_col`, then timestamp for panels, or by
  timestamp for single-series data;
- rejects duplicate timestamps within a series;
- infers frequency when possible, or validates an explicit frequency;
- rejects irregular data unless `allow_irregular=True`;
- records static, known-future, and historical covariate column roles.

Panel validation is isolated per series. Duplicate timestamps are only compared
within each pickup or route series, and every regular panel must use one shared
frequency.

`ForecastFrame.to_metadata()` returns a JSON-friendly summary:

```python
{
    "timestamp_col": "pickup_hour",
    "target_col": "fare",
    "series_id_col": "PULocationID",
    "freq": "h",
    "is_panel": True,
    "n_rows": 5000,
    "series_ids": ["132", "138"],
    "static_covariates": ["DOLocationID"],
    "known_future_covariates": ["hour", "day_of_week"],
    "historical_covariates": ["trip_distance"],
    "allow_irregular": False,
}
```

## ForecastResult

`ForecastResult` standardizes model predictions into deterministic columns:

```python
from cartoboost.forecasting import ForecastResult, PredictionInterval

result = ForecastResult.from_predictions(
    series_id=[132, 132],
    timestamps=["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
    predictions=[18.25, 19.10],
    intervals=[
        PredictionInterval(level=0.9, lower=[14.0, 15.2], upper=[23.5, 24.1]),
    ],
)
```

Panel outputs are sorted by series id and timestamp. Columns are stable:
`series_id`, `timestamp`, `prediction`, then interval lower/upper pairs sorted
by interval level, such as `prediction_lower_90` and `prediction_upper_90`.

Use `to_json()` and `ForecastResult.from_json(...)` for deterministic roundtrips
through API boundaries.

`PredictionInterval` validates that levels are unique and between 0 and 1, that
lower and upper bounds have the same length as the predictions, and that all
bounds are finite with `lower <= upper`.

## Base Classes

`BaseForecaster` provides shared fitted-state and positive-integer horizon
validation. `SingleSeriesForecasterMixin` rejects panel frames, and
`PanelForecasterMixin` rejects single-series frames so estimator implementations
can fail before training state is mutated.
