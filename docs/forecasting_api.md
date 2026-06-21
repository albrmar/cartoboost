# Forecasting API

The forecasting API defines the experiment boundary before a model sees any
data. `ForecastFrame` records what the panel is, which timestamps are valid,
which column is the target, which covariates are allowed at forecast time, and
whether the data are regular enough for the selected workflow. `ForecastResult`
records predictions in stable columns so outputs can be compared, serialized,
and scored without relying on model-specific shapes.

These classes are wrapper concerns. They do not decide whether theta, ETS,
ARIMA, lag boosting, or an ensemble is scientifically appropriate. They make
sure every model receives the same taxi panel contract and emits auditable
forecast rows.

## ForecastFrame

Use `ForecastFrame.from_pandas` when turning taxi trip aggregates into a
forecasting panel. A common demand panel has one series per pickup zone, hourly
timestamps, and a target such as `pickup_trips`. A fare or duration panel might
use pickup to dropoff lanes as the series identity.

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

The constructor makes the dataset contract explicit:

- timestamps are parsed with pandas and null or unparseable values are rejected;
- targets must be finite numeric values;
- rows are sorted deterministically by series id and timestamp for panels, or by
  timestamp for single-series data;
- duplicate timestamps are rejected within each series;
- frequency is inferred when possible, or an explicit frequency is validated;
- irregular data are rejected unless `allow_irregular=True`;
- static, known-future, and historical covariate roles are recorded.

Panel validation is isolated per series. Duplicate timestamps are only compared
within each pickup zone or pickup/dropoff lane, and every regular panel must use
one shared frequency. This matters for scientific comparison: two models should
not be scored on subtly different panel definitions.

`ForecastFrame.to_metadata()` returns a JSON-friendly summary that can be saved
with a run, logged beside a benchmark, or embedded in an artifact manifest:

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

Treat this metadata as part of the result. If a later run changes
`series_id_col`, `freq`, horizon, or covariate roles, it is a different
forecasting experiment even when the model name stays the same.

## Covariate Roles

CartoBoost separates covariates by when they are knowable:

- static covariates do not change within a series, such as a fixed taxi-zone or
  route descriptor;
- known-future covariates are available for future timestamps before prediction,
  such as hour, day-of-week, holiday flags, or published dispatch plans;
- historical covariates are observed only after the fact, such as realized trip
  distance, queue length, or completed fare totals.

Use these roles to prevent accidental leakage. A historical value from the
validation horizon should not be used as if it were known at forecast creation
time.

`AutoForecaster` uses numeric `static_covariates` automatically when fitting the
native guarded lag spine. Pass `covariate_features=[]` to disable that behavior,
or pass an explicit list to use a narrower native covariate set. Known-future
and historical covariates remain part of the experiment contract; they are not
silently promoted into recursive native lag features. `AutoForecaster.metadata_`
records both the configured covariate override and the effective covariate list
used during fit.

`AutoForecaster` also accepts `ewm_alpha_percents=[...]` to opt into native
exponentially weighted target-mean features. The default is empty: EWM is a
production feature for validation-gated experiments, not a blanket benchmark
default.

Candidate selection uses deterministic rolling-origin validation in the native
model. `validation_window` controls the horizon-sized trailing window and
`validation_origin_count` controls how many non-overlapping trailing origins are
averaged when history supports them. The default origin count is `2`; set
`validation_origin_count=1` for the previous single-holdout behavior when
latency is more important than selector stability.

## ForecastResult

`ForecastResult` standardizes predictions into deterministic columns:

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
through API boundaries. Use `to_pandas()` when pandas is installed and the next
step is row-level scoring, plotting, or artifact writing.

`PredictionInterval` validates that levels are unique and between 0 and 1, that
lower and upper bounds have the same length as the predictions, and that all
bounds are finite with `lower <= upper`. Interval validity is part of the
forecast contract, not a plotting detail.

## Base Classes

`BaseForecaster` provides shared fitted-state and positive-integer horizon
validation. `SingleSeriesForecasterMixin` rejects panel frames, and
`PanelForecasterMixin` rejects single-series frames so estimator implementations
fail before training state is mutated.

Those checks keep wrapper behavior predictable: a model either accepts the
declared experiment shape or fails before producing evidence.
