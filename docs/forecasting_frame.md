# Forecasting Frame

`ForecastFrame` is the Rust-owned container for taxi forecasting history. It
stores sorted `(series_id, timestamp, target)` rows, validates finite targets,
rejects duplicate timestamps within a pickup/dropoff lane, and checks regular
hourly, daily, or weekly spacing before model code sees the data.

Use one `series_id` per taxi panel member, such as a pickup zone, dropoff zone,
or pickup/dropoff lane. Single-series data uses the internal single-series id,
but public examples should still describe taxi demand, fare, duration, trip
distance, and hour/day features.

## Deterministic Diagnostics

`ForecastDiagnostics::from_frame(&frame)` summarizes an existing frame without
modifying it. Output is deterministic because `ForecastFrame` sorts rows by
series id and timestamp during construction.

The frame summary reports row count, series count, total zero count, weighted
zero fraction, and the number of intermittent series. Each
`SeriesDiagnostics` row reports:

- first and last timestamp;
- min, max, and mean target;
- zero and nonzero counts;
- zero fraction;
- zero-to-nonzero intermittency ratio;
- mean interval between nonzero observations when at least two nonzero targets
  exist;
- longest zero run;
- whether the series contains intermittent zero demand.

These diagnostics are descriptive. They do not replace rolling-origin
backtests, and they do not make benchmark quality claims.
