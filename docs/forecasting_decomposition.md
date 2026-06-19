# Forecasting Decomposition

CartoBoost includes native Rust additive decomposition utilities for
forecasting workflows. The implementations are deterministic and are intended
for model construction, validation, and artifact inspection rather than
benchmark claims by themselves.

## STL

`STLDecomposition` decomposes one numeric series into:

- `trend`
- `seasonal`
- `remainder`

The recomposition contract is exact up to floating point arithmetic:

```text
observed[t] = trend[t] + seasonal[t] + remainder[t]
```

The implementation uses a centered moving-average trend estimate and an
additive seasonal phase pattern centered to mean zero. Short taxi histories are
accepted as long as the series has at least one finite observation; phases that
are not observed contribute zero after centering.

## MSTL

`MSTLDecomposition` extends the same additive contract to multiple seasonal
periods:

```text
observed[t] = trend[t] + seasonal_1[t] + ... + seasonal_k[t] + remainder[t]
```

Season lengths are sorted and de-duplicated during construction so reruns are
stable. Components are extracted sequentially from the residual after trend
removal.

## Decomposition Hybrids

`STLCartoBoostForecaster` and `MSTLCartoBoostForecaster` fit an existing native
forecasting model to the decomposition remainder, then recompose predictions by
adding deterministic trend and seasonal projections back to the remainder
forecast.

The default remainder model is native `auto_arima`. Callers can pass another
native Rust `Forecaster` implementation when a taxi panel should use a
different residual model.

These classes do not create benchmark evidence. Quality claims still require
held-out or rolling-origin validation with exact commands, data source, split,
model settings, metrics, and timing.
