# ARIMA Examples

This page gives deterministic taxi-demand examples for checking fixed ARIMA and
AutoARIMA behavior before using real TLC-derived pickup/dropoff lane data. Use
them for API familiarity, plotting, model-order interpretation, and speed smoke
checks; do not treat their metrics as evidence that one model wins on real taxi
demand.

## Runnable Example

Run the example from the repository root:

```bash
uv run python examples/forecasting/arima_example_visualization.py
```

To write the forecast and residual diagnostic plot:

```bash
uv run python examples/forecasting/arima_example_visualization.py \
  --output target/examples/arima_example_visualization.png
```

The script creates two hourly pickup/dropoff lane series:

- `PU132->DO138`: airport-style morning ramp with stronger trend.
- `PU79->DO230`: flatter Midtown-style lane with weaker trend.

It fits:

- `ArimaForecaster(p=2, d=1, q=1)`
- `AutoARIMAForecaster(max_p=3, max_d=1, max_q=2)`

The JSON payload reports held-out MAE, RMSE, bias, maximum absolute error,
per-horizon residuals for `PU132->DO138`, the selected AutoARIMA order, and the
top AutoARIMA candidate scores.

## What The Plot Shows

When `--output` is provided, the plot overlays:

- observed pickup counts,
- the fixed-order ARIMA forecast,
- the selected AutoARIMA forecast, and
- the train/test boundary.

It also includes held-out residual bars by horizon. Residuals are
`prediction - actual`, so positive bars mean overprediction and negative bars
mean underprediction.

The output path should normally live under `target/` or `/tmp` during local
experiments:

```bash
open target/examples/arima_example_visualization.png
```

Generated plots are not committed by default. If benchmark or docs assets are
intentionally committed, update the benchmark narrative with the command, data
source, metrics, and reason the artifact is meaningful.

## Reading The Output

The compact fields are the ones to check first:

```json
{
  "auto_arima_selected_label": "ARIMA(3,0,0)",
  "heldout_winner_by_rmse": "arima_2_1_1",
  "arima_2_1_1": {"mae": 1.49, "rmse": 1.72, "bias": -0.04},
  "auto_arima": {"mae": 9.67, "rmse": 10.48, "bias": -9.67}
}
```

Exact values can change if you pass different `--hours`, `--train-hours`, or
`--horizon` settings. The important lesson is that AutoARIMA selection metadata
and held-out performance answer different questions. The selected order is the
best order by native fitted residual score inside the bounded grid after each
candidate's lag warm-up; the held-out winner is the model that scored best on
unseen taxi pickup hours.

Use the detailed fields for diagnosis:

| Field | Use |
| --- | --- |
| `auto_arima_top_candidates` | Check whether the selected order was a clear winner or one of several close orders. |
| `residuals.arima_2_1_1` | Inspect horizon-by-horizon fixed ARIMA misses for the plotted lane. |
| `residuals.auto_arima` | Inspect horizon-by-horizon AutoARIMA misses for the plotted lane. |
| `bias` | Detect directional overprediction or underprediction that RMSE alone can hide. |
| `max_abs_error` | Find worst operational miss on the held-out tail. |

## Model-Order Interpretation

For taxi pickup/dropoff lanes, read `(p, d, q)` as the local memory structure:

| Order | Practical meaning |
| --- | --- |
| `p` | Number of recent differenced pickup-count lags used by the next forecast. |
| `d` | Number of differencing passes used to remove local level or trend movement. |
| `q` | Number of recent fitted residual lags used to correct the next forecast. |

Useful examples:

- `ARIMA(0,0,0)` is a constant local mean baseline.
- `ARIMA(1,0,0)` carries one recent local pickup-count lag.
- `ARIMA(0,1,0)` behaves like a random-walk-style local forecast.
- `ARIMA(2,1,1)` uses two recent differenced lags and one residual correction.

The current Rust binding is non-seasonal. If hour-of-day or day-of-week cycles
drive the lane, compare against seasonal naive, ETS, Theta, Kalman, and a lag
forecaster with calendar features before making a production choice.

## Held-Out Evaluation

Keep splits fixed across candidates. A practical taxi lane check is:

1. Train on the first `N` regular hourly observations for each lane.
2. Predict the same horizon for every candidate.
3. Join predictions to the held-out pickup counts by `lane_id` and
   `pickup_hour`.
4. Compute RMSE, MAE, bias, and horizon-specific residuals.
5. Inspect the residual plot before accepting a lower scalar metric.

This is exactly what the committed example does. It intentionally keeps the
data generator, train boundary, candidate orders, and horizon deterministic so
reruns are comparable.

## Code Shape

The example uses the same public APIs as production workflows:

```python
from cartoboost.forecasting import AutoARIMAForecaster, ForecastFrame
from cartoboost.forecasting.local import ArimaForecaster

frame = ForecastFrame.from_pandas(
    train,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="lane_id",
    freq="h",
)

model = AutoARIMAForecaster(max_p=3, max_d=1, max_q=2)
model.fit(frame)
forecast = model.predict(12)
```

`ForecastFrame` performs timestamp, duplicate-row, frequency, and finite-target
validation before the native model runs.

For held-out scoring, convert native prediction tuples to a table and join on
the actual taxi lane timestamps:

```python
forecast_rows = [
    {
        "lane_id": series_id,
        "pickup_hour": pd.Timestamp(timestamp),
        "horizon": horizon,
        "model": model_name,
        "prediction": prediction,
    }
    for series_id, timestamp, horizon, model_name, prediction in forecast.predictions()
]
forecast_table = pd.DataFrame(forecast_rows)
joined = actual.merge(forecast_table, on=["lane_id", "pickup_hour"])
joined["residual"] = joined["prediction"] - joined["pickup_count"]
```

## GIL Smoke Pattern

The ARIMA bindings release the Python GIL during native fit and predict. For
independent local lane series, you can schedule one-shot utility forecasts from
threads:

```python
from concurrent.futures import ThreadPoolExecutor

from cartoboost import auto_arima_forecast

lanes = {
    "PU132->DO138": [92.0, 89.0, 101.0, 118.0, 126.0, 119.0],
    "PU79->DO230": [63.0, 61.0, 68.0, 73.0, 75.0, 72.0],
}

with ThreadPoolExecutor(max_workers=2) as pool:
    forecasts = dict(
        zip(
            lanes,
            pool.map(
                lambda values: auto_arima_forecast(values, 3, max_p=2, max_d=1, max_q=1),
                lanes.values(),
            ),
        )
    )
```

This pattern is for independent local models. For a panel `ForecastFrame`, fit a
single native model on the panel and let Rust handle internal parallel work.

## Benchmark Command

For implementation-speed checks, run the native Criterion target:

```bash
cargo bench -p cartoboost-core --bench forecasting
```

The target measures fixed ARIMA and AutoARIMA fit+predict on deterministic
taxi-shaped lane panels. It complements, but does not replace, rolling-origin
quality benchmarks on real data.
