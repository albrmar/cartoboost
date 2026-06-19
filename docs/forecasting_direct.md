# Direct Forecasting

CartoBoost includes Rust-core direct multi-horizon forecasting for temporal taxi
signals such as pickup-zone demand, fare, duration, and trip-distance summaries.
The direct forecaster trains one native `Booster` per horizon and uses
`LagFeatureBuilder` features built from history that is strictly prior to the
forecast timestamp.

## Direct Strategy

`CartoBoostDirectForecaster` is deterministic for a fixed frame, lag feature
configuration, and booster configuration:

```rust
use cartoboost_core::booster::BoosterConfig;
use cartoboost_core::forecasting::{
    CartoBoostDirectForecaster, Forecaster, LagFeatureConfig,
};

let mut forecaster = CartoBoostDirectForecaster::new(
    LagFeatureConfig::default(),
    BoosterConfig::default(),
)?;
forecaster.fit_horizon(&taxi_zone_frame, 6)?;
let forecasts = forecaster.predict(6)?;
```

For horizon `h`, each training row uses lag features available at the origin and
the observed target at `origin + h`. Prediction uses the original fitted history
for every requested horizon instead of feeding horizon-one predictions into
later horizons.

## Rectified Recursive Strategy

`RectifiedRecursiveForecaster` fits a recursive lag forecaster and then trains
horizon-specific correction models on recursive residuals. This keeps the
recursive path available for smooth multi-step behavior while letting native
boosters learn systematic horizon-specific bias.

```rust
use cartoboost_core::forecasting::RectifiedRecursiveForecaster;

let mut forecaster = RectifiedRecursiveForecaster::new(
    LagFeatureConfig::default(),
    BoosterConfig::default(),
)?;
forecaster.fit_horizon(&pickup_demand_frame, 3)?;
let forecasts = forecaster.predict(3)?;
```

The correction target is `actual - recursive_prediction` for the same horizon.
The final prediction is the recursive prediction plus the correction model
output.

## Intermittent Demand Experts

The forecasting module also exposes deterministic intermittent-demand experts:

| Function | Use |
| --- | --- |
| `croston_forecast` | Sparse non-zero pickup or dispatch counts. |
| `sba_forecast` | Croston with Syntetos-Boylan bias adjustment. |
| `tsb_forecast` | Separate smoothing for occurrence probability and non-zero size. |
| `adida_forecast` | Aggregate-disaggregate baseline for sparse demand. |

All intermittent helpers require a positive horizon, finite non-negative
observations, smoothing parameters in `(0, 1]`, and at least one non-zero
observation.
