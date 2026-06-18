//! Rust-native forecasting primitives.
//!
//! The Python forecasting API should wrap these types instead of implementing
//! forecasting behavior itself.

mod frequency;
mod local;
mod metrics;
mod result;
mod schema;
mod traits;

pub use frequency::ForecastFrequency;
pub use local::{NaiveForecaster, SeasonalNaiveForecaster};
pub use metrics::{evaluate_forecast, ForecastActual, ForecastMetricSet};
pub use result::{ForecastPrediction, ForecastResult};
pub use schema::{ForecastFrame, ForecastRow, SINGLE_SERIES_ID};
pub use traits::Forecaster;
