//! Rust-native forecasting primitives.
//!
//! The Python forecasting API should wrap these types instead of implementing
//! forecasting behavior itself.

mod artifacts;
mod backtesting;
mod config;
mod ensemble;
mod frequency;
mod global;
mod lag_features;
mod local;
mod metrics;
mod registry;
mod result;
mod schema;
pub(crate) mod splitters;
mod traits;

pub use artifacts::{
    ForecastArtifact, ForecastArtifactManifest, DEFAULT_FORECAST_FILE, DEFAULT_MANIFEST_FILE,
    FORECAST_ARTIFACT_SCHEMA_VERSION,
};
pub use backtesting::{BacktestFoldResult, BacktestResult, RollingOriginBacktester};
pub use config::{ForecastModelConfig, ForecastingConfig};
pub use ensemble::WeightedEnsembleForecaster;
pub use frequency::{parse_forecast_timestamp, ForecastFrequency};
pub use global::{CartoBoostLagForecaster, GlobalForecastTargetMode};
pub use lag_features::{CalendarFeature, LagFeatureBuilder, LagFeatureConfig, LagFeatureRow};
pub use local::{
    ArimaForecaster, ArimaValidationScore, AutoARIMAForecaster, ETSForecaster, KalmanForecaster,
    KrigingForecaster, NaiveForecaster, OptimizedThetaForecaster, SeasonalNaiveForecaster,
    ThetaForecaster, ThetaSeasonality,
};
pub use metrics::{
    evaluate_forecast, evaluate_forecast_with_training, ForecastActual, ForecastMetricSet,
};
pub use registry::{ForecastModelSpec, ForecastRegistry, RegisterMode, RegisteredForecastModel};
pub use result::{ForecastPrediction, ForecastResult};
pub use schema::{ForecastFrame, ForecastFrameMetadata, ForecastRow, SINGLE_SERIES_ID};
pub use splitters::{ForecastFold, ForecastFoldMetadata, ForecastWindow, RollingOriginSplitter};
pub use traits::Forecaster;
