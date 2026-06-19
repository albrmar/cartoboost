//! Rust-native forecasting primitives.
//!
//! The Python forecasting API should wrap these types instead of implementing
//! forecasting behavior itself.

mod artifacts;
mod autostats;
mod backtesting;
mod classical_bank;
mod config;
mod conformal;
mod decomposition;
mod diagnostics;
mod direct;
mod ensemble;
mod features;
mod frequency;
mod gating;
mod global;
mod hierarchy;
mod horizon;
mod intermittent;
mod lag_features;
mod local;
mod metrics;
mod mstl;
mod probabilistic;
mod quantiles;
mod rank_probability;
mod reconciliation;
mod registry;
mod result;
mod schema;
pub(crate) mod splitters;
mod stl;
mod temporal_hierarchy;
mod traits;

pub use artifacts::{
    ForecastArtifact, ForecastArtifactManifest, DEFAULT_FORECAST_FILE, DEFAULT_MANIFEST_FILE,
    FORECAST_ARTIFACT_SCHEMA_VERSION,
};
pub use autostats::AutoStatsBank;
pub use backtesting::{BacktestFoldResult, BacktestResult, RollingOriginBacktester};
pub use classical_bank::{ClassicalExpert, ClassicalExpertBank, ClassicalExpertScore};
pub use config::{ForecastModelConfig, ForecastingConfig};
pub use conformal::{ConformalCalibrator, ConformalInterval};
pub use decomposition::{MSTLCartoBoostForecaster, STLCartoBoostForecaster};
pub use diagnostics::{ForecastDiagnostics, SeriesDiagnostics};
pub use direct::{
    CartoBoostDirectForecaster, DirectForecastStrategy, RectifiedRecursiveForecaster,
};
pub use ensemble::{ForecastEnsemble, GatedEnsembleForecaster, WeightedEnsembleForecaster};
pub use features::{DirectFeatureMatrix, ForecastFeatureFactory};
pub use frequency::{parse_forecast_timestamp, ForecastFrequency};
pub use gating::{ExpertScore, RuleBasedGating, ValidationScoreTable};
pub use global::{CartoBoostLagForecaster, GlobalForecastTargetMode};
pub use hierarchy::{HierarchyNode, HierarchySpec};
pub use horizon::{ForecastOutput, ForecastRequest, ForecastStrategy, QuantileForecastOutput};
pub use intermittent::{adida_forecast, croston_forecast, sba_forecast, tsb_forecast};
pub use lag_features::{CalendarFeature, LagFeatureBuilder, LagFeatureConfig, LagFeatureRow};
pub use local::{
    ArimaForecaster, ArimaValidationScore, AutoARIMAForecaster, AutoKalmanForecaster,
    AutoLocalLevelKalmanForecaster, ETSForecaster, KalmanForecaster, KalmanParameterSet,
    KalmanValidationScore, KrigingForecaster, LocalLevelKalmanForecaster,
    LocalLevelKalmanParameterSet, LocalLevelKalmanValidationScore, NaiveForecaster,
    OptimizedThetaForecaster, SeasonalNaiveForecaster, ThetaForecaster, ThetaSeasonality,
};
pub use metrics::{
    evaluate_forecast, evaluate_forecast_with_training, ForecastActual, ForecastMetricSet,
};
pub use mstl::MSTLDecomposition;
pub use probabilistic::{ProbabilisticDirectForecaster, ProbabilisticForecaster};
pub use quantiles::{pinball_loss, repair_non_crossing_quantiles, QuantileForecast};
pub use rank_probability::{rank_probability_score, RankProbabilityForecast};
pub use reconciliation::{Reconciler, ReconciliationMethod};
pub use registry::{ForecastModelSpec, ForecastRegistry, RegisterMode, RegisteredForecastModel};
pub use result::{ForecastPrediction, ForecastResult};
pub use schema::{ForecastFrame, ForecastFrameMetadata, ForecastRow, SINGLE_SERIES_ID};
pub use splitters::{ForecastFold, ForecastFoldMetadata, ForecastWindow, RollingOriginSplitter};
pub use stl::STLDecomposition;
pub use temporal_hierarchy::{TemporalAggregation, TemporalHierarchy};
pub use traits::Forecaster;
