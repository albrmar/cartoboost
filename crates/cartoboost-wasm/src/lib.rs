use cartoboost_core::booster::BoosterConfig;
use cartoboost_core::data::{Dataset, FeatureKind, FeatureSchema, SparseSetColumn};
use cartoboost_core::forecasting::{
    AutoARIMAForecaster, AutoETSForecaster, AutoForecastConfig, AutoForecastModel,
    AutoKalmanForecaster, AutoLocalLevelKalmanForecaster, AutoStatsBank, CalendarFeature,
    CartoBoostDirectForecaster, CartoBoostLagForecaster, ClassicalExpertBank, ETSForecaster,
    ForecastFrame, ForecastFrameMetadata, ForecastFrequency, ForecastResult, Forecaster,
    IntermittentDemandConfig, IntermittentDemandForecaster, KalmanForecaster, KrigingForecaster,
    LagFeatureConfig, LagPlusConfig, LagPlusForecaster, LocalLevelKalmanForecaster,
    LocalStandardScaledForecaster, Log1pForecaster, MSTLCartoBoostForecaster, NaiveForecaster,
    OptimizedThetaForecaster, RectifiedRecursiveForecaster, ReferencePathConfig, ReferenceSignal,
    STLCartoBoostForecaster, SeasonalNaiveForecaster, SeasonalWindowAverageForecaster,
    SequenceCandidate, SequenceCandidateEnsemble, SequenceCandidatePrediction, SequenceFrame,
    SequenceGroupPrediction, SequenceOofCandidateRow, SequenceOofFold, SequenceSeries,
    SequenceStateSpaceConfig, ThetaForecaster, ThetaSeasonality, WindowAverageForecaster,
};
use cartoboost_core::loss::{HuberLossConfig, LogL2LossConfig, LossConfig, QuantileLossConfig};
use cartoboost_core::tree::{Node, Split, SplitterKind};
use cartoboost_core::Booster;
use cartoboost_core::{CartoBoostError, Result};
use cartoboost_neural::{
    ArtifactFallbackKind, GraphSageConfig, GraphSageRegressor, HeteroGraphSageConfig,
    HeteroGraphSageRegressor, HinSageConfig, HinSageRegressor, NeuralEmbeddingRegressor,
    Node2VecConfig, Node2VecRegressor, StandaloneBoosterConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use wasm_bindgen::prelude::*;

type BrowserNeuralPipelineOutput = (Vec<f64>, Vec<String>, Vec<cartoboost_core::Tree>, Value);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastRequest {
    rows: Vec<BrowserForecastRow>,
    frequency: String,
    horizon: usize,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    options: BrowserForecastOptions,
    #[serde(default)]
    metadata: BrowserForecastMetadata,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastRow {
    #[serde(default)]
    series_id: Option<String>,
    timestamp: String,
    target: f64,
    #[serde(default)]
    covariates: BTreeMap<String, f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastOptions {
    season_length: Option<usize>,
    theta: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
    damping_phi: Option<f64>,
    theta_grid: Option<Vec<f64>>,
    alpha_grid: Option<Vec<f64>>,
    theta_seasonality: Option<String>,
    max_p: Option<usize>,
    max_d: Option<usize>,
    max_q: Option<usize>,
    level_process_variance: Option<f64>,
    trend_process_variance: Option<f64>,
    observation_variance: Option<f64>,
    window_size: Option<usize>,
    window_count: Option<usize>,
    validation_window: Option<usize>,
    max_direct_horizon: Option<usize>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    lags: Option<Vec<usize>>,
    rolling_mean_windows: Option<Vec<usize>>,
    rolling_std_windows: Option<Vec<usize>>,
    rolling_min_windows: Option<Vec<usize>>,
    rolling_max_windows: Option<Vec<usize>>,
    difference_lags: Option<Vec<usize>>,
    rolling_trend_windows: Option<Vec<usize>>,
    calendar_features: Option<Vec<String>>,
    mstl_season_lengths: Option<Vec<usize>>,
    coordinate_x: Option<String>,
    coordinate_y: Option<String>,
    kriging_range: Option<f64>,
    kriging_nugget: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastMetadata {
    timestamp_col: Option<String>,
    target_col: Option<String>,
    series_id_col: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionRequest {
    rows: Vec<BrowserRegressionRow>,
    feature_names: Vec<String>,
    #[serde(default)]
    sparse_feature_names: Vec<String>,
    #[serde(default)]
    options: BrowserRegressionOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionRow {
    features: Vec<f64>,
    #[serde(default)]
    sparse_sets: Vec<Vec<u64>>,
    target: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionOptions {
    #[serde(default = "default_holdout_fraction")]
    holdout_fraction: f64,
    splitter_mode: Option<String>,
    #[serde(default)]
    feature_kinds: BTreeMap<String, String>,
    #[serde(default)]
    periodic_periods: BTreeMap<String, u32>,
    loss: Option<String>,
    quantile_alpha: Option<f64>,
    huber_delta: Option<f64>,
    log_offset: Option<f64>,
    interval_lower_alpha: Option<f64>,
    interval_upper_alpha: Option<f64>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
}

impl Default for BrowserRegressionOptions {
    fn default() -> Self {
        Self {
            holdout_fraction: default_holdout_fraction(),
            splitter_mode: None,
            feature_kinds: BTreeMap::new(),
            periodic_periods: BTreeMap::new(),
            loss: None,
            quantile_alpha: None,
            huber_delta: None,
            log_offset: None,
            interval_lower_alpha: None,
            interval_upper_alpha: None,
            n_estimators: None,
            learning_rate: None,
            max_depth: None,
            min_samples_leaf: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralRequest {
    rows: Vec<BrowserNeuralRow>,
    dense_feature_names: Vec<String>,
    #[serde(default)]
    node_features: Vec<Vec<f32>>,
    #[serde(default)]
    node_types: Vec<usize>,
    #[serde(default)]
    edge_type_triples: Vec<(usize, usize, usize)>,
    #[serde(default = "default_neural_pipeline")]
    pipeline: String,
    #[serde(default)]
    options: BrowserNeuralOptions,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSequenceRequest {
    operation: String,
    series: Option<SequenceSeries>,
    frame: Option<SequenceFrame>,
    reference: Option<ReferenceSignal>,
    state_space_config: Option<SequenceStateSpaceConfig>,
    reference_path_config: Option<ReferencePathConfig>,
    candidates: Option<Vec<SequenceCandidate>>,
    weights: Option<BTreeMap<String, f64>>,
    actuals: Option<Vec<SequenceCandidatePrediction>>,
    oof_fold: Option<SequenceOofFold>,
    oof_rows: Option<Vec<SequenceOofCandidateRow>>,
    group_predictions: Option<Vec<SequenceGroupPrediction>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralRow {
    id: Option<u64>,
    source: Option<usize>,
    target_node: Option<usize>,
    edge_weight: Option<f32>,
    edge_type: Option<usize>,
    dense: Vec<f64>,
    target: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralOptions {
    #[serde(default = "default_holdout_fraction")]
    holdout_fraction: f64,
    embedding_dim: Option<usize>,
    random_state: Option<u64>,
    support_prior_strength: Option<f64>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    node2vec_walk_length: Option<usize>,
    node2vec_walks_per_node: Option<usize>,
    node2vec_window_size: Option<usize>,
    node2vec_epochs: Option<usize>,
    node2vec_learning_rate: Option<f32>,
    node2vec_p: Option<f32>,
    node2vec_q: Option<f32>,
    node2vec_seed: Option<u64>,
    graph_sage_epochs: Option<usize>,
    graph_sage_learning_rate: Option<f32>,
    graph_sage_negative_samples: Option<usize>,
    graph_sage_seed: Option<u64>,
}

impl Default for BrowserNeuralOptions {
    fn default() -> Self {
        Self {
            holdout_fraction: default_holdout_fraction(),
            embedding_dim: None,
            random_state: None,
            support_prior_strength: None,
            n_estimators: None,
            learning_rate: None,
            max_depth: None,
            min_samples_leaf: None,
            node2vec_walk_length: None,
            node2vec_walks_per_node: None,
            node2vec_window_size: None,
            node2vec_epochs: None,
            node2vec_learning_rate: None,
            node2vec_p: None,
            node2vec_q: None,
            node2vec_seed: None,
            graph_sage_epochs: None,
            graph_sage_learning_rate: None,
            graph_sage_negative_samples: None,
            graph_sage_seed: None,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserForecastResponse {
    metadata: Value,
    forecast: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionResponse {
    metadata: Value,
    metrics: BrowserRegressionMetrics,
    predictions: Vec<BrowserRegressionPrediction>,
    feature_importance: Vec<BrowserFeatureImportance>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserNeuralResponse {
    metadata: Value,
    metrics: BrowserRegressionMetrics,
    predictions: Vec<BrowserRegressionPrediction>,
    feature_importance: Vec<BrowserFeatureImportance>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionMetrics {
    rmse: f64,
    mae: f64,
    r2: f64,
    train_rows: usize,
    holdout_rows: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserRegressionPrediction {
    row_index: usize,
    actual: f64,
    prediction: f64,
    lower_prediction: Option<f64>,
    upper_prediction: Option<f64>,
    residual: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserFeatureImportance {
    feature: String,
    split_count: usize,
}

#[derive(Clone, Debug, Serialize)]
struct BrowserForecastModel {
    name: &'static str,
    label: &'static str,
    pipeline: &'static str,
}

#[wasm_bindgen(js_name = runForecast)]
pub fn run_forecast(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserForecastRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid forecast request: {error}")))?;
    let response =
        run_forecast_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode forecast response: {error}")))
}

#[wasm_bindgen(js_name = runRegressionModel)]
pub fn run_regression_model(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserRegressionRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid regression request: {error}")))?;
    let response =
        run_regression_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response.serialize(&serializer).map_err(|error| {
        JsValue::from_str(&format!("could not encode regression response: {error}"))
    })
}

#[wasm_bindgen(js_name = runNeuralModel)]
pub fn run_neural_model(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserNeuralRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid neural request: {error}")))?;
    let response =
        run_neural_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode neural response: {error}")))
}

#[wasm_bindgen(js_name = runSequence)]
pub fn run_sequence(request: JsValue) -> std::result::Result<JsValue, JsValue> {
    let request: BrowserSequenceRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid sequence request: {error}")))?;
    let response =
        run_sequence_request(request).map_err(|error| JsValue::from_str(&error.to_string()))?;
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode sequence response: {error}")))
}

#[wasm_bindgen(js_name = availableForecastModels)]
pub fn available_forecast_models() -> std::result::Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    forecast_model_registry()
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("could not encode model registry: {error}")))
}

fn forecast_model_registry() -> Vec<BrowserForecastModel> {
    vec![
        BrowserForecastModel {
            name: "auto_forecast",
            label: "CartoBoost AutoForecast",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "cartoboost_lag",
            label: "CartoBoost Lag",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "cartoboost_direct",
            label: "CartoBoost Direct",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "rectified_recursive",
            label: "Rectified Recursive",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "lag_plus",
            label: "Lag Plus",
            pipeline: "global",
        },
        BrowserForecastModel {
            name: "scaled_cartoboost_lag",
            label: "Scaled CartoBoost Lag",
            pipeline: "transform",
        },
        BrowserForecastModel {
            name: "log1p_cartoboost_lag",
            label: "Log1p CartoBoost Lag",
            pipeline: "transform",
        },
        BrowserForecastModel {
            name: "classical_expert_bank",
            label: "Classical Expert Bank",
            pipeline: "selection",
        },
        BrowserForecastModel {
            name: "autostats_bank",
            label: "AutoStats Bank",
            pipeline: "selection",
        },
        BrowserForecastModel {
            name: "intermittent_demand",
            label: "Intermittent Demand",
            pipeline: "demand",
        },
        BrowserForecastModel {
            name: "stl_cartoboost",
            label: "STL + ARIMA",
            pipeline: "decomposition",
        },
        BrowserForecastModel {
            name: "mstl_cartoboost",
            label: "MSTL + ARIMA",
            pipeline: "decomposition",
        },
        BrowserForecastModel {
            name: "seasonal_naive",
            label: "Seasonal Naive",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "window_average",
            label: "Window Average",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "seasonal_window_average",
            label: "Seasonal Window Average",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "theta",
            label: "Theta",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_ets",
            label: "Auto ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "ets",
            label: "ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "seasonal_ets",
            label: "Seasonal ETS",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_arima",
            label: "Auto ARIMA",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "arima",
            label: "ARIMA",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "kalman",
            label: "Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "local_level_kalman",
            label: "Local Level Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_kalman",
            label: "Auto Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "auto_local_level_kalman",
            label: "Auto Local Level Kalman",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "kriging",
            label: "Kriging",
            pipeline: "spatial",
        },
        BrowserForecastModel {
            name: "optimized_theta",
            label: "Optimized Theta",
            pipeline: "local",
        },
        BrowserForecastModel {
            name: "naive",
            label: "Naive",
            pipeline: "local",
        },
    ]
}

fn run_sequence_request(request: BrowserSequenceRequest) -> Result<Value> {
    match request.operation.trim().to_ascii_lowercase().as_str() {
        "validate" | "validate_frame" => {
            let frame = request.frame.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence validate requires frame".to_string())
            })?;
            frame.validate()?;
            Ok(json!({ "ok": true }))
        }
        "ekf" | "forward_ekf" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::forward_ekf(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "ukf" | "ukf_reference" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::ukf_reference(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "rts" | "rts_smoother" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::rts_smoother(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "continuation" | "missing_target_continuation" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.state_space_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::missing_target_continuation(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "viterbi" | "reference_path_viterbi" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.reference_path_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::reference_path_viterbi(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "posterior_mean" | "reference_path_posterior_mean" => {
            let series = sequence_series_arg(&request)?;
            let reference = sequence_reference_arg(&request)?;
            let config = request.reference_path_config.unwrap_or_default();
            serde_json::to_value(cartoboost_core::forecasting::reference_path_posterior_mean(
                &series, &reference, config,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "blend_fixed" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let weights = request.weights.ok_or_else(|| {
                CartoBoostError::InvalidInput("fixed sequence blend requires weights".to_string())
            })?;
            let ensemble = SequenceCandidateEnsemble::fixed(weights)?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "blend_validation" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let actuals = request.actuals.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "validation sequence blend requires actuals".to_string(),
                )
            })?;
            let ensemble = SequenceCandidateEnsemble::validation_derived(&candidates, &actuals)?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "blend_constrained" => {
            let candidates = request.candidates.ok_or_else(|| {
                CartoBoostError::InvalidInput("sequence blend requires candidates".to_string())
            })?;
            let actuals = request.actuals.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "constrained sequence blend requires actuals".to_string(),
                )
            })?;
            let ensemble = SequenceCandidateEnsemble::constrained_nonnegative_linear_blend(
                &candidates,
                &actuals,
            )?;
            Ok(json!({
                "weights": ensemble.weights,
                "predictions": ensemble.predict(&candidates)?,
            }))
        }
        "validate_oof" | "validate_oof_meta_training" => {
            let rows = request.oof_rows.ok_or_else(|| {
                CartoBoostError::InvalidInput("OOF validation requires oofRows".to_string())
            })?;
            cartoboost_core::forecasting::validate_oof_meta_training(&rows)?;
            Ok(json!({ "ok": true }))
        }
        "generate_oof" | "generate_group_oof_candidate_rows" => {
            let fold = request.oof_fold.ok_or_else(|| {
                CartoBoostError::InvalidInput("OOF generation requires oofFold".to_string())
            })?;
            serde_json::to_value(
                cartoboost_core::forecasting::generate_group_oof_candidate_rows(&fold)?,
            )
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        "group_metrics" | "per_group_error_summary" => {
            let rows = request.group_predictions.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "group metric summary requires groupPredictions".to_string(),
                )
            })?;
            serde_json::to_value(cartoboost_core::forecasting::per_group_error_summary(
                &rows,
            )?)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown sequence operation {other:?}"
        ))),
    }
}

fn sequence_series_arg(request: &BrowserSequenceRequest) -> Result<SequenceSeries> {
    request.series.clone().ok_or_else(|| {
        CartoBoostError::InvalidInput("sequence operation requires series".to_string())
    })
}

fn sequence_reference_arg(request: &BrowserSequenceRequest) -> Result<ReferenceSignal> {
    request.reference.clone().ok_or_else(|| {
        CartoBoostError::InvalidInput("sequence operation requires reference".to_string())
    })
}

fn run_forecast_request(request: BrowserForecastRequest) -> Result<BrowserForecastResponse> {
    if request.horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    if request.rows.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast request must include at least one row".to_string(),
        ));
    }

    let frequency = ForecastFrequency::parse(&request.frequency)?;
    let metadata = ForecastFrameMetadata {
        timestamp_col: request.metadata.timestamp_col,
        target_col: request.metadata.target_col,
        series_id_col: request.metadata.series_id_col,
        static_covariates: Vec::new(),
        known_future_covariates: Vec::new(),
        historical_covariates: Vec::new(),
    };
    let rows = request
        .rows
        .into_iter()
        .map(|row| {
            cartoboost_core::forecasting::ForecastRow::from_timestamp_str_with_covariates(
                row.series_id.unwrap_or_default(),
                &row.timestamp,
                row.target,
                row.covariates,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let frame = ForecastFrame::with_metadata(rows, frequency, metadata)?;
    let mut forecaster =
        build_forecaster(&request.model, &request.options, &frame, request.horizon)?;
    forecaster.fit(&frame)?;
    let forecast = forecaster.predict(request.horizon)?;
    Ok(BrowserForecastResponse {
        metadata: json!({
            "model": forecaster.model_name(),
            "input": frame.metadata_value(),
            "modelMetadata": forecaster.metadata(),
        }),
        forecast: forecast.to_json_value(),
    })
}

fn run_regression_request(request: BrowserRegressionRequest) -> Result<BrowserRegressionResponse> {
    if request.rows.len() < 4 {
        return Err(CartoBoostError::InvalidInput(
            "regression modeling requires at least four rows".to_string(),
        ));
    }
    if request.feature_names.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "regression modeling requires at least one feature".to_string(),
        ));
    }
    if !request.options.holdout_fraction.is_finite()
        || request.options.holdout_fraction <= 0.0
        || request.options.holdout_fraction >= 0.8
    {
        return Err(CartoBoostError::InvalidInput(
            "holdout_fraction must be finite and between 0 and 0.8".to_string(),
        ));
    }
    let feature_count = request.feature_names.len();
    let sparse_feature_count = request.sparse_feature_names.len();
    let mut features = Vec::with_capacity(request.rows.len());
    let mut sparse_rows = Vec::with_capacity(request.rows.len());
    let mut targets = Vec::with_capacity(request.rows.len());
    for row in request.rows {
        if row.features.len() != feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "feature row has {} columns but feature_names has {feature_count}",
                row.features.len()
            )));
        }
        if row.sparse_sets.len() != sparse_feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "sparse feature row has {} columns but sparse_feature_names has {sparse_feature_count}",
                row.sparse_sets.len()
            )));
        }
        if row.features.iter().any(|value| !value.is_finite()) || !row.target.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "regression features and targets must be finite".to_string(),
            ));
        }
        features.push(row.features);
        sparse_rows.push(row.sparse_sets);
        targets.push(row.target);
    }

    let requested_holdout =
        ((features.len() as f64) * request.options.holdout_fraction).round() as usize;
    let holdout_rows = requested_holdout.clamp(1, features.len().saturating_sub(2));
    let train_rows = features.len() - holdout_rows;
    let schema = regression_feature_schema(
        &request.feature_names,
        &request.sparse_feature_names,
        &request.options,
    )?;
    let train_x = Dataset::mixed(
        features[..train_rows].to_vec(),
        sparse_columns_from_rows(&sparse_rows[..train_rows], sparse_feature_count),
        Some(schema.clone()),
    )?;
    let holdout_x = Dataset::mixed(
        features[train_rows..].to_vec(),
        sparse_columns_from_rows(&sparse_rows[train_rows..], sparse_feature_count),
        Some(schema),
    )?;
    let train_y = &targets[..train_rows];
    let holdout_y = &targets[train_rows..];

    let model =
        Booster::new(regression_booster_config(&request.options)?).fit(&train_x, train_y, None)?;
    let predictions = model.try_predict(&holdout_x)?;
    let interval_predictions =
        regression_interval_predictions(&request.options, &train_x, train_y, &holdout_x)?;
    let metrics = regression_metrics(holdout_y, &predictions, train_rows, holdout_rows)?;
    let prediction_rows = predictions
        .iter()
        .zip(holdout_y.iter())
        .enumerate()
        .map(
            |(offset, (prediction, actual))| BrowserRegressionPrediction {
                row_index: train_rows + offset,
                actual: *actual,
                prediction: *prediction,
                lower_prediction: interval_predictions
                    .as_ref()
                    .map(|(lower, _)| lower[offset]),
                upper_prediction: interval_predictions
                    .as_ref()
                    .map(|(_, upper)| upper[offset]),
                residual: actual - prediction,
            },
        )
        .collect::<Vec<_>>();
    let feature_importance = feature_importance(
        &model.trees,
        &request.feature_names,
        &request.sparse_feature_names,
    );

    Ok(BrowserRegressionResponse {
        metadata: json!({
            "model": "cartoboost_regressor",
            "featureNames": request.feature_names,
            "sparseFeatureNames": request.sparse_feature_names,
            "trainingConfig": model.training_config,
            "splitterMode": request.options.splitter_mode.as_deref().unwrap_or("auto"),
            "loss": regression_loss_label(&request.options),
            "intervalLowerAlpha": request.options.interval_lower_alpha,
            "intervalUpperAlpha": request.options.interval_upper_alpha,
            "treeCount": model.trees.len(),
        }),
        metrics,
        predictions: prediction_rows,
        feature_importance,
    })
}

fn run_neural_request(request: BrowserNeuralRequest) -> Result<BrowserNeuralResponse> {
    if request.rows.len() < 4 {
        return Err(CartoBoostError::InvalidInput(
            "neural modeling requires at least four rows".to_string(),
        ));
    }
    if !request.options.holdout_fraction.is_finite()
        || request.options.holdout_fraction <= 0.0
        || request.options.holdout_fraction >= 0.8
    {
        return Err(CartoBoostError::InvalidInput(
            "holdout_fraction must be finite and between 0 and 0.8".to_string(),
        ));
    }
    let dense_width = request.dense_feature_names.len();
    let mut dense = Vec::with_capacity(request.rows.len());
    let mut targets = Vec::with_capacity(request.rows.len());
    for row in &request.rows {
        if row.dense.len() != dense_width {
            return Err(CartoBoostError::InvalidInput(format!(
                "neural dense row has {} columns but dense_feature_names has {dense_width}",
                row.dense.len()
            )));
        }
        if row.dense.iter().any(|value| !value.is_finite()) || !row.target.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "neural dense features and targets must be finite".to_string(),
            ));
        }
        dense.push(row.dense.clone());
        targets.push(row.target);
    }

    let requested_holdout =
        ((dense.len() as f64) * request.options.holdout_fraction).round() as usize;
    let holdout_rows = requested_holdout.clamp(1, dense.len().saturating_sub(2));
    let train_rows = dense.len() - holdout_rows;
    let pipeline = request.pipeline.trim().to_ascii_lowercase();
    let (predictions, feature_names, trees, metadata) = match pipeline.as_str() {
        "" | "embedding" | "embedding_table" | "neural_embedding" => {
            run_embedding_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "node2vec" | "node2vec_graph" | "graph_node2vec" => {
            run_node2vec_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "graphsage" | "graph_sage" | "graphsage_graph" => {
            run_graphsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "hetero_graphsage" | "heterographsage" | "typed_graphsage" => {
            run_hetero_graphsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "hinsage" | "hin_sage" | "typed_hinsage" => {
            run_hinsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        other => {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported browser neural pipeline {other:?}"
            )));
        }
    };

    let holdout_y = &targets[train_rows..];
    let metrics = regression_metrics(holdout_y, &predictions, train_rows, holdout_rows)?;
    let prediction_rows = predictions
        .iter()
        .zip(holdout_y.iter())
        .enumerate()
        .map(
            |(offset, (prediction, actual))| BrowserRegressionPrediction {
                row_index: train_rows + offset,
                actual: *actual,
                prediction: *prediction,
                lower_prediction: None,
                upper_prediction: None,
                residual: actual - prediction,
            },
        )
        .collect::<Vec<_>>();
    let feature_importance = feature_importance(&trees, &feature_names, &[]);

    Ok(BrowserNeuralResponse {
        metadata: json!({
            "model": metadata["model"].as_str().unwrap_or("cartoboost_neural"),
            "pipeline": pipeline,
            "denseFeatureNames": request.dense_feature_names,
            "treeCount": trees.len(),
            "details": metadata,
        }),
        metrics,
        predictions: prediction_rows,
        feature_importance,
    })
}

fn run_embedding_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let ids = request
        .rows
        .iter()
        .map(|row| {
            row.id.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "embedding neural pipeline requires an id column".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut model = NeuralEmbeddingRegressor::new(
        request.options.embedding_dim.unwrap_or(8),
        ArtifactFallbackKind::GlobalMeanVector,
        request.options.random_state,
        request.options.support_prior_strength.unwrap_or(1.0),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &ids[..train_rows],
            &targets[..train_rows],
            Some(&dense[..train_rows]),
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(&ids[train_rows..], Some(&dense[train_rows..]))
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "embedding",
        artifact.dim,
        &request.dense_feature_names,
        None,
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "neural_embedding_regressor",
            "embeddingDim": artifact.dim,
            "embeddingRows": artifact.table.rows.len(),
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_node2vec_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let sources = request
        .rows
        .iter()
        .map(|row| {
            row.source.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "Node2Vec neural pipeline requires a source column".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let target_nodes = request
        .rows
        .iter()
        .map(|row| row.target_node)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "Node2Vec neural pipeline requires a target node column".to_string(),
            )
        })?;
    let edges = sources
        .iter()
        .zip(target_nodes.iter())
        .map(|(source, target)| (*source, *target))
        .collect::<Vec<_>>();
    let edge_weights = request
        .rows
        .iter()
        .map(|row| row.edge_weight.unwrap_or(1.0))
        .collect::<Vec<_>>();
    let node_count = edges
        .iter()
        .flat_map(|(source, target)| [*source, *target])
        .max()
        .map(|max_node| max_node + 1)
        .unwrap_or(0);
    let mut model = Node2VecRegressor::new(
        node2vec_config(&request.options),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            node_count,
            &edges,
            Some(&edge_weights),
            &sources[..train_rows],
            Some(&target_nodes[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &sources[train_rows..],
            Some(&target_nodes[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "node2vec",
        artifact.encoder.output_dim,
        &request.dense_feature_names,
        Some("target_node2vec"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "node2vec_regressor",
            "mode": artifact.mode,
            "embeddingDim": artifact.encoder.output_dim,
            "nodeCount": artifact.encoder.node_count,
            "edgeCount": edges.len(),
            "lossCurve": artifact.encoder.loss_curve,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_graphsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "GraphSAGE")?;
    let config = graph_sage_config(&request.options);
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let mut model = GraphSageRegressor::new(
        config,
        graph.input_dim,
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "graphsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_graphsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "graphsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.edges.len(),
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_hetero_graphsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "HeteroGraphSAGE")?;
    let config = hetero_graph_sage_config(&request.options);
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let relation_count = graph
        .typed_edges
        .iter()
        .map(|(_, _, relation)| *relation)
        .max()
        .map(|relation| relation + 1)
        .unwrap_or(1);
    let mut model = HeteroGraphSageRegressor::new(
        config,
        graph.input_dim,
        relation_count,
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.typed_edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "hetero_graphsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_hetero_graphsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "hetero_graphsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.typed_edges.len(),
            "relationCount": relation_count,
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_hinsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "HinSAGE")?;
    let config = hin_sage_config(&request.options);
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let node_type_count = graph
        .node_types
        .iter()
        .max()
        .map(|node_type| node_type + 1)
        .unwrap_or(1);
    let edge_type_triples = if request.edge_type_triples.is_empty() {
        vec![(0, 0, 0)]
    } else {
        request.edge_type_triples.clone()
    };
    let mut model = HinSageRegressor::new(
        config,
        graph.input_dim,
        node_type_count,
        edge_type_triples.clone(),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.node_types,
            &graph.typed_edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "hinsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_hinsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "hinsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.typed_edges.len(),
            "nodeTypeCount": node_type_count,
            "edgeTypeTriples": edge_type_triples,
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

struct BrowserGraphInputs {
    node_features: Vec<Vec<f32>>,
    node_types: Vec<usize>,
    sources: Vec<usize>,
    targets: Vec<usize>,
    edges: Vec<(usize, usize)>,
    typed_edges: Vec<(usize, usize, usize)>,
    input_dim: usize,
}

fn browser_graph_inputs(
    request: &BrowserNeuralRequest,
    pipeline_name: &str,
) -> Result<BrowserGraphInputs> {
    if request.node_features.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} neural pipeline requires inferred node features"
        )));
    }
    let input_dim = request.node_features[0].len();
    if input_dim == 0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} neural pipeline requires at least one node feature"
        )));
    }
    for features in &request.node_features {
        if features.len() != input_dim || features.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "{pipeline_name} node features must be finite and rectangular"
            )));
        }
    }
    let sources = request
        .rows
        .iter()
        .map(|row| {
            row.source.ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "{pipeline_name} neural pipeline requires a source column"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let targets = request
        .rows
        .iter()
        .map(|row| row.target_node)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "{pipeline_name} neural pipeline requires a target node column"
            ))
        })?;
    let node_count = request.node_features.len();
    if sources
        .iter()
        .chain(targets.iter())
        .any(|node| *node >= node_count)
    {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} graph rows reference node ids outside node_features"
        )));
    }
    let edges = sources
        .iter()
        .zip(targets.iter())
        .map(|(source, target)| (*source, *target))
        .collect::<Vec<_>>();
    let typed_edges = request
        .rows
        .iter()
        .zip(edges.iter())
        .map(|(row, (source, target))| (*source, *target, row.edge_type.unwrap_or(0)))
        .collect::<Vec<_>>();
    let node_types = if request.node_types.is_empty() {
        vec![0; node_count]
    } else {
        if request.node_types.len() != node_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "{pipeline_name} node_types length must match node_features"
            )));
        }
        request.node_types.clone()
    };
    Ok(BrowserGraphInputs {
        node_features: request.node_features.clone(),
        node_types,
        sources,
        targets,
        edges,
        typed_edges,
        input_dim,
    })
}

fn embedding_feature_names(
    prefix: &str,
    dim: usize,
    dense_feature_names: &[String],
    secondary_prefix: Option<&str>,
) -> Vec<String> {
    let mut names = (0..dim)
        .map(|idx| format!("{prefix}_{idx}"))
        .collect::<Vec<_>>();
    if let Some(secondary_prefix) = secondary_prefix {
        names.extend((0..dim).map(|idx| format!("{secondary_prefix}_{idx}")));
    }
    names.extend(dense_feature_names.iter().cloned());
    names
}

fn standalone_booster_config(options: &BrowserNeuralOptions) -> StandaloneBoosterConfig {
    StandaloneBoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(80),
        learning_rate: options.learning_rate.unwrap_or(0.07),
        max_depth: options.max_depth.unwrap_or(4),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(2),
        min_gain: 0.0,
    }
}

fn node2vec_config(options: &BrowserNeuralOptions) -> Node2VecConfig {
    let mut config = Node2VecConfig::default();
    if let Some(dim) = options.embedding_dim {
        config.dim = dim;
    }
    if let Some(walk_length) = options.node2vec_walk_length {
        config.walk_length = walk_length;
    }
    if let Some(walks_per_node) = options.node2vec_walks_per_node {
        config.walks_per_node = walks_per_node;
    }
    if let Some(window_size) = options.node2vec_window_size {
        config.window_size = window_size;
    }
    if let Some(epochs) = options.node2vec_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.node2vec_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(p) = options.node2vec_p {
        config.p = p;
    }
    if let Some(q) = options.node2vec_q {
        config.q = q;
    }
    if let Some(seed) = options.node2vec_seed {
        config.seed = seed;
    }
    config
}

fn graph_sage_config(options: &BrowserNeuralOptions) -> GraphSageConfig {
    let mut config = GraphSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        ..GraphSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    config
}

fn hetero_graph_sage_config(options: &BrowserNeuralOptions) -> HeteroGraphSageConfig {
    let mut config = HeteroGraphSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        ..HeteroGraphSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    config
}

fn hin_sage_config(options: &BrowserNeuralOptions) -> HinSageConfig {
    let mut config = HinSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        ..HinSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    config
}

fn graph_sage_dim(hidden_dims: &[usize]) -> usize {
    hidden_dims.last().copied().unwrap_or(8)
}

fn neural_to_core(error: cartoboost_neural::NeuralError) -> CartoBoostError {
    CartoBoostError::InvalidInput(error.to_string())
}

fn sparse_columns_from_rows(
    sparse_rows: &[Vec<Vec<u64>>],
    sparse_feature_count: usize,
) -> Vec<SparseSetColumn> {
    (0..sparse_feature_count)
        .map(|feature_idx| {
            SparseSetColumn::new(
                sparse_rows
                    .iter()
                    .map(|row| row.get(feature_idx).cloned().unwrap_or_default())
                    .collect(),
            )
        })
        .collect()
}

fn regression_booster_config(options: &BrowserRegressionOptions) -> Result<BoosterConfig> {
    Ok(BoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(120),
        learning_rate: options.learning_rate.unwrap_or(0.06),
        max_depth: options.max_depth.unwrap_or(3),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(4),
        splitters: regression_splitters(options),
        loss: regression_loss_config(options)?,
        ..Default::default()
    })
}

fn regression_interval_predictions(
    options: &BrowserRegressionOptions,
    train_x: &Dataset,
    train_y: &[f64],
    holdout_x: &Dataset,
) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
    let Some(lower_alpha) = options.interval_lower_alpha else {
        return Ok(None);
    };
    let Some(upper_alpha) = options.interval_upper_alpha else {
        return Ok(None);
    };
    if !lower_alpha.is_finite()
        || !upper_alpha.is_finite()
        || lower_alpha <= 0.0
        || upper_alpha >= 1.0
        || lower_alpha >= upper_alpha
    {
        return Err(CartoBoostError::InvalidInput(
            "interval alphas must be finite with 0 < lower < upper < 1".to_string(),
        ));
    }
    let lower_model = Booster::new(regression_booster_config_with_loss(
        options,
        LossConfig::Quantile(QuantileLossConfig { alpha: lower_alpha }),
    ))
    .fit(train_x, train_y, None)?;
    let upper_model = Booster::new(regression_booster_config_with_loss(
        options,
        LossConfig::Quantile(QuantileLossConfig { alpha: upper_alpha }),
    ))
    .fit(train_x, train_y, None)?;
    let lower = lower_model.try_predict(holdout_x)?;
    let upper = upper_model.try_predict(holdout_x)?;
    let (lower, upper): (Vec<_>, Vec<_>) = lower
        .into_iter()
        .zip(upper)
        .map(|(left, right)| {
            if left <= right {
                (left, right)
            } else {
                (right, left)
            }
        })
        .unzip();
    Ok(Some((lower, upper)))
}

fn regression_booster_config_with_loss(
    options: &BrowserRegressionOptions,
    loss: LossConfig,
) -> BoosterConfig {
    BoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(120),
        learning_rate: options.learning_rate.unwrap_or(0.06),
        max_depth: options.max_depth.unwrap_or(3),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(4),
        splitters: regression_splitters(options),
        loss,
        ..Default::default()
    }
}

fn regression_loss_label(options: &BrowserRegressionOptions) -> String {
    options
        .loss
        .as_deref()
        .unwrap_or("l2")
        .trim()
        .to_ascii_lowercase()
}

fn regression_loss_config(options: &BrowserRegressionOptions) -> Result<LossConfig> {
    match regression_loss_label(options).as_str() {
        "" | "l2" | "squared_error" => Ok(LossConfig::L2),
        "l1" | "absolute_error" | "median" => Ok(LossConfig::L1),
        "huber" => Ok(LossConfig::Huber(HuberLossConfig {
            delta: options.huber_delta.unwrap_or(1.0),
        })),
        "log_l2" | "logl2" => Ok(LossConfig::LogL2(LogL2LossConfig {
            offset: options.log_offset.unwrap_or(1.0),
        })),
        "quantile" => Ok(LossConfig::Quantile(QuantileLossConfig {
            alpha: options.quantile_alpha.unwrap_or(0.5),
        })),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported browser regression loss {other:?}"
        ))),
    }
}

fn regression_splitters(options: &BrowserRegressionOptions) -> Vec<SplitterKind> {
    match options
        .splitter_mode
        .as_deref()
        .unwrap_or("auto")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "axis" | "dense_axis" => vec![SplitterKind::Axis],
        "spatial" => vec![
            SplitterKind::Axis,
            SplitterKind::Diagonal2D,
            SplitterKind::Gaussian2D,
        ],
        "periodic" => vec![
            SplitterKind::Axis,
            SplitterKind::Periodic {
                period: default_periodic_period(options),
            },
        ],
        "sparse" | "sparse_set" | "sparse_sets" => {
            vec![SplitterKind::Axis, SplitterKind::SparseSet]
        }
        "full" | "toolkit" | "spatial_periodic" => vec![
            SplitterKind::Axis,
            SplitterKind::Diagonal2D,
            SplitterKind::Gaussian2D,
            SplitterKind::Periodic {
                period: default_periodic_period(options),
            },
            SplitterKind::SparseSet,
        ],
        _ => vec![SplitterKind::Auto],
    }
}

fn default_periodic_period(options: &BrowserRegressionOptions) -> f64 {
    options
        .periodic_periods
        .values()
        .next()
        .copied()
        .unwrap_or(24) as f64
}

fn regression_feature_schema(
    feature_names: &[String],
    sparse_feature_names: &[String],
    options: &BrowserRegressionOptions,
) -> Result<FeatureSchema> {
    let mut kinds = feature_names
        .iter()
        .map(|name| {
            let kind = options
                .feature_kinds
                .get(name)
                .map(|value| value.trim().to_ascii_lowercase())
                .unwrap_or_else(|| "numeric".to_string());
            match kind.as_str() {
                "" | "numeric" => Ok(FeatureKind::Numeric),
                "spatial" => Ok(FeatureKind::Spatial),
                "periodic" => {
                    let period = options.periodic_periods.get(name).copied().unwrap_or(24);
                    if period == 0 {
                        return Err(CartoBoostError::InvalidInput(format!(
                            "periodic feature {name:?} must have a positive period"
                        )));
                    }
                    Ok(FeatureKind::Periodic { period })
                }
                other => Err(CartoBoostError::InvalidInput(format!(
                    "unsupported browser regression feature kind {other:?} for {name:?}"
                ))),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    kinds.extend(
        sparse_feature_names
            .iter()
            .map(|_| FeatureKind::SparseSet)
            .collect::<Vec<_>>(),
    );
    let mut names = feature_names.to_vec();
    names.extend(sparse_feature_names.iter().cloned());
    Ok(FeatureSchema { names, kinds })
}

fn regression_metrics(
    actuals: &[f64],
    predictions: &[f64],
    train_rows: usize,
    holdout_rows: usize,
) -> Result<BrowserRegressionMetrics> {
    if actuals.len() != predictions.len() || actuals.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "actual and prediction lengths must match and be non-empty".to_string(),
        ));
    }
    let mut squared_error_sum = 0.0;
    let mut absolute_error_sum = 0.0;
    let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
    let mut total_sum_squares = 0.0;
    for (actual, prediction) in actuals.iter().zip(predictions.iter()) {
        let residual = actual - prediction;
        squared_error_sum += residual * residual;
        absolute_error_sum += residual.abs();
        total_sum_squares += (actual - mean_actual).powi(2);
    }
    let rmse = (squared_error_sum / actuals.len() as f64).sqrt();
    let mae = absolute_error_sum / actuals.len() as f64;
    let r2 = if total_sum_squares <= f64::EPSILON {
        0.0
    } else {
        1.0 - squared_error_sum / total_sum_squares
    };
    Ok(BrowserRegressionMetrics {
        rmse,
        mae,
        r2,
        train_rows,
        holdout_rows,
    })
}

fn feature_importance(
    trees: &[cartoboost_core::Tree],
    feature_names: &[String],
    sparse_feature_names: &[String],
) -> Vec<BrowserFeatureImportance> {
    let dense_feature_count = feature_names.len();
    let mut names = feature_names.to_vec();
    names.extend(sparse_feature_names.iter().cloned());
    let mut counts = vec![0usize; names.len()];
    for tree in trees {
        count_split_features(&tree.root, &mut counts, dense_feature_count);
    }
    let mut importance = names
        .iter()
        .enumerate()
        .map(|(idx, feature)| BrowserFeatureImportance {
            feature: feature.clone(),
            split_count: counts[idx],
        })
        .collect::<Vec<_>>();
    importance.sort_by(|left, right| {
        right
            .split_count
            .cmp(&left.split_count)
            .then_with(|| left.feature.cmp(&right.feature))
    });
    importance
}

fn count_split_features(node: &Node, counts: &mut [usize], dense_feature_count: usize) {
    if let Node::Branch {
        split, left, right, ..
    } = node
    {
        count_split(split, counts, dense_feature_count);
        count_split_features(left, counts, dense_feature_count);
        count_split_features(right, counts, dense_feature_count);
    }
}

fn count_split(split: &Split, counts: &mut [usize], dense_feature_count: usize) {
    match split {
        Split::Axis { feature, .. }
        | Split::PeriodicInterval { feature, .. }
        | Split::SparseSetContainsAny { feature, .. } => increment_feature(*feature, counts),
        Split::Diagonal2D {
            x_feature,
            y_feature,
            ..
        }
        | Split::Gaussian2D {
            x_feature,
            y_feature,
            ..
        } => {
            increment_feature(*x_feature, counts);
            increment_feature(*y_feature, counts);
        }
        Split::SparseListContainsAny { sparse_feature, .. } => {
            increment_feature(dense_feature_count + *sparse_feature, counts);
        }
        Split::Fuzzy { base, .. } => count_split(base, counts, dense_feature_count),
    }
}

fn increment_feature(feature: usize, counts: &mut [usize]) {
    if let Some(count) = counts.get_mut(feature) {
        *count += 1;
    }
}

fn default_holdout_fraction() -> f64 {
    0.2
}

fn default_neural_pipeline() -> String {
    "embedding".to_string()
}

fn build_forecaster(
    model: &str,
    options: &BrowserForecastOptions,
    frame: &ForecastFrame,
    horizon: usize,
) -> Result<Box<dyn Forecaster>> {
    let normalized = model.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "naive" => Ok(Box::new(NaiveForecaster::new())),
        "seasonal_naive" => Ok(Box::new(SeasonalNaiveForecaster::new(
            options.season_length.unwrap_or(7),
        )?)),
        "window_average" => Ok(Box::new(WindowAverageForecaster::new(
            options.window_size.unwrap_or(7),
        )?)),
        "seasonal_window_average" => Ok(Box::new(SeasonalWindowAverageForecaster::new(
            options.season_length.unwrap_or(7),
            options.window_count.unwrap_or(3),
        )?)),
        "theta" => Ok(Box::new(ThetaForecaster::with_seasonality(
            options.theta.unwrap_or(2.0),
            options.alpha.unwrap_or(0.2),
            theta_seasonality(options)?,
        )?)),
        "optimized_theta" => Ok(Box::new(OptimizedThetaForecaster::with_seasonality(
            options
                .theta_grid
                .clone()
                .unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]),
            options
                .alpha_grid
                .clone()
                .unwrap_or_else(|| vec![0.1, 0.2, 0.3, 0.5, 0.8]),
            theta_seasonality(options)?,
        )?)),
        "ets" => Ok(Box::new(ETSForecaster::with_additive_damped_trend(
            options.alpha.unwrap_or(0.3),
            options.beta.unwrap_or(0.1),
            options.gamma,
            None,
            options.damping_phi.unwrap_or(1.0),
        )?)),
        "seasonal_ets" => Ok(Box::new(ETSForecaster::with_additive_damped_trend(
            options.alpha.unwrap_or(0.3),
            options.beta.unwrap_or(0.1),
            Some(options.gamma.unwrap_or(0.1)),
            Some(options.season_length.unwrap_or(7)),
            options.damping_phi.unwrap_or(1.0),
        )?)),
        "auto_ets" => Ok(Box::new(AutoETSForecaster::new(options.season_length)?)),
        "arima" => Ok(Box::new(
            cartoboost_core::forecasting::ArimaForecaster::new(
                options.max_p.unwrap_or(1),
                options.max_d.unwrap_or(1),
                options.max_q.unwrap_or(0),
            )?,
        )),
        "auto_arima" => Ok(Box::new(AutoARIMAForecaster::with_max_order(
            options.max_p.unwrap_or(2),
            options.max_d.unwrap_or(1),
            options.max_q.unwrap_or(1),
        )?)),
        "kalman" => Ok(Box::new(KalmanForecaster::new(
            options.level_process_variance.unwrap_or(0.05),
            options.trend_process_variance.unwrap_or(0.005),
            options.observation_variance.unwrap_or(1.0),
        )?)),
        "local_level_kalman" => Ok(Box::new(LocalLevelKalmanForecaster::new(
            options.level_process_variance.unwrap_or(0.05),
            options.observation_variance.unwrap_or(1.0),
        )?)),
        "auto_kalman" => Ok(Box::new(AutoKalmanForecaster::new()?)),
        "auto_local_level_kalman" => Ok(Box::new(AutoLocalLevelKalmanForecaster::new()?)),
        "kriging" => Ok(Box::new(KrigingForecaster::new(
            coordinates_from_frame(frame, options)?,
            options.kriging_range.unwrap_or(1.0),
            options.kriging_nugget.unwrap_or(1e-6),
        )?)),
        "intermittent_demand" => {
            let config = IntermittentDemandConfig {
                alpha: options.alpha.unwrap_or(0.2),
                beta: options.beta.unwrap_or(0.2),
                validation_window: options.validation_window,
                ..IntermittentDemandConfig::default()
            };
            Ok(Box::new(IntermittentDemandForecaster::new(config)?))
        }
        "classical_expert_bank" => Ok(Box::new(ClassicalExpertBank::default_for_season_length(
            options.season_length.unwrap_or(7),
        )?)),
        "autostats_bank" => Ok(Box::new(AutoStatsBank::with_validation_window(
            options.season_length.unwrap_or(7),
            options.validation_window,
        )?)),
        "stl_cartoboost" => Ok(Box::new(STLCartoBoostForecaster::new(
            options.season_length.unwrap_or(7),
        )?)),
        "mstl_cartoboost" => Ok(Box::new(MSTLCartoBoostForecaster::new(
            options
                .mstl_season_lengths
                .clone()
                .unwrap_or_else(|| vec![options.season_length.unwrap_or(7)]),
        )?)),
        "cartoboost_lag" => Ok(Box::new(CartoBoostLagForecaster::new(
            lag_config(options),
            booster_config(options),
        )?)),
        "cartoboost_direct" => Ok(Box::new(BrowserDirectForecaster::new(
            lag_config(options),
            booster_config(options),
            horizon,
        )?)),
        "rectified_recursive" => Ok(Box::new(BrowserRectifiedRecursiveForecaster::new(
            lag_config(options),
            booster_config(options),
            horizon,
        )?)),
        "lag_plus" => Ok(Box::new(LagPlusForecaster::new(LagPlusConfig::new(
            lag_config(options),
            booster_config(options),
        ))?)),
        "auto_forecast" => {
            let mut config = AutoForecastConfig {
                lag_config: lag_config(options),
                booster_config: booster_config(options),
                ..AutoForecastConfig::default()
            };
            if let Some(season_length) = options.season_length {
                config.season_length = season_length;
            }
            if let Some(validation_window) = options.validation_window {
                config.validation_window = Some(validation_window);
            }
            config.max_direct_horizon = options.max_direct_horizon.unwrap_or(horizon);
            Ok(Box::new(AutoForecastModel::new(config)?))
        }
        "scaled_cartoboost_lag" => Ok(Box::new(LocalStandardScaledForecaster::new(
            Box::new(CartoBoostLagForecaster::new(
                lag_config(options),
                booster_config(options),
            )?),
            1e-6,
            "scaled_cartoboost_lag",
        )?)),
        "log1p_cartoboost_lag" => Ok(Box::new(Log1pForecaster::new(
            Box::new(CartoBoostLagForecaster::new(
                lag_config(options),
                booster_config(options),
            )?),
            "log1p_cartoboost_lag",
        ))),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported browser forecast model {other:?}"
        ))),
    }
}

struct BrowserDirectForecaster {
    inner: CartoBoostDirectForecaster,
    fit_horizon: usize,
}

impl BrowserDirectForecaster {
    fn new(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        fit_horizon: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: CartoBoostDirectForecaster::new(lag_config, booster_config)?,
            fit_horizon,
        })
    }
}

impl Forecaster for BrowserDirectForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.inner.fit_horizon(frame, self.fit_horizon)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.inner.predict(horizon)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn metadata(&self) -> Value {
        self.inner.metadata()
    }
}

struct BrowserRectifiedRecursiveForecaster {
    inner: RectifiedRecursiveForecaster,
    fit_horizon: usize,
}

impl BrowserRectifiedRecursiveForecaster {
    fn new(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        fit_horizon: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: RectifiedRecursiveForecaster::new(lag_config, booster_config)?,
            fit_horizon,
        })
    }
}

impl Forecaster for BrowserRectifiedRecursiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.inner.fit_horizon(frame, self.fit_horizon)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.inner.predict(horizon)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn metadata(&self) -> Value {
        self.inner.metadata()
    }
}

fn theta_seasonality(options: &BrowserForecastOptions) -> Result<Option<ThetaSeasonality>> {
    let Some(kind) = options.theta_seasonality.as_deref() else {
        return Ok(None);
    };
    let season_length = options.season_length.unwrap_or(7);
    match kind.trim().to_ascii_lowercase().as_str() {
        "" | "none" => Ok(None),
        "additive" => ThetaSeasonality::additive(season_length).map(Some),
        "multiplicative" => ThetaSeasonality::multiplicative(season_length).map(Some),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported theta seasonality {other:?}"
        ))),
    }
}

fn default_model() -> String {
    "auto_forecast".to_string()
}

fn booster_config(options: &BrowserForecastOptions) -> BoosterConfig {
    let mut config = BoosterConfig::default();
    if let Some(n_estimators) = options.n_estimators {
        config.n_estimators = n_estimators;
    }
    if let Some(learning_rate) = options.learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(max_depth) = options.max_depth {
        config.max_depth = max_depth;
    }
    if let Some(min_samples_leaf) = options.min_samples_leaf {
        config.min_samples_leaf = min_samples_leaf;
    }
    config
}

fn lag_config(options: &BrowserForecastOptions) -> LagFeatureConfig {
    LagFeatureConfig {
        lags: options
            .lags
            .clone()
            .unwrap_or_else(|| vec![1, 2, 3, options.season_length.unwrap_or(7)]),
        rolling_mean_windows: options
            .rolling_mean_windows
            .clone()
            .unwrap_or_else(|| vec![options.season_length.unwrap_or(7)]),
        rolling_std_windows: options.rolling_std_windows.clone().unwrap_or_default(),
        rolling_min_windows: options.rolling_min_windows.clone().unwrap_or_default(),
        rolling_max_windows: options.rolling_max_windows.clone().unwrap_or_default(),
        difference_lags: options.difference_lags.clone().unwrap_or_default(),
        rolling_trend_windows: options.rolling_trend_windows.clone().unwrap_or_default(),
        calendar_features: calendar_features(options),
        ..LagFeatureConfig::default()
    }
}

fn calendar_features(options: &BrowserForecastOptions) -> Vec<CalendarFeature> {
    let Some(features) = &options.calendar_features else {
        return vec![CalendarFeature::DayOfWeek, CalendarFeature::Month];
    };
    features
        .iter()
        .filter_map(
            |feature| match feature.trim().to_ascii_lowercase().as_str() {
                "day_of_week" | "dow" => Some(CalendarFeature::DayOfWeek),
                "day_of_week_sin" | "dow_sin" => Some(CalendarFeature::DayOfWeekSin),
                "day_of_week_cos" | "dow_cos" => Some(CalendarFeature::DayOfWeekCos),
                "month" => Some(CalendarFeature::Month),
                "month_sin" => Some(CalendarFeature::MonthSin),
                "month_cos" => Some(CalendarFeature::MonthCos),
                "day" => Some(CalendarFeature::Day),
                "day_sin" => Some(CalendarFeature::DaySin),
                "day_cos" => Some(CalendarFeature::DayCos),
                "day_of_year" | "doy" => Some(CalendarFeature::DayOfYear),
                "elapsed_index" => Some(CalendarFeature::ElapsedIndex),
                "elapsed_phase" => Some(CalendarFeature::ElapsedPhase(
                    options.season_length.unwrap_or(7).max(2),
                )),
                _ => None,
            },
        )
        .collect()
}

fn coordinates_from_frame(
    frame: &ForecastFrame,
    options: &BrowserForecastOptions,
) -> Result<BTreeMap<String, (f64, f64)>> {
    let x_name = options
        .coordinate_x
        .as_deref()
        .unwrap_or_else(|| infer_covariate(frame, &["longitude", "lon", "lng", "x"]).unwrap_or(""));
    let y_name = options
        .coordinate_y
        .as_deref()
        .unwrap_or_else(|| infer_covariate(frame, &["latitude", "lat", "y"]).unwrap_or(""));
    if x_name.is_empty() || y_name.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging requires coordinate covariates such as longitude/latitude or x/y".to_string(),
        ));
    }
    let mut coordinates = BTreeMap::new();
    for row in frame.rows() {
        if coordinates.contains_key(&row.series_id) {
            continue;
        }
        let x = row.covariates.get(x_name).copied().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing kriging x coordinate covariate {x_name:?}"
            ))
        })?;
        let y = row.covariates.get(y_name).copied().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "missing kriging y coordinate covariate {y_name:?}"
            ))
        })?;
        coordinates.insert(row.series_id.clone(), (x, y));
    }
    Ok(coordinates)
}

fn infer_covariate<'a>(frame: &'a ForecastFrame, names: &[&str]) -> Option<&'a str> {
    let first = frame.rows().first()?;
    for candidate in names {
        if let Some((name, _)) = first
            .covariates
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(candidate))
        {
            return Some(name.as_str());
        }
    }
    None
}

#[allow(dead_code)]
fn _assert_forecast_result_is_serializable(result: &ForecastResult) -> Value {
    result.to_json_value()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn forecast_model_registry_keeps_cartoboost_first_without_duplicates() {
        let registry = forecast_model_registry();
        let names = registry.iter().map(|model| model.name).collect::<Vec<_>>();
        assert_eq!(
            &names[..7],
            &[
                "auto_forecast",
                "cartoboost_lag",
                "cartoboost_direct",
                "rectified_recursive",
                "lag_plus",
                "scaled_cartoboost_lag",
                "log1p_cartoboost_lag",
            ]
        );
        let unique = names.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn browser_sequence_viterbi_runs_through_wasm_dispatch() {
        let response = run_sequence_request(BrowserSequenceRequest {
            operation: "reference_path_viterbi".to_string(),
            frame: None,
            series: Some(sample_sequence_series()),
            reference: Some(ReferenceSignal {
                axis: vec![0.0, 1.0, 2.0, 3.0],
                signal: vec![0.0, 1.0, 2.0, 3.0],
            }),
            state_space_config: None,
            reference_path_config: Some(ReferencePathConfig::default()),
            candidates: None,
            weights: None,
            actuals: None,
            oof_fold: None,
            oof_rows: None,
            group_predictions: None,
        })
        .expect("sequence request");
        let points = response
            .get("points")
            .and_then(Value::as_array)
            .expect("path points");
        assert_eq!(points.len(), 4);
        assert_eq!(points[1]["axis"].as_f64(), Some(1.0));
    }

    #[test]
    fn browser_sequence_oof_generation_runs_through_wasm_dispatch() {
        let response = run_sequence_request(BrowserSequenceRequest {
            operation: "generate_group_oof_candidate_rows".to_string(),
            frame: None,
            series: None,
            reference: None,
            state_space_config: None,
            reference_path_config: None,
            candidates: None,
            weights: None,
            actuals: None,
            oof_fold: Some(SequenceOofFold {
                validation_group_id: "pickup_zone_1".to_string(),
                train_group_ids: vec!["pickup_zone_2".to_string()],
                actuals: vec![SequenceCandidatePrediction {
                    series_id: "pickup_zone_1".to_string(),
                    row_id: "hour_01".to_string(),
                    value: 10.0,
                }],
                candidates: vec![SequenceCandidate {
                    name: "candidate_a".to_string(),
                    predictions: vec![SequenceCandidatePrediction {
                        series_id: "pickup_zone_1".to_string(),
                        row_id: "hour_01".to_string(),
                        value: 11.0,
                    }],
                }],
            }),
            oof_rows: None,
            group_predictions: None,
        })
        .expect("sequence OOF request");
        let rows = response.as_array().expect("OOF rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0]["candidate_predictions"]["candidate_a"].as_f64(),
            Some(11.0)
        );
    }

    #[test]
    fn every_registered_browser_forecast_model_runs_on_representative_panel() {
        for model in forecast_model_registry() {
            let request = BrowserForecastRequest {
                rows: sample_panel_rows(),
                frequency: "daily".to_string(),
                horizon: 7,
                model: model.name.to_string(),
                options: BrowserForecastOptions {
                    season_length: Some(7),
                    coordinate_x: Some("longitude".to_string()),
                    coordinate_y: Some("latitude".to_string()),
                    ..BrowserForecastOptions::default()
                },
                metadata: BrowserForecastMetadata {
                    timestamp_col: Some("timestamp".to_string()),
                    target_col: Some("target".to_string()),
                    series_id_col: Some("series_id".to_string()),
                },
            };
            let response = run_forecast_request(request)
                .unwrap_or_else(|error| panic!("{} failed: {error}", model.name));
            let records = response
                .forecast
                .get("records")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("{} returned no forecast records", model.name));
            assert_eq!(records.len(), 21, "{} record count", model.name);
        }
    }

    #[test]
    fn browser_auto_forecast_caps_direct_horizon_to_requested_horizon() {
        let request = BrowserForecastRequest {
            rows: (0..56)
                .map(|day| BrowserForecastRow {
                    series_id: Some("pickup_zone_1".to_string()),
                    timestamp: date_string(day),
                    target: 120.0 + day as f64 * 2.0 + (day % 7) as f64 * 4.0,
                    covariates: BTreeMap::new(),
                })
                .collect(),
            frequency: "daily".to_string(),
            horizon: 14,
            model: "auto_forecast".to_string(),
            options: BrowserForecastOptions {
                season_length: Some(7),
                ..BrowserForecastOptions::default()
            },
            metadata: BrowserForecastMetadata {
                timestamp_col: Some("timestamp".to_string()),
                target_col: Some("target".to_string()),
                series_id_col: Some("series_id".to_string()),
            },
        };
        let response = run_forecast_request(request).expect("auto forecast run");
        let records = response
            .forecast
            .get("records")
            .and_then(Value::as_array)
            .expect("forecast records");
        assert_eq!(records.len(), 14);
    }

    #[test]
    fn browser_regression_model_scores_holdout_and_reports_importance() {
        let request = BrowserRegressionRequest {
            rows: sample_regression_rows(),
            feature_names: vec![
                "trip_distance".to_string(),
                "pickup_hour".to_string(),
                "route_pressure".to_string(),
                "pickup_x".to_string(),
                "pickup_y".to_string(),
            ],
            sparse_feature_names: vec!["zone_memberships".to_string()],
            options: BrowserRegressionOptions {
                holdout_fraction: 0.25,
                splitter_mode: Some("full".to_string()),
                feature_kinds: BTreeMap::from([
                    ("trip_distance".to_string(), "numeric".to_string()),
                    ("pickup_hour".to_string(), "periodic".to_string()),
                    ("route_pressure".to_string(), "numeric".to_string()),
                    ("pickup_x".to_string(), "spatial".to_string()),
                    ("pickup_y".to_string(), "spatial".to_string()),
                ]),
                periodic_periods: BTreeMap::from([("pickup_hour".to_string(), 24)]),
                loss: Some("huber".to_string()),
                quantile_alpha: None,
                huber_delta: Some(5.0),
                log_offset: None,
                interval_lower_alpha: Some(0.1),
                interval_upper_alpha: Some(0.9),
                n_estimators: Some(80),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
            },
        };
        let response = run_regression_request(request).expect("regression run");
        assert_eq!(response.metrics.train_rows, 45);
        assert_eq!(response.metrics.holdout_rows, 15);
        assert_eq!(response.predictions.len(), 15);
        assert_eq!(response.metadata["splitterMode"].as_str(), Some("full"));
        assert!(response.metrics.rmse.is_finite());
        assert!(response.metrics.mae.is_finite());
        assert!(response.metrics.r2.is_finite());
        assert_eq!(response.feature_importance.len(), 6);
        assert_eq!(
            response.metadata["sparseFeatureNames"][0].as_str(),
            Some("zone_memberships")
        );
        assert_eq!(response.metadata["loss"].as_str(), Some("huber"));
        assert!(response
            .predictions
            .iter()
            .all(|row| row.lower_prediction.is_some() && row.upper_prediction.is_some()));
        assert!(response.predictions.iter().all(|row| row
            .lower_prediction
            .zip(row.upper_prediction)
            .is_some_and(|(lower, upper)| lower <= upper)));
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.split_count > 0));
    }

    #[test]
    fn browser_regression_model_rejects_unknown_loss() {
        let mut request = BrowserRegressionRequest {
            rows: sample_regression_rows(),
            feature_names: vec![
                "trip_distance".to_string(),
                "pickup_hour".to_string(),
                "route_pressure".to_string(),
                "pickup_x".to_string(),
                "pickup_y".to_string(),
            ],
            sparse_feature_names: vec!["zone_memberships".to_string()],
            options: BrowserRegressionOptions::default(),
        };
        request.options.loss = Some("not_a_loss".to_string());
        let error = run_regression_request(request).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported browser regression loss"));
    }

    #[test]
    fn browser_regression_model_rejects_bad_feature_width() {
        let error = run_regression_request(BrowserRegressionRequest {
            rows: vec![
                BrowserRegressionRow {
                    features: vec![1.0, 2.0],
                    sparse_sets: Vec::new(),
                    target: 3.0,
                },
                BrowserRegressionRow {
                    features: vec![2.0],
                    sparse_sets: Vec::new(),
                    target: 4.0,
                },
                BrowserRegressionRow {
                    features: vec![3.0, 4.0],
                    sparse_sets: Vec::new(),
                    target: 5.0,
                },
                BrowserRegressionRow {
                    features: vec![4.0, 5.0],
                    sparse_sets: Vec::new(),
                    target: 6.0,
                },
            ],
            feature_names: vec!["x".to_string(), "z".to_string()],
            sparse_feature_names: Vec::new(),
            options: BrowserRegressionOptions::default(),
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("feature row has 1 columns but feature_names has 2"));
    }

    #[test]
    fn browser_neural_embedding_model_scores_holdout() {
        let request = BrowserNeuralRequest {
            rows: sample_neural_rows(),
            dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
            node_features: Vec::new(),
            node_types: Vec::new(),
            edge_type_triples: Vec::new(),
            pipeline: "embedding".to_string(),
            options: BrowserNeuralOptions {
                holdout_fraction: 0.25,
                embedding_dim: Some(4),
                random_state: Some(42),
                n_estimators: Some(40),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
                ..BrowserNeuralOptions::default()
            },
        };
        let response = run_neural_request(request).expect("embedding neural run");
        assert_eq!(response.metrics.train_rows, 36);
        assert_eq!(response.metrics.holdout_rows, 12);
        assert_eq!(response.predictions.len(), 12);
        assert_eq!(
            response.metadata["details"]["model"].as_str(),
            Some("neural_embedding_regressor")
        );
        assert!(response.metrics.rmse.is_finite());
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.feature.starts_with("embedding_")));
    }

    #[test]
    fn browser_node2vec_model_scores_pair_holdout() {
        let request = BrowserNeuralRequest {
            rows: sample_neural_rows(),
            dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
            node_features: sample_node_features(),
            node_types: Vec::new(),
            edge_type_triples: Vec::new(),
            pipeline: "node2vec".to_string(),
            options: BrowserNeuralOptions {
                holdout_fraction: 0.25,
                embedding_dim: Some(4),
                node2vec_walk_length: Some(6),
                node2vec_walks_per_node: Some(3),
                node2vec_window_size: Some(2),
                node2vec_epochs: Some(2),
                node2vec_seed: Some(7),
                n_estimators: Some(40),
                learning_rate: Some(0.08),
                max_depth: Some(3),
                min_samples_leaf: Some(2),
                ..BrowserNeuralOptions::default()
            },
        };
        let response = run_neural_request(request).expect("node2vec neural run");
        assert_eq!(response.metrics.train_rows, 36);
        assert_eq!(response.metrics.holdout_rows, 12);
        assert_eq!(response.predictions.len(), 12);
        assert_eq!(
            response.metadata["details"]["model"].as_str(),
            Some("node2vec_regressor")
        );
        assert_eq!(response.metadata["details"]["nodeCount"].as_u64(), Some(8));
        assert!(response.metrics.mae.is_finite());
        assert!(response
            .feature_importance
            .iter()
            .any(|item| item.feature.starts_with("node2vec_")));
    }

    #[test]
    fn browser_graphsage_family_models_score_pair_holdout() {
        for (pipeline, expected_model, expected_prefix) in [
            ("graphsage", "graphsage_regressor", "graphsage_"),
            (
                "hetero_graphsage",
                "hetero_graphsage_regressor",
                "hetero_graphsage_",
            ),
            ("hinsage", "hinsage_regressor", "hinsage_"),
        ] {
            let request = BrowserNeuralRequest {
                rows: sample_neural_rows(),
                dense_feature_names: vec!["trip_distance".to_string(), "pickup_hour".to_string()],
                node_features: sample_node_features(),
                node_types: vec![0, 0, 0, 0, 1, 1, 1, 1],
                edge_type_triples: vec![(0, 0, 1)],
                pipeline: pipeline.to_string(),
                options: BrowserNeuralOptions {
                    holdout_fraction: 0.25,
                    embedding_dim: Some(4),
                    graph_sage_epochs: Some(2),
                    graph_sage_negative_samples: Some(2),
                    graph_sage_seed: Some(11),
                    n_estimators: Some(40),
                    learning_rate: Some(0.08),
                    max_depth: Some(3),
                    min_samples_leaf: Some(2),
                    ..BrowserNeuralOptions::default()
                },
            };
            let response = run_neural_request(request).expect("graph neural run");
            assert_eq!(response.metrics.train_rows, 36);
            assert_eq!(response.metrics.holdout_rows, 12);
            assert_eq!(response.predictions.len(), 12);
            assert_eq!(
                response.metadata["details"]["model"].as_str(),
                Some(expected_model)
            );
            assert_eq!(response.metadata["details"]["nodeCount"].as_u64(), Some(8));
            assert!(response.metrics.rmse.is_finite());
            assert!(response
                .feature_importance
                .iter()
                .any(|item| item.feature.starts_with(expected_prefix)));
        }
    }

    fn sample_panel_rows() -> Vec<BrowserForecastRow> {
        let mut rows = Vec::new();
        for (series_index, series_id) in ["pickup_zone_1", "pickup_zone_2", "pickup_zone_3"]
            .iter()
            .enumerate()
        {
            for day in 0..70 {
                let weekly = (day % 7) as f64;
                let level = 120.0 + series_index as f64 * 30.0;
                let target = level + day as f64 * 1.4 + weekly * 3.0;
                rows.push(BrowserForecastRow {
                    series_id: Some((*series_id).to_string()),
                    timestamp: date_string(day),
                    target,
                    covariates: BTreeMap::from([
                        ("longitude".to_string(), -73.98 + series_index as f64 * 0.02),
                        ("latitude".to_string(), 40.74 + series_index as f64 * 0.02),
                    ]),
                });
            }
        }
        rows
    }

    fn sample_regression_rows() -> Vec<BrowserRegressionRow> {
        (0..60)
            .map(|idx| {
                let trip_distance = 0.8 + idx as f64 * 0.12;
                let pickup_hour = (idx % 24) as f64;
                let route_pressure = ((idx * 7) % 11) as f64;
                let pickup_x = -73.98 + (idx as f64 / 6.0).sin() * 0.04;
                let pickup_y = 40.74 + (idx as f64 / 7.0).cos() * 0.03;
                let neighborhood_signal = if idx % 3 == 0 { 5.0 } else { 0.0 };
                BrowserRegressionRow {
                    features: vec![
                        trip_distance,
                        pickup_hour,
                        route_pressure,
                        pickup_x,
                        pickup_y,
                    ],
                    sparse_sets: vec![vec![101 + (idx % 3) as u64, 200 + (idx % 5) as u64]],
                    target: 6.0
                        + trip_distance * 2.4
                        + pickup_hour * 0.35
                        + route_pressure * 1.1
                        + (pickup_x + 74.0) * 10.0
                        + (pickup_y - 40.7) * 12.0
                        + neighborhood_signal,
                }
            })
            .collect()
    }

    fn sample_neural_rows() -> Vec<BrowserNeuralRow> {
        (0..48)
            .map(|idx| {
                let source = idx % 4;
                let target_node = 4 + ((idx * 3) % 4);
                let trip_distance = 1.0 + (idx % 8) as f64 * 0.35;
                let pickup_hour = (idx % 24) as f64;
                BrowserNeuralRow {
                    id: Some((source + 1) as u64),
                    source: Some(source),
                    target_node: Some(target_node),
                    edge_weight: Some(1.0 + (idx % 3) as f32 * 0.2),
                    edge_type: Some(0),
                    dense: vec![trip_distance, pickup_hour],
                    target: 20.0
                        + source as f64 * 4.0
                        + target_node as f64 * 2.5
                        + trip_distance * 3.0
                        + pickup_hour * 0.4,
                }
            })
            .collect()
    }

    fn sample_sequence_series() -> SequenceSeries {
        SequenceSeries {
            series_id: "pickup_zone_1".to_string(),
            rows: vec![
                sequence_row("r0", 0.0, Some(0.0)),
                sequence_row("r1", 1.0, Some(1.0)),
                sequence_row("r2", 2.0, None),
                sequence_row("r3", 3.0, None),
            ],
        }
    }

    fn sequence_row(
        row_id: &str,
        position: f64,
        target: Option<f64>,
    ) -> cartoboost_core::forecasting::SequenceRow {
        cartoboost_core::forecasting::SequenceRow {
            row_id: row_id.to_string(),
            position,
            target,
            reference_axis: None,
            reference_signal: None,
            auxiliary_rate: None,
        }
    }

    fn sample_node_features() -> Vec<Vec<f32>> {
        (0..8)
            .map(|node| {
                vec![
                    node as f32 / 8.0,
                    if node < 4 { 0.0 } else { 1.0 },
                    ((node * 3) % 5) as f32 / 5.0,
                ]
            })
            .collect()
    }

    fn date_string(day_index: usize) -> String {
        const MONTH_LENGTHS: [usize; 3] = [31, 28, 31];
        let mut remaining = day_index;
        for (month_index, month_length) in MONTH_LENGTHS.iter().enumerate() {
            if remaining < *month_length {
                return format!(
                    "2026-{month:02}-{day:02}",
                    month = month_index + 1,
                    day = remaining + 1
                );
            }
            remaining -= month_length;
        }
        panic!("sample day index out of range");
    }
}
