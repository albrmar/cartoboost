use cartoboost_core::data::{FeatureSchema, SparseSetColumn};
use cartoboost_core::forecasting::{
    ArimaForecaster as CoreArimaForecaster, AutoARIMAForecaster as CoreAutoARIMAForecaster,
    AutoKalmanForecaster as CoreAutoKalmanForecaster,
    AutoLocalLevelKalmanForecaster as CoreAutoLocalLevelKalmanForecaster,
    BacktestFoldResult as CoreBacktestFoldResult, BacktestResult as CoreBacktestResult,
    CalendarFeature, CartoBoostLagForecaster as CoreCartoBoostLagForecaster,
    ETSForecaster as CoreETSForecaster, ForecastActual, ForecastFold as CoreForecastFold,
    ForecastFrame as CoreForecastFrame, ForecastFrameMetadata, ForecastFrequency,
    ForecastMetricSet as CoreForecastMetricSet, ForecastPrediction,
    ForecastResult as CoreForecastResult, ForecastRow as CoreForecastRow, ForecastWindow,
    Forecaster, GlobalForecastTargetMode, KalmanForecaster as CoreKalmanForecaster,
    KrigingForecaster as CoreKrigingForecaster, LagFeatureConfig,
    LocalLevelKalmanForecaster as CoreLocalLevelKalmanForecaster,
    NaiveForecaster as CoreNaiveForecaster,
    OptimizedThetaForecaster as CoreOptimizedThetaForecaster,
    RollingOriginBacktester as CoreRollingOriginBacktester,
    RollingOriginSplitter as CoreRollingOriginSplitter,
    SeasonalNaiveForecaster as CoreSeasonalNaiveForecaster, ThetaForecaster as CoreThetaForecaster,
    ThetaSeasonality, WeightedEnsembleForecaster as CoreWeightedEnsembleForecaster,
};
use cartoboost_core::geo::{
    assemble_sparse_column, assemble_sparse_row, expand_h3_sparse_set as core_expand_h3_sparse_set,
    normalize_coordinate as core_normalize_coordinate, normalize_h3_id_text,
    normalize_h3_resolution, normalize_s2_id_text, normalize_s2_level, scaffold_h3_parent_id,
    validate_equal_row_count, validate_parent_levels, GeoGridKind,
};
use cartoboost_core::loss::{HuberLossConfig, LogL2LossConfig, LossConfig, QuantileLossConfig};
use cartoboost_core::tree::{FlatAxisPredictor, FuzzyKernel, LeafPredictorKind, SplitterKind};
use cartoboost_core::utilities::{
    empirical_variogram, fit_local_level_kalman, fit_local_linear_kalman,
    fit_ordinary_kriging_variogram, intermittent_demand_forecast, local_level_kalman_forecast,
    local_level_kalman_forecast_distribution, local_linear_kalman_forecast,
    local_linear_kalman_forecast_distribution, ordinary_kriging_leave_one_out,
    ordinary_kriging_leave_one_out_diagnostics, ordinary_kriging_predict_many,
    IntermittentDemandMethod, KrigingDrift, KrigingObservation, KrigingVariogramModel,
    LocalLevelKalmanConfig, LocalLinearKalmanConfig, OrdinaryKrigingConfig,
};
use cartoboost_core::{Booster, BoosterConfig, CartoBoostError, Dataset, Model};
use cartoboost_neural::{
    build_embedding_table_artifact, compute_directional_features, fit_embedding_table_with_options,
    materialize_source_target_pair_nodes, validate_directed_metapath,
    write_embedding_table_artifact, ArtifactFallbackKind, EmbeddingTable, GraphSageConfig,
    GraphSageEncoder, GraphSageLinkPredictor, GraphSageRegressor, HeteroGraph,
    HeteroGraphSageConfig, HeteroGraphSageEncoder, HeteroGraphSageLinkPredictor,
    HeteroGraphSageRegressor, HeteroTypedEdge, HinSageConfig, HinSageEncoder, HinSageGraph,
    HinSageLinkPredictor, HinSageRegressor, HomogeneousGraph,
    NeuralEmbeddingRegressor as StandaloneNeuralEmbeddingRegressor, Node2VecConfig,
    Node2VecEncoder, Node2VecLinkPredictor, Node2VecRegressor, StandaloneBoosterConfig,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyType};
use rayon::ThreadPoolBuilder;
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::path::PathBuf;

type StringTypedEdges = Vec<(String, String, String)>;
type PyKrigingPrediction = (f64, f64, f64, Vec<f64>);
type PyDetailedKrigingPrediction = (f64, f64, f64, f64, Vec<f64>, Vec<usize>);

#[pyclass(name = "ForecastFrame")]
#[derive(Clone, Debug)]
struct NativeForecastFrame {
    frame: CoreForecastFrame,
}

#[pymethods]
impl NativeForecastFrame {
    #[new]
    #[pyo3(signature = (rows, frequency, timestamp_col=None, target_col=None, series_id_col=None, static_covariates=None, known_future_covariates=None, historical_covariates=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        rows: Vec<(String, String, f64)>,
        frequency: &str,
        timestamp_col: Option<String>,
        target_col: Option<String>,
        series_id_col: Option<String>,
        static_covariates: Option<Vec<String>>,
        known_future_covariates: Option<Vec<String>>,
        historical_covariates: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let frequency = ForecastFrequency::parse(frequency).map_err(to_py_value_error)?;
        let frequency_name = frequency.as_str().to_string();
        let metadata = ForecastFrameMetadata {
            timestamp_col,
            target_col,
            series_id_col,
            static_covariates: static_covariates.unwrap_or_default(),
            known_future_covariates: known_future_covariates.unwrap_or_default(),
            historical_covariates: historical_covariates.unwrap_or_default(),
        };
        let frame = py
            .allow_threads(|| {
                let frequency = ForecastFrequency::parse(&frequency_name)?;
                CoreForecastFrame::from_string_rows(rows, frequency, metadata)
            })
            .map_err(to_py_value_error)?;
        Ok(Self { frame })
    }

    fn row_count(&self) -> usize {
        self.frame.rows().len()
    }

    fn frequency(&self) -> String {
        self.frame.frequency().as_str().to_string()
    }

    fn series_ids(&self) -> Vec<String> {
        self.frame.series_ids()
    }

    fn metadata_json(&self) -> PyResult<String> {
        self.frame.metadata_json_string().map_err(to_py_value_error)
    }

    fn rows(&self) -> Vec<(String, String, f64)> {
        self.frame
            .rows()
            .iter()
            .map(|row| {
                (
                    row.series_id.clone(),
                    row.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    row.target,
                )
            })
            .collect()
    }
}

#[pyclass(name = "ForecastResult")]
#[derive(Clone, Debug)]
struct NativeForecastResult {
    result: CoreForecastResult,
}

#[pymethods]
impl NativeForecastResult {
    #[new]
    fn new(
        py: Python<'_>,
        predictions: Vec<(String, String, usize, String, f64)>,
    ) -> PyResult<Self> {
        let result = py
            .allow_threads(|| {
                let predictions = predictions
                    .into_iter()
                    .map(|(series_id, timestamp, horizon, model, mean)| {
                        Ok(ForecastPrediction {
                            series_id,
                            timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(
                                &timestamp,
                            )?,
                            horizon,
                            model,
                            mean,
                        })
                    })
                    .collect::<cartoboost_core::Result<Vec<_>>>()?;
                CoreForecastResult::new(predictions)
            })
            .map_err(to_py_value_error)?;
        Ok(Self { result })
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, value: &str) -> PyResult<Self> {
        let value = value.to_string();
        let result = py
            .allow_threads(|| CoreForecastResult::from_json_string(&value))
            .map_err(to_py_value_error)?;
        Ok(Self { result })
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.result.to_json_string())
            .map_err(to_py_value_error)
    }

    fn columns(&self) -> Vec<String> {
        CoreForecastResult::prediction_columns()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    fn predictions(&self) -> Vec<(String, String, usize, String, f64)> {
        self.result
            .predictions()
            .iter()
            .map(|prediction| {
                (
                    prediction.series_id.clone(),
                    prediction.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    prediction.horizon,
                    prediction.model.clone(),
                    prediction.mean,
                )
            })
            .collect()
    }
}

#[pyclass(name = "ForecastFold")]
#[derive(Clone, Debug)]
struct NativeForecastFold {
    fold: CoreForecastFold,
}

#[pymethods]
impl NativeForecastFold {
    #[getter]
    fn fold_id(&self) -> String {
        self.fold.fold_id.clone()
    }

    #[getter]
    fn train_indices(&self) -> Vec<usize> {
        self.fold.train_indices.clone()
    }

    #[getter]
    fn validation_indices(&self) -> Vec<usize> {
        self.fold.validation_indices.clone()
    }

    #[getter]
    fn train_start(&self) -> String {
        format_forecast_timestamp(self.fold.train_start)
    }

    #[getter]
    fn train_end(&self) -> String {
        format_forecast_timestamp(self.fold.train_end)
    }

    #[getter]
    fn validation_start(&self) -> String {
        format_forecast_timestamp(self.fold.validation_start)
    }

    #[getter]
    fn validation_end(&self) -> String {
        format_forecast_timestamp(self.fold.validation_end)
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.fold.horizon
    }

    #[getter]
    fn step(&self) -> usize {
        self.fold.step
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.fold.metadata)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.fold).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "RollingOriginSplitter")]
#[derive(Clone, Debug)]
struct NativeRollingOriginSplitter {
    splitter: CoreRollingOriginSplitter,
}

#[pymethods]
impl NativeRollingOriginSplitter {
    #[new]
    #[pyo3(signature = (horizon, step=1, min_train_size=1, max_train_size=None, n_splits=None, window="expanding"))]
    fn new(
        horizon: usize,
        step: usize,
        min_train_size: usize,
        max_train_size: Option<usize>,
        n_splits: Option<usize>,
        window: &str,
    ) -> PyResult<Self> {
        let window = parse_forecast_window(window)?;
        Ok(Self {
            splitter: CoreRollingOriginSplitter::new(
                horizon,
                step,
                min_train_size,
                max_train_size,
                n_splits,
                window,
            )
            .map_err(to_py_value_error)?,
        })
    }

    #[staticmethod]
    fn expanding(horizon: usize, min_train_size: usize) -> PyResult<Self> {
        Ok(Self {
            splitter: CoreRollingOriginSplitter::expanding(horizon, min_train_size)
                .map_err(to_py_value_error)?,
        })
    }

    #[staticmethod]
    fn sliding(horizon: usize, min_train_size: usize, max_train_size: usize) -> PyResult<Self> {
        Ok(Self {
            splitter: CoreRollingOriginSplitter::sliding(horizon, min_train_size, max_train_size)
                .map_err(to_py_value_error)?,
        })
    }

    #[getter]
    fn horizon(&self) -> usize {
        self.splitter.horizon
    }

    #[getter]
    fn step(&self) -> usize {
        self.splitter.step
    }

    #[getter]
    fn min_train_size(&self) -> usize {
        self.splitter.min_train_size
    }

    #[getter]
    fn max_train_size(&self) -> Option<usize> {
        self.splitter.max_train_size
    }

    #[getter]
    fn n_splits(&self) -> Option<usize> {
        self.splitter.n_splits
    }

    #[getter]
    fn window(&self) -> &'static str {
        forecast_window_name(&self.splitter.window)
    }

    fn split(
        &self,
        py: Python<'_>,
        frame: &NativeForecastFrame,
    ) -> PyResult<Vec<NativeForecastFold>> {
        Ok(py
            .allow_threads(|| self.splitter.split(&frame.frame))
            .map_err(to_py_value_error)?
            .into_iter()
            .map(|fold| NativeForecastFold { fold })
            .collect())
    }

    fn n_splits_for_frame(&self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<usize> {
        Ok(self.split(py, frame)?.len())
    }
}

#[pyclass(name = "ForecastMetricSet")]
#[derive(Clone, Debug)]
struct NativeForecastMetricSet {
    metrics: CoreForecastMetricSet,
}

#[pymethods]
impl NativeForecastMetricSet {
    #[new]
    #[pyo3(signature = (mae=0.0, rmse=0.0, wape=0.0, smape=0.0, bias=0.0, mase=None))]
    fn new(mae: f64, rmse: f64, wape: f64, smape: f64, bias: f64, mase: Option<f64>) -> Self {
        Self {
            metrics: CoreForecastMetricSet {
                mae,
                rmse,
                wape,
                smape,
                bias,
                mase,
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (forecast, actuals, training_actuals=None, mase_seasonality=None))]
    fn evaluate(
        py: Python<'_>,
        forecast: &NativeForecastResult,
        actuals: Vec<(String, String, usize, f64)>,
        training_actuals: Option<Vec<(String, String, usize, f64)>>,
        mase_seasonality: Option<usize>,
    ) -> PyResult<Self> {
        let actuals = parse_forecast_actuals(actuals)?;
        let training_actuals = parse_forecast_actuals(training_actuals.unwrap_or_default())?;
        let metrics = py
            .allow_threads(|| {
                cartoboost_core::forecasting::evaluate_forecast_with_training(
                    &forecast.result,
                    &actuals,
                    &training_actuals,
                    mase_seasonality,
                )
            })
            .map_err(to_py_value_error)?;
        Ok(Self { metrics })
    }

    #[getter]
    fn mae(&self) -> f64 {
        self.metrics.mae
    }

    #[getter]
    fn rmse(&self) -> f64 {
        self.metrics.rmse
    }

    #[getter]
    fn wape(&self) -> f64 {
        self.metrics.wape
    }

    #[getter]
    fn smape(&self) -> f64 {
        self.metrics.smape
    }

    #[getter]
    fn bias(&self) -> f64 {
        self.metrics.bias
    }

    #[getter]
    fn mase(&self) -> Option<f64> {
        self.metrics.mase
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.metrics).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyfunction]
#[pyo3(signature = (forecast, actuals, training_actuals=None, mase_seasonality=None))]
fn forecast_evaluate_metrics(
    py: Python<'_>,
    forecast: &NativeForecastResult,
    actuals: Vec<(String, String, usize, f64)>,
    training_actuals: Option<Vec<(String, String, usize, f64)>>,
    mase_seasonality: Option<usize>,
) -> PyResult<NativeForecastMetricSet> {
    NativeForecastMetricSet::evaluate(py, forecast, actuals, training_actuals, mase_seasonality)
}

#[pyclass(name = "BacktestFoldResult")]
#[derive(Clone, Debug)]
struct NativeBacktestFoldResult {
    result: CoreBacktestFoldResult,
}

#[pymethods]
impl NativeBacktestFoldResult {
    #[getter]
    fn fold(&self) -> NativeForecastFold {
        NativeForecastFold {
            fold: self.result.fold.clone(),
        }
    }

    #[getter]
    fn metrics(&self) -> NativeForecastMetricSet {
        NativeForecastMetricSet {
            metrics: self.result.metrics.clone(),
        }
    }

    #[getter]
    fn predictions(&self) -> Vec<(String, String, usize, String, f64)> {
        self.result
            .predictions
            .iter()
            .map(forecast_prediction_tuple)
            .collect()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "BacktestResult")]
#[derive(Clone, Debug)]
struct NativeBacktestResult {
    result: CoreBacktestResult,
}

#[pymethods]
impl NativeBacktestResult {
    #[getter]
    fn folds(&self) -> Vec<NativeBacktestFoldResult> {
        self.result
            .folds
            .iter()
            .cloned()
            .map(|result| NativeBacktestFoldResult { result })
            .collect()
    }

    #[getter]
    fn metrics(&self) -> Option<NativeForecastMetricSet> {
        self.result
            .metrics
            .clone()
            .map(|metrics| NativeForecastMetricSet { metrics })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.result).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "RollingOriginBacktester")]
#[derive(Clone, Debug)]
struct NativeRollingOriginBacktester {
    backtester: CoreRollingOriginBacktester,
}

#[pymethods]
impl NativeRollingOriginBacktester {
    #[new]
    #[pyo3(signature = (splitter, mase_seasonality=None))]
    fn new(
        splitter: &NativeRollingOriginSplitter,
        mase_seasonality: Option<usize>,
    ) -> PyResult<Self> {
        let mut backtester = CoreRollingOriginBacktester::new(splitter.splitter.clone());
        if let Some(seasonality) = mase_seasonality {
            backtester = backtester
                .with_mase_seasonality(seasonality)
                .map_err(to_py_value_error)?;
        }
        Ok(Self { backtester })
    }

    #[getter]
    fn splitter(&self) -> NativeRollingOriginSplitter {
        NativeRollingOriginSplitter {
            splitter: self.backtester.splitter.clone(),
        }
    }

    #[getter]
    fn mase_seasonality(&self) -> Option<usize> {
        self.backtester.mase_seasonality
    }

    fn run_naive(
        &self,
        py: Python<'_>,
        model: &NativeNaiveForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_seasonal_naive(
        &self,
        py: Python<'_>,
        model: &NativeSeasonalNaiveForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_theta(
        &self,
        py: Python<'_>,
        model: &NativeThetaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_optimized_theta(
        &self,
        py: Python<'_>,
        model: &NativeOptimizedThetaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_ets(
        &self,
        py: Python<'_>,
        model: &NativeETSForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_arima(
        &self,
        py: Python<'_>,
        model: &NativeArimaForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_auto_arima(
        &self,
        py: Python<'_>,
        model: &NativeAutoARIMAForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }

    fn run_cartoboost_lag(
        &self,
        py: Python<'_>,
        model: &NativeCartoBoostLagForecaster,
        frame: &NativeForecastFrame,
    ) -> PyResult<NativeBacktestResult> {
        backtest_to_py(py.allow_threads(|| self.backtester.run(model.model.clone(), &frame.frame)))
    }
}

#[pyfunction]
fn forecast_parse_frequency(value: &str) -> PyResult<String> {
    Ok(ForecastFrequency::parse(value)
        .map_err(to_py_value_error)?
        .as_str()
        .to_string())
}

#[pyfunction]
#[pyo3(signature = (values, level_process_variance=0.05, trend_process_variance=0.005, observation_variance=1.0, horizon=0, interval_z=1.959963984540054))]
fn utility_kalman_filter(
    py: Python<'_>,
    values: Vec<f64>,
    level_process_variance: f64,
    trend_process_variance: f64,
    observation_variance: f64,
    horizon: usize,
    interval_z: f64,
) -> PyResult<String> {
    let config = LocalLinearKalmanConfig::new(
        level_process_variance,
        trend_process_variance,
        observation_variance,
    )
    .map_err(to_py_value_error)?;
    let (result, forecast, forecast_distribution) = py
        .allow_threads(|| {
            let result = fit_local_linear_kalman(&values, config)?;
            let forecast = if horizon == 0 {
                Vec::new()
            } else {
                local_linear_kalman_forecast(result.final_state, horizon)?
            };
            let forecast_distribution = if horizon == 0 {
                Vec::new()
            } else {
                local_linear_kalman_forecast_distribution(
                    result.final_state,
                    result.final_covariance,
                    config,
                    horizon,
                    interval_z,
                )?
            };
            Ok((result, forecast, forecast_distribution))
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "final_state": {
            "level": result.final_state.level,
            "trend": result.final_state.trend,
            "covariance": result.final_covariance,
        },
        "estimates": result.estimates.iter().map(|estimate| {
            json!({
                "step": estimate.step,
                "observed": estimate.observed,
                "prior_level": estimate.prior_level,
                "prior_trend": estimate.prior_trend,
                "prior_level_variance": estimate.prior_level_variance,
                "prior_trend_variance": estimate.prior_trend_variance,
                "prior_covariance": estimate.prior_covariance,
                "level": estimate.level,
                "trend": estimate.trend,
                "level_variance": estimate.level_variance,
                "trend_variance": estimate.trend_variance,
                "covariance": estimate.covariance,
                "fitted": estimate.prior_level,
                "residual": estimate.innovation,
                "innovation": estimate.innovation,
                "innovation_variance": estimate.innovation_variance,
                "standardized_innovation": estimate.innovation / estimate.innovation_variance.sqrt(),
                "level_gain": estimate.level_gain,
                "trend_gain": estimate.trend_gain,
                "log_likelihood": estimate.log_likelihood,
            })
        }).collect::<Vec<_>>(),
        "smoothed_states": result.smoothed_states.iter().map(|state| {
            json!({
                "step": state.step,
                "level": state.level,
                "trend": state.trend,
                "covariance": state.covariance,
            })
        }).collect::<Vec<_>>(),
        "forecast": forecast,
        "forecast_distribution": forecast_distribution.iter().map(|point| {
            json!({
                "step": point.step,
                "mean": point.mean,
                "variance": point.variance,
                "lower": point.lower,
                "upper": point.upper,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "log_likelihood": result.log_likelihood,
            "interval_z": interval_z,
            "observation_count": result.residual_summary.observation_count,
            "fitted_count": result.residual_summary.fitted_count,
            "aic": result.residual_summary.aic,
            "bic": result.residual_summary.bic,
            "mse": result.residual_summary.mse,
            "rmse": result.residual_summary.rmse,
            "mae": result.residual_summary.mae,
            "mean_standardized_innovation": result.residual_summary.mean_standardized_innovation,
            "max_abs_standardized_innovation": result.residual_summary.max_abs_standardized_innovation,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (values, level_process_variance=0.05, observation_variance=1.0, horizon=0, interval_z=1.959963984540054))]
fn utility_local_level_kalman_filter(
    py: Python<'_>,
    values: Vec<f64>,
    level_process_variance: f64,
    observation_variance: f64,
    horizon: usize,
    interval_z: f64,
) -> PyResult<String> {
    let config = LocalLevelKalmanConfig::new(level_process_variance, observation_variance)
        .map_err(to_py_value_error)?;
    let (result, forecast, forecast_distribution) = py
        .allow_threads(|| {
            let result = fit_local_level_kalman(&values, config)?;
            let forecast = if horizon == 0 {
                Vec::new()
            } else {
                local_level_kalman_forecast(result.final_level, horizon)?
            };
            let forecast_distribution = if horizon == 0 {
                Vec::new()
            } else {
                local_level_kalman_forecast_distribution(
                    result.final_level,
                    result.final_variance,
                    config,
                    horizon,
                    interval_z,
                )?
            };
            Ok((result, forecast, forecast_distribution))
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "final_state": {
            "level": result.final_level,
            "variance": result.final_variance,
        },
        "estimates": result.estimates.iter().map(|estimate| {
            json!({
                "step": estimate.step,
                "observed": estimate.observed,
                "prior_level": estimate.prior_level,
                "prior_variance": estimate.prior_variance,
                "level": estimate.level,
                "variance": estimate.variance,
                "fitted": estimate.prior_level,
                "residual": estimate.innovation,
                "innovation": estimate.innovation,
                "innovation_variance": estimate.innovation_variance,
                "standardized_innovation": estimate.innovation / estimate.innovation_variance.sqrt(),
                "gain": estimate.gain,
                "log_likelihood": estimate.log_likelihood,
            })
        }).collect::<Vec<_>>(),
        "smoothed_states": result.smoothed_states.iter().map(|state| {
            json!({
                "step": state.step,
                "level": state.level,
                "variance": state.variance,
            })
        }).collect::<Vec<_>>(),
        "forecast": forecast,
        "forecast_distribution": forecast_distribution.iter().map(|point| {
            json!({
                "step": point.step,
                "mean": point.mean,
                "variance": point.variance,
                "lower": point.lower,
                "upper": point.upper,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "log_likelihood": result.log_likelihood,
            "interval_z": interval_z,
            "observation_count": result.residual_summary.observation_count,
            "fitted_count": result.residual_summary.fitted_count,
            "aic": result.residual_summary.aic,
            "bic": result.residual_summary.bic,
            "mse": result.residual_summary.mse,
            "rmse": result.residual_summary.rmse,
            "mae": result.residual_summary.mae,
            "mean_standardized_innovation": result.residual_summary.mean_standardized_innovation,
            "max_abs_standardized_innovation": result.residual_summary.max_abs_standardized_innovation,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (values, horizon, method="croston", alpha=0.1, beta=0.1))]
fn utility_intermittent_demand_forecast(
    py: Python<'_>,
    values: Vec<f64>,
    horizon: usize,
    method: &str,
    alpha: f64,
    beta: f64,
) -> PyResult<Vec<f64>> {
    let method = match method {
        "croston" => IntermittentDemandMethod::Croston,
        "sba" => IntermittentDemandMethod::Sba,
        "tsb" => IntermittentDemandMethod::Tsb,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported intermittent demand method {other:?}"
            )));
        }
    };
    py.allow_threads(|| intermittent_demand_forecast(&values, horizon, alpha, beta, method))
        .map_err(to_py_value_error)
}

#[pyfunction]
#[pyo3(signature = (observations, targets, range=1.0, nugget=1.0e-6))]
fn utility_ordinary_kriging_predict(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    targets: Vec<(f64, f64)>,
    range: f64,
    nugget: f64,
) -> PyResult<Vec<PyKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = OrdinaryKrigingConfig::new(range, nugget).map_err(to_py_value_error)?;
    let predictions = py
        .allow_threads(|| ordinary_kriging_predict_many(&observations, &targets, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.weights,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    targets,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_predict_detailed(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    targets: Vec<(f64, f64)>,
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<Vec<PyDetailedKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let predictions = py
        .allow_threads(|| ordinary_kriging_predict_many(&observations, &targets, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.variance,
                prediction.weights,
                prediction.neighbor_indices,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_leave_one_out(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<Vec<PyDetailedKrigingPrediction>> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let predictions = py
        .allow_threads(|| ordinary_kriging_leave_one_out(&observations, config))
        .map_err(to_py_value_error)?;
    Ok(predictions
        .into_iter()
        .map(|prediction| {
            (
                prediction.x,
                prediction.y,
                prediction.mean,
                prediction.variance,
                prediction.weights,
                prediction.neighbor_indices,
            )
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    bin_count=10,
    max_distance=None,
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0
))]
fn utility_empirical_variogram(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let bins = py
        .allow_threads(|| {
            empirical_variogram(
                &observations,
                bin_count,
                max_distance,
                anisotropy_angle_degrees,
                anisotropy_scaling,
            )
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "bins": bins.iter().map(|bin| {
            json!({
                "lag_min": bin.lag_min,
                "lag_max": bin.lag_max,
                "lag_center": bin.lag_center,
                "mean_distance": bin.mean_distance,
                "semivariance": bin.semivariance,
                "pair_count": bin.pair_count,
            })
        }).collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    variogram_models=None,
    range_candidates=None,
    nugget_candidates=None,
    sill_candidates=None,
    bin_count=10,
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0
))]
#[allow(clippy::too_many_arguments)]
fn utility_fit_ordinary_kriging_variogram(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    variogram_models: Option<Vec<String>>,
    range_candidates: Option<Vec<f64>>,
    nugget_candidates: Option<Vec<f64>>,
    sill_candidates: Option<Vec<f64>>,
    bin_count: usize,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let models = variogram_models
        .unwrap_or_default()
        .iter()
        .map(|model| parse_kriging_variogram_model(model))
        .collect::<PyResult<Vec<_>>>()?;
    let ranges = range_candidates.unwrap_or_default();
    let nuggets = nugget_candidates.unwrap_or_default();
    let sills = sill_candidates.unwrap_or_default();
    let fit = py
        .allow_threads(|| {
            fit_ordinary_kriging_variogram(
                &observations,
                &models,
                &ranges,
                &nuggets,
                &sills,
                bin_count,
                anisotropy_angle_degrees,
                anisotropy_scaling,
            )
        })
        .map_err(to_py_value_error)?;
    let payload = json!({
        "config": kriging_config_json(fit.config),
        "weighted_sse": fit.weighted_sse,
        "bins": fit.bins.iter().map(|bin| {
            json!({
                "lag_min": bin.lag_min,
                "lag_max": bin.lag_max,
                "lag_center": bin.lag_center,
                "mean_distance": bin.mean_distance,
                "semivariance": bin.semivariance,
                "pair_count": bin.pair_count,
            })
        }).collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    observations,
    range=1.0,
    nugget=1.0e-6,
    sill=1.0,
    variogram_model="exponential",
    drift="ordinary",
    anisotropy_angle_degrees=0.0,
    anisotropy_scaling=1.0,
    max_neighbors=None,
    min_neighbors=1,
    max_distance=None
))]
#[allow(clippy::too_many_arguments)]
fn utility_ordinary_kriging_leave_one_out_diagnostics(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<String> {
    let observations = observations
        .into_iter()
        .map(|(x, y, value)| KrigingObservation { x, y, value })
        .collect::<Vec<_>>();
    let config = build_kriging_config(
        range,
        nugget,
        sill,
        variogram_model,
        drift,
        anisotropy_angle_degrees,
        anisotropy_scaling,
        max_neighbors,
        min_neighbors,
        max_distance,
    )?;
    let (predictions, diagnostics) = py
        .allow_threads(|| ordinary_kriging_leave_one_out_diagnostics(&observations, config))
        .map_err(to_py_value_error)?;
    let payload = json!({
        "predictions": predictions.iter().map(|prediction| {
            json!({
                "x": prediction.x,
                "y": prediction.y,
                "mean": prediction.mean,
                "variance": prediction.variance,
                "weights": prediction.weights,
                "neighbor_indices": prediction.neighbor_indices,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": {
            "observation_count": diagnostics.observation_count,
            "mean_error": diagnostics.mean_error,
            "mae": diagnostics.mae,
            "rmse": diagnostics.rmse,
            "mean_standardized_error": diagnostics.mean_standardized_error,
            "rmse_standardized_error": diagnostics.rmse_standardized_error,
            "max_abs_standardized_error": diagnostics.max_abs_standardized_error,
            "interval_coverage_95": diagnostics.interval_coverage_95,
            "average_variance": diagnostics.average_variance,
        },
    });
    serde_json::to_string(&payload).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (model, values, horizon, params_json=None))]
fn utility_series_forecast(
    py: Python<'_>,
    model: &str,
    values: Vec<f64>,
    horizon: usize,
    params_json: Option<&str>,
) -> PyResult<Vec<f64>> {
    let params = match params_json {
        Some(raw) => serde_json::from_str::<Value>(raw).map_err(|err| {
            PyValueError::new_err(format!("params_json must be valid JSON: {err}"))
        })?,
        None => json!({}),
    };
    let frame = utility_frame_from_values(values).map_err(to_py_value_error)?;
    let mut forecaster = utility_forecaster(model, &params).map_err(to_py_value_error)?;
    let result = py
        .allow_threads(|| {
            forecaster.fit(&frame)?;
            forecaster.predict(horizon)
        })
        .map_err(to_py_value_error)?;
    Ok(result
        .predictions()
        .iter()
        .map(|prediction| prediction.mean)
        .collect())
}

fn utility_frame_from_values(values: Vec<f64>) -> cartoboost_core::Result<CoreForecastFrame> {
    let frequency = ForecastFrequency::Daily;
    let start = cartoboost_core::forecasting::parse_forecast_timestamp("1970-01-01")?;
    let rows = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            Ok(CoreForecastRow::single(
                frequency.advance(start, index)?,
                value,
            ))
        })
        .collect::<cartoboost_core::Result<Vec<_>>>()?;
    CoreForecastFrame::new(rows, frequency)
}

fn utility_forecaster(model: &str, params: &Value) -> cartoboost_core::Result<Box<dyn Forecaster>> {
    match model {
        "naive" => Ok(Box::new(CoreNaiveForecaster::new())),
        "seasonal_naive" | "seasonal-naive" => {
            let season_length = utility_usize_param(params, "season_length")?.unwrap_or(1);
            Ok(Box::new(CoreSeasonalNaiveForecaster::new(season_length)?))
        }
        "theta" => {
            let theta = utility_f64_param(params, "theta")?.unwrap_or(2.0);
            let alpha = utility_f64_param(params, "alpha")?.unwrap_or(0.5);
            Ok(Box::new(CoreThetaForecaster::new(theta, alpha)?))
        }
        "optimized_theta" | "optimized-theta" => {
            let theta_grid =
                utility_f64_vec_param(params, "theta_grid")?.unwrap_or_else(|| vec![1.0, 2.0]);
            let alpha_grid =
                utility_f64_vec_param(params, "alpha_grid")?.unwrap_or_else(|| vec![0.2, 0.5, 0.8]);
            Ok(Box::new(CoreOptimizedThetaForecaster::new(
                theta_grid, alpha_grid,
            )?))
        }
        "ets" => {
            let alpha = utility_f64_param(params, "alpha")?.unwrap_or(0.5);
            let beta = utility_f64_param(params, "beta")?.unwrap_or(0.1);
            let gamma = utility_f64_param(params, "gamma")?;
            let season_length = utility_usize_param(params, "season_length")?;
            Ok(Box::new(CoreETSForecaster::with_additive_seasonality(
                alpha,
                beta,
                gamma,
                season_length,
            )?))
        }
        "arima" => {
            let p = utility_usize_param(params, "p")?.unwrap_or(1);
            let d = utility_usize_param(params, "d")?.unwrap_or(0);
            let q = utility_usize_param(params, "q")?.unwrap_or(0);
            Ok(Box::new(CoreArimaForecaster::new(p, d, q)?))
        }
        "auto_arima" | "auto-arima" => {
            let max_p = utility_usize_param(params, "max_p")?.unwrap_or(3);
            let max_d = utility_usize_param(params, "max_d")?.unwrap_or(1);
            let max_q = utility_usize_param(params, "max_q")?.unwrap_or(2);
            Ok(Box::new(CoreAutoARIMAForecaster::with_max_order(
                max_p, max_d, max_q,
            )?))
        }
        "kalman" | "local_linear_trend_kalman" | "local-linear-trend-kalman" => {
            let level_process_variance =
                utility_f64_param(params, "level_process_variance")?.unwrap_or(0.05);
            let trend_process_variance =
                utility_f64_param(params, "trend_process_variance")?.unwrap_or(0.005);
            let observation_variance =
                utility_f64_param(params, "observation_variance")?.unwrap_or(1.0);
            Ok(Box::new(CoreKalmanForecaster::new(
                level_process_variance,
                trend_process_variance,
                observation_variance,
            )?))
        }
        "auto_kalman" | "self_tuning_kalman" | "self-tuning-kalman" => {
            let level_process_variance_grid =
                utility_f64_vec_param(params, "level_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]);
            let trend_process_variance_grid =
                utility_f64_vec_param(params, "trend_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.0001, 0.001, 0.005, 0.01]);
            let observation_variance_grid =
                utility_f64_vec_param(params, "observation_variance_grid")?
                    .unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]);
            let validation_window = utility_usize_param(params, "validation_window")?;
            Ok(Box::new(CoreAutoKalmanForecaster::with_grids(
                level_process_variance_grid,
                trend_process_variance_grid,
                observation_variance_grid,
                validation_window,
            )?))
        }
        "local_level_kalman" | "local-level-kalman" => {
            let level_process_variance =
                utility_f64_param(params, "level_process_variance")?.unwrap_or(0.05);
            let observation_variance =
                utility_f64_param(params, "observation_variance")?.unwrap_or(1.0);
            Ok(Box::new(CoreLocalLevelKalmanForecaster::new(
                level_process_variance,
                observation_variance,
            )?))
        }
        "auto_local_level_kalman" | "auto-local-level-kalman" => {
            let level_process_variance_grid =
                utility_f64_vec_param(params, "level_process_variance_grid")?
                    .unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]);
            let observation_variance_grid =
                utility_f64_vec_param(params, "observation_variance_grid")?
                    .unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]);
            let validation_window = utility_usize_param(params, "validation_window")?;
            Ok(Box::new(CoreAutoLocalLevelKalmanForecaster::with_grids(
                level_process_variance_grid,
                observation_variance_grid,
                validation_window,
            )?))
        }
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown utility series forecast model {other:?}"
        ))),
    }
}

fn utility_f64_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<f64>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_f64()
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!("parameter {name} must be numeric"))
            })
            .map(Some),
    }
}

fn utility_usize_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<usize>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => {
            let raw = value.as_u64().ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "parameter {name} must be a nonnegative integer"
                ))
            })?;
            usize::try_from(raw)
                .map_err(|_| {
                    CartoBoostError::InvalidInput(format!("parameter {name} is too large"))
                })
                .map(Some)
        }
    }
}

fn utility_f64_vec_param(params: &Value, name: &str) -> cartoboost_core::Result<Option<Vec<f64>>> {
    match params.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_f64().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "parameter {name} must contain only numbers"
                    ))
                })
            })
            .collect::<cartoboost_core::Result<Vec<_>>>()
            .map(Some),
        Some(_) => Err(CartoBoostError::InvalidInput(format!(
            "parameter {name} must be a numeric array"
        ))),
    }
}

#[pyclass(name = "NaiveForecaster")]
#[derive(Clone, Debug)]
struct NativeNaiveForecaster {
    model: CoreNaiveForecaster,
}

#[pymethods]
impl NativeNaiveForecaster {
    #[new]
    #[pyo3(signature = (prediction_interval_levels=None))]
    fn new(prediction_interval_levels: Option<Vec<f64>>) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        Ok(Self {
            model: CoreNaiveForecaster::new(),
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "SeasonalNaiveForecaster")]
#[derive(Clone, Debug)]
struct NativeSeasonalNaiveForecaster {
    model: CoreSeasonalNaiveForecaster,
}

#[pymethods]
impl NativeSeasonalNaiveForecaster {
    #[new]
    #[pyo3(signature = (season_length, prediction_interval_levels=None))]
    fn new(season_length: usize, prediction_interval_levels: Option<Vec<f64>>) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        Ok(Self {
            model: CoreSeasonalNaiveForecaster::new(season_length).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "ThetaForecaster")]
#[derive(Clone, Debug)]
struct NativeThetaForecaster {
    model: CoreThetaForecaster,
}

#[pymethods]
impl NativeThetaForecaster {
    #[new]
    #[pyo3(signature = (theta=2.0, alpha=0.2, season_length=None, seasonality=None, prediction_interval_levels=None))]
    fn new(
        theta: f64,
        alpha: f64,
        season_length: Option<usize>,
        seasonality: Option<String>,
        prediction_interval_levels: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let seasonality = parse_theta_seasonality(season_length, seasonality)?;
        Ok(Self {
            model: CoreThetaForecaster::with_seasonality(theta, alpha, seasonality)
                .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "OptimizedThetaForecaster")]
#[derive(Clone, Debug)]
struct NativeOptimizedThetaForecaster {
    model: CoreOptimizedThetaForecaster,
}

#[pymethods]
impl NativeOptimizedThetaForecaster {
    #[new]
    #[pyo3(signature = (theta_grid=None, alpha_grid=None, season_length=None, seasonality=None, prediction_interval_levels=None))]
    fn new(
        theta_grid: Option<Vec<f64>>,
        alpha_grid: Option<Vec<f64>>,
        season_length: Option<usize>,
        seasonality: Option<String>,
        prediction_interval_levels: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let seasonality = parse_theta_seasonality(season_length, seasonality)?;
        Ok(Self {
            model: CoreOptimizedThetaForecaster::with_seasonality(
                theta_grid.unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]),
                alpha_grid.unwrap_or_else(|| vec![0.1, 0.2, 0.4, 0.6, 0.8]),
                seasonality,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "ETSForecaster")]
#[derive(Clone, Debug)]
struct NativeETSForecaster {
    model: CoreETSForecaster,
}

#[pymethods]
impl NativeETSForecaster {
    #[new]
    #[pyo3(signature = (alpha=0.5, beta=0.1, gamma=None, season_length=None))]
    fn new(
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreETSForecaster::with_additive_seasonality(alpha, beta, gamma, season_length)
                .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn fitted_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.fitted_values(series_id),
            series_id,
            "fitted values",
        )
    }

    fn residuals(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(self.model.residuals(series_id), series_id, "residuals")
    }

    fn level_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.level_values(series_id),
            series_id,
            "level values",
        )
    }

    fn trend_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.trend_values(series_id),
            series_id,
            "trend values",
        )
    }

    fn seasonal_values(&self, series_id: &str) -> PyResult<Vec<f64>> {
        ets_diagnostic_values(
            self.model.seasonal_values(series_id),
            series_id,
            "seasonal values",
        )
    }
}

#[pyclass(name = "ArimaForecaster")]
#[derive(Clone, Debug)]
struct NativeArimaForecaster {
    model: CoreArimaForecaster,
}

#[pymethods]
impl NativeArimaForecaster {
    #[new]
    #[pyo3(signature = (p=1, d=0, q=0))]
    fn new(p: usize, d: usize, q: usize) -> PyResult<Self> {
        Ok(Self {
            model: CoreArimaForecaster::new(p, d, q).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoARIMAForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoARIMAForecaster {
    model: CoreAutoARIMAForecaster,
}

#[pymethods]
impl NativeAutoARIMAForecaster {
    #[new]
    #[pyo3(signature = (seasonal=false, m=1, error_policy="raise", max_p=3, max_d=1, max_q=2))]
    fn new(
        seasonal: bool,
        m: usize,
        error_policy: &str,
        max_p: usize,
        max_d: usize,
        max_q: usize,
    ) -> PyResult<Self> {
        if seasonal {
            return Err(PyValueError::new_err(
                "AutoARIMAForecaster Rust binding currently supports seasonal=false",
            ));
        }
        if m == 0 {
            return Err(PyValueError::new_err("m must be a positive integer"));
        }
        if error_policy != "raise" {
            return Err(PyValueError::new_err(
                "AutoARIMAForecaster Rust binding supports error_policy='raise' only",
            ));
        }
        Ok(Self {
            model: CoreAutoARIMAForecaster::with_max_order(max_p, max_d, max_q)
                .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "KalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeKalmanForecaster {
    model: CoreKalmanForecaster,
}

#[pymethods]
impl NativeKalmanForecaster {
    #[new]
    #[pyo3(signature = (level_process_variance=0.05, trend_process_variance=0.005, observation_variance=1.0))]
    fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreKalmanForecaster::new(
                level_process_variance,
                trend_process_variance,
                observation_variance,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "LocalLevelKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeLocalLevelKalmanForecaster {
    model: CoreLocalLevelKalmanForecaster,
}

#[pymethods]
impl NativeLocalLevelKalmanForecaster {
    #[new]
    #[pyo3(signature = (level_process_variance=0.05, observation_variance=1.0))]
    fn new(level_process_variance: f64, observation_variance: f64) -> PyResult<Self> {
        Ok(Self {
            model: CoreLocalLevelKalmanForecaster::new(
                level_process_variance,
                observation_variance,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoKalmanForecaster {
    model: CoreAutoKalmanForecaster,
}

#[pymethods]
impl NativeAutoKalmanForecaster {
    #[new]
    #[pyo3(signature = (
        level_process_variance_grid=None,
        trend_process_variance_grid=None,
        observation_variance_grid=None,
        validation_window=None
    ))]
    fn new(
        level_process_variance_grid: Option<Vec<f64>>,
        trend_process_variance_grid: Option<Vec<f64>>,
        observation_variance_grid: Option<Vec<f64>>,
        validation_window: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreAutoKalmanForecaster::with_grids(
                level_process_variance_grid.unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]),
                trend_process_variance_grid.unwrap_or_else(|| vec![0.0001, 0.001, 0.005, 0.01]),
                observation_variance_grid.unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]),
                validation_window,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "AutoLocalLevelKalmanForecaster")]
#[derive(Clone, Debug)]
struct NativeAutoLocalLevelKalmanForecaster {
    model: CoreAutoLocalLevelKalmanForecaster,
}

#[pymethods]
impl NativeAutoLocalLevelKalmanForecaster {
    #[new]
    #[pyo3(signature = (
        level_process_variance_grid=None,
        observation_variance_grid=None,
        validation_window=None
    ))]
    fn new(
        level_process_variance_grid: Option<Vec<f64>>,
        observation_variance_grid: Option<Vec<f64>>,
        validation_window: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            model: CoreAutoLocalLevelKalmanForecaster::with_grids(
                level_process_variance_grid.unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]),
                observation_variance_grid.unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]),
                validation_window,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "KrigingForecaster")]
#[derive(Clone, Debug)]
struct NativeKrigingForecaster {
    model: CoreKrigingForecaster,
}

#[pymethods]
impl NativeKrigingForecaster {
    #[new]
    #[pyo3(signature = (
        coordinates,
        range=1.0,
        nugget=1.0e-9,
        sill=1.0,
        variogram_model="exponential",
        drift="ordinary",
        anisotropy_angle_degrees=0.0,
        anisotropy_scaling=1.0,
        max_neighbors=None,
        min_neighbors=1,
        max_distance=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        coordinates: Vec<(String, f64, f64)>,
        range: f64,
        nugget: f64,
        sill: f64,
        variogram_model: &str,
        drift: &str,
        anisotropy_angle_degrees: f64,
        anisotropy_scaling: f64,
        max_neighbors: Option<usize>,
        min_neighbors: usize,
        max_distance: Option<f64>,
    ) -> PyResult<Self> {
        let coordinates = coordinates
            .into_iter()
            .map(|(series_id, x, y)| (series_id, (x, y)))
            .collect::<BTreeMap<_, _>>();
        let config = build_kriging_config(
            range,
            nugget,
            sill,
            variogram_model,
            drift,
            anisotropy_angle_degrees,
            anisotropy_scaling,
            max_neighbors,
            min_neighbors,
            max_distance,
        )?;
        Ok(Self {
            model: CoreKrigingForecaster::with_config(coordinates, config)
                .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(name = "CartoBoostLagForecaster")]
#[derive(Clone, Debug)]
struct NativeCartoBoostLagForecaster {
    model: CoreCartoBoostLagForecaster,
}

#[pymethods]
impl NativeCartoBoostLagForecaster {
    #[new]
    #[pyo3(signature = (lags=None, rolling_windows=None, difference_lags=None, rolling_trend_windows=None, calendar_features=true, recursive=true, prediction_interval_levels=None, n_estimators=None, learning_rate=None, max_depth=None, min_samples_leaf=None, min_gain=None, splitters=None, trend_features=true, target_mode="level"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lags: Option<Vec<usize>>,
        rolling_windows: Option<Vec<usize>>,
        difference_lags: Option<Vec<usize>>,
        rolling_trend_windows: Option<Vec<usize>>,
        calendar_features: bool,
        recursive: bool,
        prediction_interval_levels: Option<Vec<f64>>,
        n_estimators: Option<usize>,
        learning_rate: Option<f64>,
        max_depth: Option<usize>,
        min_samples_leaf: Option<usize>,
        min_gain: Option<f64>,
        splitters: Option<Vec<String>>,
        trend_features: bool,
        target_mode: &str,
    ) -> PyResult<Self> {
        if !recursive {
            return Err(PyValueError::new_err(
                "CartoBoostLagForecaster currently supports recursive=true only",
            ));
        }
        validate_interval_levels(prediction_interval_levels.as_deref())?;
        let lags = lags.unwrap_or_else(|| vec![1, 7, 14]);
        let rolling_mean_windows = rolling_windows.unwrap_or_else(|| vec![7, 28]);
        let difference_lags = match difference_lags {
            Some(values) => values,
            None if trend_features => lags.iter().copied().filter(|lag| *lag > 1).collect(),
            None => Vec::new(),
        };
        let rolling_trend_windows = match rolling_trend_windows {
            Some(values) => values,
            None if trend_features => rolling_mean_windows
                .iter()
                .copied()
                .filter(|window| *window > 1)
                .collect(),
            None => Vec::new(),
        };
        let config = LagFeatureConfig {
            difference_lags,
            rolling_trend_windows,
            lags,
            rolling_mean_windows,
            calendar_features: if calendar_features {
                vec![
                    CalendarFeature::DayOfWeek,
                    CalendarFeature::Month,
                    CalendarFeature::Day,
                ]
            } else {
                Vec::new()
            },
        };
        let mut booster_config = BoosterConfig::default();
        if let Some(value) = n_estimators {
            booster_config.n_estimators = value;
        }
        if let Some(value) = learning_rate {
            booster_config.learning_rate = value;
        }
        if let Some(value) = max_depth {
            booster_config.max_depth = value;
        }
        if let Some(value) = min_samples_leaf {
            booster_config.min_samples_leaf = value;
        }
        if let Some(value) = min_gain {
            booster_config.min_gain = value;
        }
        if let Some(values) = splitters {
            booster_config.splitters = parse_splitters(&values)?;
        }
        validate_params(
            booster_config.n_estimators,
            booster_config.learning_rate,
            booster_config.max_depth,
            booster_config.min_samples_leaf,
            booster_config.min_gain,
            booster_config.linear_lambda_l2,
            booster_config.constant_lambda_l2,
            booster_config.fuzzy_bandwidth,
            0.5,
            1.0,
            1.0,
        )?;
        let target_mode = parse_global_target_mode(target_mode)?;
        Ok(Self {
            model: CoreCartoBoostLagForecaster::new_with_target_mode(
                config,
                booster_config,
                target_mode,
            )
            .map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata()).map_err(|err| {
            PyValueError::new_err(format!("failed to serialize forecaster metadata: {err}"))
        })
    }
}

#[pyclass(name = "WeightedEnsembleForecaster", unsendable)]
struct NativeWeightedEnsembleForecaster {
    model: CoreWeightedEnsembleForecaster,
}

#[pymethods]
impl NativeWeightedEnsembleForecaster {
    #[new]
    fn new(py: Python<'_>, members: Vec<(String, Py<PyAny>, f64)>) -> PyResult<Self> {
        let members = members
            .iter()
            .map(|(name, model, weight)| {
                Ok((name.clone(), boxed_forecaster_from_py(py, model)?, *weight))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            model: CoreWeightedEnsembleForecaster::new(members).map_err(to_py_value_error)?,
        })
    }

    fn fit(&mut self, py: Python<'_>, frame: &NativeForecastFrame) -> PyResult<()> {
        fit_forecaster_py(py, &mut self.model, frame)
    }

    fn predict(&self, py: Python<'_>, horizon: usize) -> PyResult<NativeForecastResult> {
        predict_forecaster_py(py, &self.model, horizon)
    }

    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.model.metadata())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

fn boxed_forecaster_from_py(py: Python<'_>, model: &Py<PyAny>) -> PyResult<Box<dyn Forecaster>> {
    let model = model.bind(py);
    if let Ok(model) = model.extract::<PyRef<'_, NativeNaiveForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeSeasonalNaiveForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeThetaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeOptimizedThetaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeETSForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeArimaForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeAutoARIMAForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeKalmanForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    if let Ok(model) = model.extract::<PyRef<'_, NativeCartoBoostLagForecaster>>() {
        return Ok(Box::new(model.model.clone()));
    }
    Err(PyValueError::new_err(
        "WeightedEnsembleForecaster members must be native forecasting models",
    ))
}

fn forecast_to_py(
    result: cartoboost_core::Result<CoreForecastResult>,
) -> PyResult<NativeForecastResult> {
    Ok(NativeForecastResult {
        result: result.map_err(to_py_value_error)?,
    })
}

fn fit_forecaster_py<M: Forecaster>(
    py: Python<'_>,
    model: &mut M,
    frame: &NativeForecastFrame,
) -> PyResult<()> {
    py.allow_threads(|| model.fit(&frame.frame))
        .map_err(to_py_value_error)
}

fn predict_forecaster_py<M: Forecaster>(
    py: Python<'_>,
    model: &M,
    horizon: usize,
) -> PyResult<NativeForecastResult> {
    forecast_to_py(py.allow_threads(|| model.predict(horizon)))
}

#[allow(dead_code)]
fn ets_diagnostic_values(
    values: Option<&[f64]>,
    series_id: &str,
    name: &str,
) -> PyResult<Vec<f64>> {
    values.map(|values| values.to_vec()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "ETS {name} are unavailable for series {series_id:?}; fit the model and check the series id"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn build_kriging_config(
    range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str,
    drift: &str,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
    max_neighbors: Option<usize>,
    min_neighbors: usize,
    max_distance: Option<f64>,
) -> PyResult<OrdinaryKrigingConfig> {
    let variogram_model = parse_kriging_variogram_model(variogram_model)?;
    let drift = parse_kriging_drift(drift)?;
    OrdinaryKrigingConfig::new(range, nugget)
        .and_then(|config| config.with_sill(sill))
        .and_then(|config| config.with_anisotropy(anisotropy_angle_degrees, anisotropy_scaling))
        .and_then(|config| config.with_neighbor_limits(max_neighbors, min_neighbors, max_distance))
        .map(|config| {
            config
                .with_variogram_model(variogram_model)
                .with_drift(drift)
        })
        .map_err(to_py_value_error)
}

fn parse_kriging_variogram_model(value: &str) -> PyResult<KrigingVariogramModel> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "exponential" | "exp" => Ok(KrigingVariogramModel::Exponential),
        "gaussian" | "gauss" => Ok(KrigingVariogramModel::Gaussian),
        "spherical" | "sphere" => Ok(KrigingVariogramModel::Spherical),
        "linear" => Ok(KrigingVariogramModel::Linear),
        other => Err(PyValueError::new_err(format!(
            "unsupported kriging variogram_model {other:?}; expected exponential, gaussian, spherical, or linear"
        ))),
    }
}

fn parse_kriging_drift(value: &str) -> PyResult<KrigingDrift> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "ordinary" | "constant" | "none" => Ok(KrigingDrift::Ordinary),
        "linear" | "universal_linear" | "universal" => Ok(KrigingDrift::Linear),
        other => Err(PyValueError::new_err(format!(
            "unsupported kriging drift {other:?}; expected ordinary or linear"
        ))),
    }
}

fn kriging_config_json(config: OrdinaryKrigingConfig) -> Value {
    json!({
        "range": config.range,
        "nugget": config.nugget,
        "sill": config.sill,
        "variogram_model": format!("{:?}", config.variogram_model).to_lowercase(),
        "drift": format!("{:?}", config.drift).to_lowercase(),
        "anisotropy_angle_degrees": config.anisotropy_angle_degrees,
        "anisotropy_scaling": config.anisotropy_scaling,
        "max_neighbors": config.max_neighbors,
        "min_neighbors": config.min_neighbors,
        "max_distance": config.max_distance,
    })
}

fn backtest_to_py(
    result: cartoboost_core::Result<CoreBacktestResult>,
) -> PyResult<NativeBacktestResult> {
    Ok(NativeBacktestResult {
        result: result.map_err(to_py_value_error)?,
    })
}

fn parse_forecast_window(value: &str) -> PyResult<ForecastWindow> {
    match value {
        "expanding" => Ok(ForecastWindow::Expanding),
        "sliding" => Ok(ForecastWindow::Sliding),
        _ => Err(PyValueError::new_err(
            "forecast window must be 'expanding' or 'sliding'",
        )),
    }
}

fn forecast_window_name(window: &ForecastWindow) -> &'static str {
    match window {
        ForecastWindow::Expanding => "expanding",
        ForecastWindow::Sliding => "sliding",
    }
}

fn parse_forecast_actuals(
    actuals: Vec<(String, String, usize, f64)>,
) -> PyResult<Vec<ForecastActual>> {
    actuals
        .into_iter()
        .map(|(series_id, timestamp, horizon, actual)| {
            Ok(ForecastActual {
                series_id,
                timestamp: cartoboost_core::forecasting::parse_forecast_timestamp(&timestamp)
                    .map_err(to_py_value_error)?,
                horizon,
                actual,
            })
        })
        .collect()
}

fn forecast_prediction_tuple(
    prediction: &ForecastPrediction,
) -> (String, String, usize, String, f64) {
    (
        prediction.series_id.clone(),
        format_forecast_timestamp(prediction.timestamp),
        prediction.horizon,
        prediction.model.clone(),
        prediction.mean,
    )
}

fn format_forecast_timestamp(timestamp: impl std::fmt::Display) -> String {
    timestamp.to_string().replace(' ', "T")
}

fn validate_interval_levels(levels: Option<&[f64]>) -> PyResult<()> {
    for level in levels.unwrap_or(&[]) {
        if !level.is_finite() || *level <= 0.0 || *level >= 1.0 {
            return Err(PyValueError::new_err(
                "prediction interval levels must be finite values between 0 and 1",
            ));
        }
    }
    Ok(())
}

fn parse_theta_seasonality(
    season_length: Option<usize>,
    seasonality: Option<String>,
) -> PyResult<Option<ThetaSeasonality>> {
    let Some(mode) = seasonality else {
        return Ok(None);
    };
    let season_length = season_length.ok_or_else(|| {
        PyValueError::new_err("season_length is required when seasonality is set")
    })?;
    match mode.as_str() {
        "additive" => ThetaSeasonality::additive(season_length)
            .map(Some)
            .map_err(to_py_value_error),
        "multiplicative" => ThetaSeasonality::multiplicative(season_length)
            .map(Some)
            .map_err(to_py_value_error),
        _ => Err(PyValueError::new_err(
            "seasonality must be 'additive' or 'multiplicative'",
        )),
    }
}

#[pyclass(name = "CartoBoostRegressor")]
#[derive(Clone, Debug)]
struct NativeCartoBoostRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    loss: String,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    fuzzy_kernel: String,
    n_threads: Option<usize>,
    monotonic_constraints: Vec<i8>,
    model: Option<Model>,
    flat_axis_predictor: Option<FlatAxisPredictor>,
}

#[pymethods]
impl NativeCartoBoostRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.05, max_depth=4, min_samples_leaf=20, min_gain=1e-8, loss="l2", quantile_alpha=0.5, huber_delta=1.0, log_offset=1.0, splitters=None, leaf_predictor="constant", linear_leaf_features=None, l2_regularization=1.0, constant_l2_regularization=0.0, fuzzy=false, fuzzy_bandwidth=0.0, fuzzy_kernel="linear", n_threads=None, monotonic_constraints=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        loss: &str,
        quantile_alpha: f64,
        huber_delta: f64,
        log_offset: f64,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        constant_l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
        fuzzy_kernel: &str,
        n_threads: Option<usize>,
        monotonic_constraints: Option<Vec<i8>>,
    ) -> PyResult<Self> {
        validate_n_threads(n_threads)?;
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            constant_l2_regularization,
            fuzzy_bandwidth,
            quantile_alpha,
            huber_delta,
            log_offset,
        )?;
        parse_loss(loss, quantile_alpha, huber_delta, log_offset)?;
        let splitters = splitters.unwrap_or_else(|| vec!["auto".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;
        parse_fuzzy_kernel(fuzzy_kernel)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            loss: loss.to_string(),
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel: fuzzy_kernel.to_string(),
            n_threads,
            monotonic_constraints: monotonic_constraints.unwrap_or_default(),
            model: None,
            flat_axis_predictor: None,
        })
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_sets=None, feature_schema_json=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_parts(x, sparse_sets, feature_schema_json)?;
        let splitters = parse_splitters(&self.splitters)?;
        let leaf_predictor = parse_leaf_predictor(&self.leaf_predictor)?;
        let config = self.booster_config(splitters, leaf_predictor)?;
        let n_threads = self.n_threads;
        let model = py
            .allow_threads(move || {
                run_with_optional_threads(n_threads, || {
                    Booster::new(config).fit(&dataset, &y, sample_weight.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.set_model(model);
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_offsets=None, sparse_ids=None, feature_schema_json=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit_arrays(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, feature_schema_json)?;
        let targets = y.as_slice()?.to_vec();
        let weights = sample_weight
            .map(|array| array.as_slice().map(|slice| slice.to_vec()))
            .transpose()?;
        let splitters = parse_splitters(&self.splitters)?;
        let leaf_predictor = parse_leaf_predictor(&self.leaf_predictor)?;
        let config = self.booster_config(splitters, leaf_predictor)?;
        let n_threads = self.n_threads;
        let model = py
            .allow_threads(move || {
                run_with_optional_threads(n_threads, || {
                    Booster::new(config).fit(&dataset, &targets, weights.as_deref())
                })
            })
            .map_err(to_py_value_error)?;
        self.set_model(model);
        Ok(())
    }

    #[pyo3(signature = (x, y, sparse_sets, feature_schema_json=None, sample_weight=None))]
    fn fit_mixed(
        &mut self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sparse_sets: Vec<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.fit(
            py,
            x,
            y,
            sample_weight,
            Some(sparse_sets),
            feature_schema_json,
        )
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| run_with_optional_threads(n_threads, || model.try_predict(&dataset)))
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_arrays<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        let n_threads = self.n_threads;
        let predictions = py
            .allow_threads(|| {
                run_with_optional_threads(n_threads, || {
                    if offsets.is_empty() && ids.is_empty() {
                        if let Some(predictor) = &self.flat_axis_predictor {
                            model.validate_dense_flat_prediction_inputs(rows, cols, values)?;
                            Ok(predictor.predict_flat(rows, cols, values))
                        } else {
                            model.try_predict_flat(rows, cols, values, &offsets, &ids)
                        }
                    } else {
                        model.try_predict_flat(rows, cols, values, &offsets, &ids)
                    }
                })
            })
            .map_err(to_py_value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    #[pyo3(signature = (x, sparse_sets))]
    fn predict_mixed(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Vec<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<f64>> {
        self.predict(py, x, Some(sparse_sets))
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict_additive(
        &self,
        py: Python<'_>,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        let n_threads = self.n_threads;
        py.allow_threads(|| {
            run_with_optional_threads(n_threads, || model.try_predict_additive(&dataset))
        })
        .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_additive_arrays(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        let n_threads = self.n_threads;
        py.allow_threads(|| {
            run_with_optional_threads(n_threads, || {
                model.try_predict_additive_flat(rows, cols, values, &offsets, &ids)
            })
        })
        .map_err(to_py_value_error)
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        py.allow_threads(|| model.save(path)).map_err(to_py_error)
    }

    fn save_weights(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        py.allow_threads(|| model.save_weights(path))
            .map_err(to_py_error)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Model::load(path))
            .map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[staticmethod]
    fn load_weights(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Model::load_weights(path))
            .map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    #[getter]
    fn min_gain(&self) -> f64 {
        self.min_gain
    }

    #[getter]
    fn splitters(&self) -> Vec<String> {
        self.splitters.clone()
    }

    #[getter]
    fn loss(&self) -> String {
        self.loss.clone()
    }

    #[getter]
    fn quantile_alpha(&self) -> f64 {
        self.quantile_alpha
    }

    #[getter]
    fn huber_delta(&self) -> f64 {
        self.huber_delta
    }

    #[getter]
    fn log_offset(&self) -> f64 {
        self.log_offset
    }

    #[getter]
    fn leaf_predictor(&self) -> String {
        self.leaf_predictor.clone()
    }

    #[getter]
    fn linear_leaf_features(&self) -> Vec<usize> {
        self.linear_leaf_features.clone()
    }

    #[getter]
    fn l2_regularization(&self) -> f64 {
        self.l2_regularization
    }

    #[getter]
    fn constant_l2_regularization(&self) -> f64 {
        self.constant_l2_regularization
    }

    #[getter]
    fn fuzzy(&self) -> bool {
        self.fuzzy
    }

    #[getter]
    fn fuzzy_bandwidth(&self) -> f64 {
        self.fuzzy_bandwidth
    }

    #[getter]
    fn fuzzy_kernel(&self) -> String {
        self.fuzzy_kernel.clone()
    }

    #[getter]
    fn n_threads(&self) -> Option<usize> {
        self.n_threads
    }

    #[getter]
    fn monotonic_constraints(&self) -> Vec<i8> {
        self.monotonic_constraints.clone()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.feature_count)
            .unwrap_or(0)
    }

    #[getter]
    fn requires_sparse_sets(&self) -> bool {
        self.model
            .as_ref()
            .map(Model::requires_sparse_sets)
            .unwrap_or(false)
    }

    #[getter]
    fn feature_schema_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.feature_schema.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.metadata.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_config_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.training_config.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

impl NativeCartoBoostRegressor {
    fn from_model(model: Model) -> PyResult<Self> {
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            loss,
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            monotonic_constraints,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                loss_name(&config.loss).to_string(),
                quantile_alpha(&config.loss),
                huber_delta(&config.loss),
                log_offset(&config.loss),
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.constant_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
                fuzzy_kernel_name(config.fuzzy_kernel).to_string(),
                config.monotonic_constraints,
            )
        } else {
            (
                1,
                1,
                0.0,
                "l2".to_string(),
                0.5,
                1.0,
                1.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                0.0,
                false,
                0.0,
                "linear".to_string(),
                Vec::new(),
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            loss,
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            n_threads: None,
            monotonic_constraints,
            model: Some(model),
            flat_axis_predictor: None,
        })
        .map(|mut regressor| {
            regressor.refresh_prediction_cache();
            regressor
        })
    }

    fn booster_config(
        &self,
        splitters: Vec<SplitterKind>,
        leaf_predictor: LeafPredictorKind,
    ) -> PyResult<BoosterConfig> {
        Ok(BoosterConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            loss: parse_loss(
                &self.loss,
                self.quantile_alpha,
                self.huber_delta,
                self.log_offset,
            )?,
            splitters,
            leaf_predictor,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            monotonic_constraints: self.monotonic_constraints.clone(),
        })
    }

    fn set_model(&mut self, model: Model) {
        self.model = Some(model);
        self.refresh_prediction_cache();
    }

    fn refresh_prediction_cache(&mut self) {
        self.flat_axis_predictor = self.model.as_ref().and_then(Model::flat_axis_predictor);
    }
}

fn parse_splitters(names: &[String]) -> PyResult<Vec<SplitterKind>> {
    let mut splitters = Vec::with_capacity(names.len());
    for name in names {
        let splitter = match name.as_str() {
            "auto" => SplitterKind::Auto,
            "axis" => SplitterKind::Axis,
            "axis_histogram" | "axis_hist" | "histogram" => {
                SplitterKind::AxisHistogram { bins: 64 }
            }
            "diagonal_2d" | "diagonal2d" => SplitterKind::Diagonal2D,
            "gaussian_2d" | "gaussian2d" | "radial" => SplitterKind::Gaussian2D,
            "periodic_time" | "periodic_24" => SplitterKind::Periodic { period: 24.0 },
            "sparse_set" | "sparse" => SplitterKind::SparseSet,
            _ => {
                if let Some(bins) = name
                    .strip_prefix("axis_histogram:")
                    .or_else(|| name.strip_prefix("axis_hist:"))
                    .and_then(|bins| bins.parse::<usize>().ok())
                    .filter(|bins| *bins >= 2)
                {
                    SplitterKind::AxisHistogram { bins }
                } else if let Some(period) = name
                    .strip_prefix("periodic:")
                    .and_then(|period| period.parse::<f64>().ok())
                    .filter(|period| period.is_finite() && *period > 0.0)
                {
                    SplitterKind::Periodic { period }
                } else {
                    return Err(PyValueError::new_err(format!(
                        "unknown splitter {name:?}; expected one of 'auto', 'axis', 'axis_histogram', \
                         'diagonal_2d', 'gaussian_2d', 'periodic_time', or 'sparse_set'"
                    )));
                }
            }
        };
        splitters.push(splitter);
    }
    if splitters.is_empty() {
        Ok(vec![SplitterKind::Auto])
    } else {
        Ok(splitters)
    }
}

fn parse_global_target_mode(name: &str) -> PyResult<GlobalForecastTargetMode> {
    match name {
        "level" => Ok(GlobalForecastTargetMode::Level),
        "delta_from_last" | "delta" => Ok(GlobalForecastTargetMode::DeltaFromLast),
        _ => Err(PyValueError::new_err(format!(
            "unknown CartoBoostLagForecaster target_mode {name:?}; expected 'level' or \
             'delta_from_last'"
        ))),
    }
}

#[pyclass(name = "NeuralEmbeddingFeatures")]
#[derive(Clone)]
struct NativeNeuralEmbeddingFeatures {
    dim: usize,
    fallback: ArtifactFallbackKind,
    random_state: Option<i64>,
    parent_resolution: Option<u8>,
    support_prior_strength: f64,
    table: Option<EmbeddingTable>,
}

#[pymethods]
impl NativeNeuralEmbeddingFeatures {
    #[new]
    #[pyo3(signature = (dim, fallback="global_mean_vector", random_state=None, parent_resolution=None, support_prior_strength=1.0))]
    fn new(
        dim: usize,
        fallback: &str,
        random_state: Option<i64>,
        parent_resolution: Option<u8>,
        support_prior_strength: f64,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be positive"));
        }
        if !support_prior_strength.is_finite() || support_prior_strength <= 0.0 {
            return Err(PyValueError::new_err(
                "support_prior_strength must be positive and finite",
            ));
        }

        let fallback = parse_embedding_fallback(fallback, parent_resolution)?;

        Ok(Self {
            dim,
            fallback,
            random_state,
            parent_resolution,
            support_prior_strength,
            table: None,
        })
    }

    #[pyo3(signature = (ids, target))]
    fn fit(
        &mut self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, u64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let ids = ids.as_slice()?.to_vec();
        let target: Vec<f32> = target
            .as_slice()?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect();
        let random_state = self.random_state.map(|value| value as u64);

        let table = py
            .allow_threads(|| {
                fit_embedding_table_with_options(
                    self.dim,
                    &ids,
                    &target,
                    self.fallback.clone(),
                    random_state,
                    self.support_prior_strength,
                )
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.table = Some(table);
        Ok(())
    }

    #[pyo3(signature = (ids, target))]
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, u64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let ids = ids.as_slice()?.to_vec();
        let target: Vec<f32> = target
            .as_slice()?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect();
        let random_state = self.random_state.map(|value| value as u64);
        let (table, block) = py
            .allow_threads(|| {
                let table = fit_embedding_table_with_options(
                    self.dim,
                    &ids,
                    &target,
                    self.fallback.clone(),
                    random_state,
                    self.support_prior_strength,
                )?;
                let block = table.encode_ids(&ids, "neural_embedding")?;
                Ok::<_, cartoboost_neural::NeuralError>((table, block))
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.table = Some(table);
        let mut output = Vec::with_capacity(ids.len());
        for row in block.values.chunks_exact(block.dim) {
            output.push(row.to_vec());
        }
        Ok(output)
    }

    #[pyo3(signature = (ids))]
    fn transform(&self, py: Python<'_>, ids: PyReadonlyArray1<'_, u64>) -> PyResult<Vec<Vec<f32>>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("transform called before fit or load"))?;

        let ids = ids.as_slice()?.to_vec();
        let block = py
            .allow_threads(|| table.encode_ids(&ids, "neural_embedding"))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut output = Vec::with_capacity(ids.len());
        for row in block.values.chunks_exact(block.dim) {
            output.push(row.to_vec());
        }
        Ok(output)
    }

    #[pyo3(signature = (path))]
    fn export(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("export called before fit or load"))?;

        py.allow_threads(|| {
            let artifact = build_embedding_table_artifact(
                self.dim,
                table.rows().to_vec(),
                table.artifact_metadata().fallback.clone(),
            )?;
            write_embedding_table_artifact(path, &artifact)
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[classmethod]
    fn from_artifact(_cls: &Bound<'_, PyType>, py: Python<'_>, path: String) -> PyResult<Self> {
        let table = py
            .allow_threads(|| EmbeddingTable::load(path))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let metadata = table.artifact_metadata().clone();
        let parent_resolution = match metadata.fallback {
            ArtifactFallbackKind::ParentCell { parent_resolution } => Some(parent_resolution),
            _ => None,
        };

        Ok(Self {
            dim: metadata.dim,
            fallback: metadata.fallback,
            random_state: None,
            parent_resolution,
            support_prior_strength: 1.0,
            table: Some(table),
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn fallback(&self) -> String {
        artifact_fallback_name(&self.fallback).to_string()
    }

    #[getter]
    fn random_state(&self) -> Option<i64> {
        self.random_state
    }

    #[getter]
    fn parent_resolution(&self) -> Option<u8> {
        self.parent_resolution
    }

    #[getter]
    fn support_prior_strength(&self) -> f64 {
        self.support_prior_strength
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.table.is_some()
    }

    fn artifact_rows(&self) -> PyResult<Vec<(u64, Vec<f32>)>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("artifact_rows called before fit or load"))?;
        Ok(table
            .rows()
            .iter()
            .map(|row| (row.id, row.values.clone()))
            .collect())
    }
}

fn parse_embedding_fallback(
    value: &str,
    parent_resolution: Option<u8>,
) -> PyResult<ArtifactFallbackKind> {
    match value {
        "zero_vector" => Ok(ArtifactFallbackKind::ZeroVector),
        "global_mean_vector" => Ok(ArtifactFallbackKind::GlobalMeanVector),
        "parent_cell" => parent_resolution
            .map(|parent_resolution| ArtifactFallbackKind::ParentCell { parent_resolution })
            .ok_or_else(|| PyValueError::new_err("parent_resolution is required for parent_cell")),
        _ => Err(PyValueError::new_err(
            "fallback must be one of zero_vector, global_mean_vector, parent_cell",
        )),
    }
}

#[pyclass(name = "GraphSageEncoder")]
#[derive(Clone)]
struct NativeGraphSageEncoder {
    config: GraphSageConfig,
    encoder: GraphSageEncoder,
}

#[pymethods]
impl NativeGraphSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
        };

        let encoder =
            GraphSageEncoder::new(config.clone(), input_dim).map_err(to_py_neural_error)?;

        Ok(Self { config, encoder })
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let graph = HomogeneousGraph::from_directed_edges(node_count, &edges)
            .map_err(to_py_neural_error)?;
        let mut model = GraphSageEncoder::new(self.config.clone(), self.encoder.input_dim())
            .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .allow_threads(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let graph = HomogeneousGraph::from_directed_edges(node_count, &edges)
            .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .allow_threads(|| GraphSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self { encoder, config })
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }

    #[getter]
    fn config_seed(&self) -> u64 {
        self.config.seed
    }

    #[getter]
    fn config_epochs(&self) -> usize {
        self.config.epochs
    }

    #[getter]
    fn config_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    #[getter]
    fn config_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    #[getter]
    fn config_add_self_loop(&self) -> bool {
        self.config.add_self_loop
    }

    #[getter]
    fn config_l2_regularization(&self) -> f32 {
        self.config.l2_regularization
    }

    #[getter]
    fn hidden_dims(&self) -> Vec<usize> {
        self.config.hidden_dims.clone()
    }
}

#[pyclass(name = "Node2VecEncoder")]
#[derive(Clone)]
struct NativeNode2VecEncoder {
    config: Node2VecConfig,
    encoder: Node2VecEncoder,
}

#[pymethods]
impl NativeNode2VecEncoder {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let encoder = Node2VecEncoder::new(config.clone()).map_err(to_py_neural_error)?;
        Ok(Self { config, encoder })
    }

    #[pyo3(signature = (node_count, edges, edge_weights=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let mut model = Node2VecEncoder::new(self.config.clone()).map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| model.fit(node_count, &edges, edge_weights.as_deref()))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .allow_threads(|| self.encoder.encode())
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .allow_threads(|| Node2VecEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self { encoder, config })
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.encoder.node_count()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }

    #[getter]
    fn config_seed(&self) -> u64 {
        self.config.seed
    }

    #[getter]
    fn config_epochs(&self) -> usize {
        self.config.epochs
    }

    #[getter]
    fn config_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    #[getter]
    fn config_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    #[getter]
    fn config_p(&self) -> f32 {
        self.config.p
    }

    #[getter]
    fn config_q(&self) -> f32 {
        self.config.q
    }
}

#[pyclass(name = "StandaloneNeuralEmbeddingRegressor")]
#[derive(Clone)]
struct NativeStandaloneNeuralEmbeddingRegressor {
    model: StandaloneNeuralEmbeddingRegressor,
}

#[pymethods]
impl NativeStandaloneNeuralEmbeddingRegressor {
    #[new]
    #[pyo3(signature = (dim, fallback="global_mean_vector", random_state=None, support_prior_strength=1.0, n_estimators=80, learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        fallback: &str,
        random_state: Option<u64>,
        support_prior_strength: f64,
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let fallback = parse_embedding_fallback(fallback, None)?;
        let model = StandaloneNeuralEmbeddingRegressor::new(
            dim,
            fallback,
            random_state,
            support_prior_strength,
            standalone_booster_config(
                n_estimators,
                learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (ids, y, dense=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        ids: Vec<u64>,
        y: Vec<f64>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&ids, &y, dense.as_deref()))
            .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (ids, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        ids: Vec<u64>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict(&ids, dense.as_deref()))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| StandaloneNeuralEmbeddingRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneNode2VecRegressor")]
#[derive(Clone)]
struct NativeStandaloneNode2VecRegressor {
    model: Node2VecRegressor,
}

#[pymethods]
impl NativeStandaloneNode2VecRegressor {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let model = Node2VecRegressor::new(
            config,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_count, edges, row_nodes, y, row_targets=None, dense=None, edge_weights=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.model.fit(
                node_count,
                &edges,
                edge_weights.as_deref(),
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.model
                .predict(&row_nodes, row_targets.as_deref(), dense.as_deref())
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Node2VecRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneGraphSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneGraphSageRegressor {
    model: GraphSageRegressor,
}

#[pymethods]
impl NativeStandaloneGraphSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
        };
        let model = GraphSageRegressor::new(
            config,
            input_dim,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.model.fit(
                &node_features,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| GraphSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHeteroGraphSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneHeteroGraphSageRegressor {
    model: HeteroGraphSageRegressor,
}

#[pymethods]
impl NativeStandaloneHeteroGraphSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
        };
        let model = HeteroGraphSageRegressor::new(
            config,
            input_dim,
            relation_count,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.model.fit(
                &node_features,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| HeteroGraphSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHinSageRegressor")]
#[derive(Clone)]
struct NativeStandaloneHinSageRegressor {
    model: HinSageRegressor,
}

#[pymethods]
impl NativeStandaloneHinSageRegressor {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None, n_estimators=80, booster_learning_rate=0.07, max_depth=4, min_samples_leaf=2, min_gain=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
        n_estimators: usize,
        booster_learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
        };
        let model = HinSageRegressor::new(
            config,
            input_dim,
            node_type_count,
            edge_type_triples,
            standalone_booster_config(
                n_estimators,
                booster_learning_rate,
                max_depth,
                min_samples_leaf,
                min_gain,
            ),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_features, node_types, edges, row_nodes, y, row_targets=None, dense=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        row_nodes: Vec<usize>,
        y: Vec<f64>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.model.fit(
                &node_features,
                &node_types,
                &edges,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
                &y,
            )
        })
        .map_err(to_py_neural_error)
    }

    #[pyo3(signature = (node_features, row_nodes, row_targets=None, dense=None))]
    fn predict(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        row_nodes: Vec<usize>,
        row_targets: Option<Vec<usize>>,
        dense: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.model.predict(
                &node_features,
                &row_nodes,
                row_targets.as_deref(),
                dense.as_deref(),
            )
        })
        .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| HinSageRegressor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneNode2VecLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneNode2VecLinkPredictor {
    model: Node2VecLinkPredictor,
}

#[pymethods]
impl NativeStandaloneNode2VecLinkPredictor {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let model = Node2VecLinkPredictor::new(config).map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (node_count, edges, edge_weights=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(node_count, &edges, edge_weights.as_deref()))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(&self, py: Python<'_>, pairs: Vec<(usize, usize)>) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict_scores(&pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| Node2VecLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneGraphSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneGraphSageLinkPredictor {
    model: GraphSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneGraphSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
        };
        let model = GraphSageLinkPredictor::new(config, input_dim).map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize)>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&node_features, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| GraphSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHeteroGraphSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneHeteroGraphSageLinkPredictor {
    model: HeteroGraphSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneHeteroGraphSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
        };
        let model = HeteroGraphSageLinkPredictor::new(config, input_dim, relation_count)
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        edges: Vec<(usize, usize, usize)>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&node_features, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| HeteroGraphSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "StandaloneHinSageLinkPredictor")]
#[derive(Clone)]
struct NativeStandaloneHinSageLinkPredictor {
    model: HinSageLinkPredictor,
}

#[pymethods]
impl NativeStandaloneHinSageLinkPredictor {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
        };
        let model =
            HinSageLinkPredictor::new(config, input_dim, node_type_count, edge_type_triples)
                .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.model.fit(&node_features, &node_types, &edges))
            .map_err(to_py_neural_error)
    }

    fn predict_scores(
        &self,
        py: Python<'_>,
        node_features: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| self.model.predict_scores(&node_features, &pairs))
            .map_err(to_py_neural_error)
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.model.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let model = py
            .allow_threads(|| HinSageLinkPredictor::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        Ok(Self { model })
    }
}

#[pyclass(name = "HeteroGraphSageEncoder")]
#[derive(Clone)]
struct NativeHeteroGraphSageEncoder {
    config: HeteroGraphSageConfig,
    relation_count: usize,
    encoder: HeteroGraphSageEncoder,
}

#[pymethods]
impl NativeHeteroGraphSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
        };
        let encoder = HeteroGraphSageEncoder::new(config.clone(), input_dim, relation_count)
            .map_err(to_py_neural_error)?;
        Ok(Self {
            config,
            relation_count,
            encoder,
        })
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HeteroGraph::from_typed_edges(node_count, self.relation_count, &typed_edges)
            .map_err(to_py_neural_error)?;
        let mut model = HeteroGraphSageEncoder::new(
            self.config.clone(),
            self.encoder.input_dim(),
            self.relation_count,
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .allow_threads(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_count: usize,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HeteroGraph::from_typed_edges(node_count, self.relation_count, &typed_edges)
            .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .allow_threads(|| HeteroGraphSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self {
            relation_count: encoder.relation_count(),
            config,
            encoder,
        })
    }

    #[getter]
    fn relation_count(&self) -> usize {
        self.relation_count
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }
}

#[pyclass(name = "HinSageEncoder")]
#[derive(Clone)]
struct NativeHinSageEncoder {
    config: HinSageConfig,
    node_type_count: usize,
    edge_type_triples: Vec<(usize, usize, usize)>,
    encoder: HinSageEncoder,
}

#[pymethods]
impl NativeHinSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
        };
        let encoder = HinSageEncoder::new(
            config.clone(),
            input_dim,
            node_type_count,
            edge_type_triples.clone(),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self {
            config,
            node_type_count,
            edge_type_triples,
            encoder,
        })
    }

    #[pyo3(signature = (node_types, edges, node_features))]
    fn fit(
        &mut self,
        py: Python<'_>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HinSageGraph::from_typed_schema(
            node_types,
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges,
        )
        .map_err(to_py_neural_error)?;
        let mut model = HinSageEncoder::new(
            self.config.clone(),
            self.encoder.input_dim(),
            self.node_type_count,
            self.edge_type_triples.clone(),
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| model.fit(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, py: Python<'_>, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = py
            .allow_threads(|| self.encoder.encode(&node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_types, edges, node_features))]
    fn encode_graph(
        &self,
        py: Python<'_>,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HinSageGraph::from_typed_schema(
            node_types,
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges,
        )
        .map_err(to_py_neural_error)?;
        let embedding = py
            .allow_threads(|| self.encoder.encode_graph(&graph, &node_features))
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn link_embeddings(
        &self,
        py: Python<'_>,
        embeddings: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<Vec<f32>>> {
        py.allow_threads(|| self.encoder.link_embeddings(&embeddings, &pairs))
            .map_err(to_py_neural_error)
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(|| self.encoder.save_artifact_json(path))
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.encoder.to_artifact_json())
            .map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        path: String,
    ) -> PyResult<Self> {
        let encoder = py
            .allow_threads(|| HinSageEncoder::load_artifact_json(path))
            .map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self {
            node_type_count: encoder.node_type_count(),
            edge_type_triples: encoder.edge_type_triples().to_vec(),
            config,
            encoder,
        })
    }

    #[getter]
    fn node_type_count(&self) -> usize {
        self.node_type_count
    }

    #[getter]
    fn relation_count(&self) -> usize {
        self.edge_type_triples.len()
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn edge_type_triples(&self) -> Vec<(usize, usize, usize)> {
        self.edge_type_triples.clone()
    }

    #[getter]
    fn neighbor_samples(&self) -> Vec<usize> {
        self.config.neighbor_samples.clone()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }
}

#[pyfunction]
#[pyo3(signature = (node_count, edges, embeddings, edge_weights=None, edge_timestamps=None, feature_prefix="graph", requested_features=None))]
#[allow(clippy::too_many_arguments)]
fn graph_compute_directional_features(
    py: Python<'_>,
    node_count: usize,
    edges: Vec<(usize, usize)>,
    embeddings: Vec<Vec<f32>>,
    edge_weights: Option<Vec<f32>>,
    edge_timestamps: Option<Vec<f32>>,
    feature_prefix: &str,
    requested_features: Option<Vec<String>>,
) -> PyResult<(Vec<Vec<f32>>, Vec<String>)> {
    let requested_features = requested_features.unwrap_or_default();
    let block = py
        .allow_threads(|| {
            compute_directional_features(
                node_count,
                &edges,
                &embeddings,
                edge_weights.as_deref(),
                edge_timestamps.as_deref(),
                feature_prefix,
                &requested_features,
            )
        })
        .map_err(to_py_neural_error)?;
    Ok((block.values, block.feature_names))
}

#[pyfunction]
fn graph_validate_directed_metapath(
    py: Python<'_>,
    steps: Vec<String>,
    edge_types: Vec<(String, String, String)>,
) -> PyResult<()> {
    py.allow_threads(|| validate_directed_metapath(&steps, &edge_types))
        .map_err(to_py_neural_error)
}

#[pyfunction]
#[pyo3(signature = (edges, source_to_pair_relation="source_to_pair", pair_to_target_relation="pair_to_target", pair_node_prefix="od_pair", include_original_edges=true))]
fn graph_materialize_source_target_pair_nodes(
    py: Python<'_>,
    edges: Vec<(String, String, String)>,
    source_to_pair_relation: &str,
    pair_to_target_relation: &str,
    pair_node_prefix: &str,
    include_original_edges: bool,
) -> PyResult<(StringTypedEdges, Vec<String>)> {
    let source_to_pair_relation = source_to_pair_relation.to_string();
    let pair_to_target_relation = pair_to_target_relation.to_string();
    let pair_node_prefix = pair_node_prefix.to_string();
    let expansion = py
        .allow_threads(|| {
            materialize_source_target_pair_nodes(
                &edges,
                &source_to_pair_relation,
                &pair_to_target_relation,
                &pair_node_prefix,
                include_original_edges,
            )
        })
        .map_err(to_py_neural_error)?;
    Ok((expansion.edges, expansion.pair_node_ids))
}

#[pyfunction]
fn h3_normalize_id_text(value: &str) -> PyResult<u64> {
    normalize_h3_id_text(value).map_err(to_py_value_error)
}

#[pyfunction]
fn s2_normalize_id_text(value: &str) -> PyResult<u64> {
    normalize_s2_id_text(value).map_err(to_py_value_error)
}

#[pyfunction]
fn h3_normalize_resolution_value(value: i64, field_name: &str) -> PyResult<u8> {
    normalize_h3_resolution(value, field_name).map_err(to_py_value_error)
}

#[pyfunction]
fn s2_normalize_level_value(value: i64, field_name: &str) -> PyResult<u8> {
    normalize_s2_level(value, field_name).map_err(to_py_value_error)
}

#[pyfunction]
fn geo_normalize_coordinate_value(value: f64, field_name: &str) -> PyResult<f64> {
    core_normalize_coordinate(value, field_name).map_err(to_py_value_error)
}

#[pyfunction]
fn h3_validate_parent_resolutions_value(
    py: Python<'_>,
    resolution: u8,
    parent_resolutions: Vec<u8>,
) -> PyResult<()> {
    py.allow_threads(|| validate_parent_levels(resolution, &parent_resolutions, GeoGridKind::H3))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn s2_validate_parent_levels_value(
    py: Python<'_>,
    level: u8,
    parent_levels: Vec<u8>,
) -> PyResult<()> {
    py.allow_threads(|| validate_parent_levels(level, &parent_levels, GeoGridKind::S2))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn h3_scaffold_parent_id_value(cell: u64, resolution: u8, parent_resolution: u8) -> PyResult<u64> {
    scaffold_h3_parent_id(cell, resolution, parent_resolution).map_err(to_py_value_error)
}

#[pyfunction]
fn h3_expand_sparse_set_value(
    py: Python<'_>,
    values: Vec<u64>,
    resolution: u8,
    parent_resolutions: Vec<u8>,
) -> PyResult<Vec<u64>> {
    py.allow_threads(|| core_expand_h3_sparse_set(&values, resolution, &parent_resolutions))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn geo_assemble_sparse_row_value(child: u64, parents: Vec<u64>) -> Vec<u64> {
    assemble_sparse_row(child, &parents)
}

#[pyfunction]
fn geo_assemble_sparse_column_value(
    py: Python<'_>,
    children: Vec<u64>,
    parent_columns: Vec<Vec<u64>>,
) -> PyResult<Vec<Vec<u64>>> {
    py.allow_threads(|| assemble_sparse_column(&children, &parent_columns))
        .map_err(to_py_value_error)
}

#[pyfunction]
fn geo_validate_equal_row_count_value(name: &str, actual: usize, expected: usize) -> PyResult<()> {
    validate_equal_row_count(name, actual, expected).map_err(to_py_value_error)
}

fn artifact_fallback_name(fallback: &ArtifactFallbackKind) -> &'static str {
    match fallback {
        ArtifactFallbackKind::ZeroVector => "zero_vector",
        ArtifactFallbackKind::GlobalMeanVector => "global_mean_vector",
        ArtifactFallbackKind::ParentCell { .. } => "parent_cell",
    }
}

fn standalone_booster_config(
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
) -> StandaloneBoosterConfig {
    StandaloneBoosterConfig {
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_leaf,
        min_gain,
    }
}

fn parse_leaf_predictor(name: &str) -> PyResult<LeafPredictorKind> {
    match name {
        "constant" => Ok(LeafPredictorKind::Constant),
        "linear" => Ok(LeafPredictorKind::Linear),
        _ => Err(PyValueError::new_err(format!(
            "unknown leaf_predictor {name:?}; expected 'constant' or 'linear'"
        ))),
    }
}

fn parse_fuzzy_kernel(name: &str) -> PyResult<FuzzyKernel> {
    match name {
        "linear" | "triangular" => Ok(FuzzyKernel::Linear),
        "gaussian" => Ok(FuzzyKernel::Gaussian),
        "exponential" => Ok(FuzzyKernel::Exponential),
        "bisquare" => Ok(FuzzyKernel::Bisquare),
        "epanechnikov" => Ok(FuzzyKernel::Epanechnikov),
        "tricube" => Ok(FuzzyKernel::Tricube),
        _ => Err(PyValueError::new_err(format!(
            "unknown fuzzy_kernel {name:?}; expected 'linear', 'gaussian', 'exponential', 'bisquare', 'epanechnikov', or 'tricube'"
        ))),
    }
}

fn fuzzy_kernel_name(kernel: FuzzyKernel) -> &'static str {
    match kernel {
        FuzzyKernel::Linear => "linear",
        FuzzyKernel::Gaussian => "gaussian",
        FuzzyKernel::Exponential => "exponential",
        FuzzyKernel::Bisquare => "bisquare",
        FuzzyKernel::Epanechnikov => "epanechnikov",
        FuzzyKernel::Tricube => "tricube",
    }
}

fn parse_loss(
    name: &str,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
) -> PyResult<LossConfig> {
    match name {
        "l2" | "squared_error" => Ok(LossConfig::L2),
        "l1" | "mae" | "absolute_error" | "least_absolute_deviation" | "lad" => Ok(LossConfig::L1),
        "huber" => {
            if !huber_delta.is_finite() || huber_delta <= 0.0 {
                return Err(PyValueError::new_err(
                    "huber_delta must be positive and finite",
                ));
            }
            Ok(LossConfig::Huber(HuberLossConfig { delta: huber_delta }))
        }
        "log_l2" | "log" | "log_squared_error" => {
            if !log_offset.is_finite() || log_offset <= 0.0 {
                return Err(PyValueError::new_err(
                    "log_offset must be positive and finite",
                ));
            }
            if (log_offset - 1.0).abs() > 1e-12 {
                return Err(PyValueError::new_err(
                    "log_l2 currently supports log_offset=1.0",
                ));
            }
            Ok(LossConfig::LogL2(LogL2LossConfig { offset: log_offset }))
        }
        "quantile" | "pinball" => {
            if !quantile_alpha.is_finite() || quantile_alpha <= 0.0 || quantile_alpha >= 1.0 {
                return Err(PyValueError::new_err(
                    "quantile_alpha must be finite and in (0, 1)",
                ));
            }
            Ok(LossConfig::Quantile(QuantileLossConfig {
                alpha: quantile_alpha,
            }))
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown loss {name:?}; expected 'l2', 'l1', 'huber', 'log_l2', or 'quantile'"
        ))),
    }
}

fn loss_name(loss: &LossConfig) -> &'static str {
    match loss {
        LossConfig::L2 => "l2",
        LossConfig::L1 => "l1",
        LossConfig::Huber(_) => "huber",
        LossConfig::LogL2(_) => "log_l2",
        LossConfig::Quantile(_) => "quantile",
    }
}

fn quantile_alpha(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::L2 | LossConfig::L1 | LossConfig::Huber(_) | LossConfig::LogL2(_) => 0.5,
        LossConfig::Quantile(config) => config.alpha,
    }
}

fn huber_delta(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::Huber(config) => config.delta,
        _ => 1.0,
    }
}

fn log_offset(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::LogL2(config) => config.offset,
        _ => 1.0,
    }
}

fn splitter_names(splitters: &[SplitterKind]) -> Vec<String> {
    splitters
        .iter()
        .map(|splitter| match splitter {
            SplitterKind::Auto => "auto".to_string(),
            SplitterKind::Axis => "axis".to_string(),
            SplitterKind::AxisHistogram { bins } => format!("axis_histogram:{bins}"),
            SplitterKind::Diagonal2D => "diagonal_2d".to_string(),
            SplitterKind::Gaussian2D => "gaussian_2d".to_string(),
            SplitterKind::Periodic { period } if (*period - 24.0).abs() < 1e-12 => {
                "periodic_time".to_string()
            }
            SplitterKind::Periodic { period } => format!("periodic:{period}"),
            SplitterKind::SparseSet => "sparse_set".to_string(),
        })
        .collect()
}

fn leaf_predictor_name(leaf_predictor: &LeafPredictorKind) -> &'static str {
    match leaf_predictor {
        LeafPredictorKind::Constant => "constant",
        LeafPredictorKind::Linear => "linear",
    }
}

fn validate_n_threads(n_threads: Option<usize>) -> PyResult<()> {
    if n_threads == Some(0) {
        return Err(PyValueError::new_err("n_threads must be positive"));
    }
    Ok(())
}

fn run_with_optional_threads<T, F>(n_threads: Option<usize>, f: F) -> Result<T, CartoBoostError>
where
    T: Send,
    F: FnOnce() -> Result<T, CartoBoostError> + Send,
{
    if let Some(n_threads) = n_threads {
        ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?
            .install(f)
    } else {
        f()
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_params(
    n_estimators: usize,
    learning_rate: f64,
    _max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy_bandwidth: f64,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
) -> PyResult<()> {
    if n_estimators == 0 {
        return Err(PyValueError::new_err("n_estimators must be positive"));
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(PyValueError::new_err(
            "learning_rate must be positive and finite",
        ));
    }
    if min_samples_leaf == 0 {
        return Err(PyValueError::new_err("min_samples_leaf must be positive"));
    }
    if !min_gain.is_finite() || min_gain < 0.0 {
        return Err(PyValueError::new_err(
            "min_gain must be finite and non-negative",
        ));
    }
    if !l2_regularization.is_finite() || l2_regularization < 0.0 {
        return Err(PyValueError::new_err(
            "l2_regularization must be finite and non-negative",
        ));
    }
    if !constant_l2_regularization.is_finite() || constant_l2_regularization < 0.0 {
        return Err(PyValueError::new_err(
            "constant_l2_regularization must be finite and non-negative",
        ));
    }
    if !fuzzy_bandwidth.is_finite() || fuzzy_bandwidth < 0.0 {
        return Err(PyValueError::new_err(
            "fuzzy_bandwidth must be finite and non-negative",
        ));
    }
    if !quantile_alpha.is_finite() || quantile_alpha <= 0.0 || quantile_alpha >= 1.0 {
        return Err(PyValueError::new_err(
            "quantile_alpha must be finite and in (0, 1)",
        ));
    }
    if !huber_delta.is_finite() || huber_delta <= 0.0 {
        return Err(PyValueError::new_err(
            "huber_delta must be positive and finite",
        ));
    }
    if !log_offset.is_finite() || log_offset <= 0.0 {
        return Err(PyValueError::new_err(
            "log_offset must be positive and finite",
        ));
    }
    Ok(())
}

fn dataset_from_rows(rows: Vec<Vec<f64>>) -> PyResult<Dataset> {
    if rows.is_empty() {
        return Err(PyValueError::new_err("X must not be empty"));
    }
    if rows[0].is_empty() {
        return Err(PyValueError::new_err(
            "X rows must contain at least one feature",
        ));
    }
    if rows
        .iter()
        .any(|row| row.iter().any(|value| !value.is_finite()))
    {
        return Err(PyValueError::new_err("X must contain only finite values"));
    }
    Dataset::from_rows(rows).map_err(to_py_value_error)
}

fn dataset_from_parts(
    rows: Vec<Vec<f64>>,
    sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    feature_schema_json: Option<String>,
) -> PyResult<Dataset> {
    let dataset = dataset_from_rows(rows)?;
    let sparse_sets = sparse_sets
        .unwrap_or_default()
        .into_iter()
        .map(SparseSetColumn::new)
        .collect::<Vec<_>>();
    let schema = feature_schema_json
        .map(|payload| serde_json::from_str::<FeatureSchema>(&payload))
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema: {err}")))?;
    let dataset = dataset
        .with_sparse_sets(sparse_sets)
        .map_err(to_py_value_error)?;
    match schema {
        Some(schema) => dataset.with_schema(schema).map_err(to_py_value_error),
        None => Ok(dataset),
    }
}

fn dataset_from_arrays(
    x: PyReadonlyArray2<'_, f64>,
    sparse_offsets: Option<Vec<Vec<usize>>>,
    sparse_ids: Option<Vec<Vec<u64>>>,
    feature_schema_json: Option<String>,
) -> PyResult<Dataset> {
    let shape = x.shape();
    let rows = shape[0];
    let cols = shape[1];
    let values = x.as_slice()?.to_vec();
    let dataset = Dataset::from_flat(rows, cols, values).map_err(to_py_value_error)?;
    let sparse_sets = encoded_sparse_sets(rows, sparse_offsets, sparse_ids)?
        .into_iter()
        .map(SparseSetColumn::new)
        .collect::<Vec<_>>();
    let schema = feature_schema_json
        .map(|payload| serde_json::from_str::<FeatureSchema>(&payload))
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema: {err}")))?;
    let dataset = dataset
        .with_sparse_sets(sparse_sets)
        .map_err(to_py_value_error)?;
    match schema {
        Some(schema) => dataset.with_schema(schema).map_err(to_py_value_error),
        None => Ok(dataset),
    }
}

fn encoded_sparse_sets(
    rows: usize,
    sparse_offsets: Option<Vec<Vec<usize>>>,
    sparse_ids: Option<Vec<Vec<u64>>>,
) -> PyResult<Vec<Vec<Vec<u64>>>> {
    let offsets = sparse_offsets.unwrap_or_default();
    let ids = sparse_ids.unwrap_or_default();
    if offsets.len() != ids.len() {
        return Err(PyValueError::new_err(
            "sparse_offsets and sparse_ids must contain the same number of columns",
        ));
    }
    let mut columns = Vec::with_capacity(offsets.len());
    for (column_index, (column_offsets, column_ids)) in offsets.into_iter().zip(ids).enumerate() {
        if column_offsets.len() != rows + 1 {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must have rows + 1 entries"
            )));
        }
        if column_offsets.first().copied() != Some(0) {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must start at 0"
            )));
        }
        if column_offsets.last().copied() != Some(column_ids.len()) {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} final offset must match sparse_ids length"
            )));
        }
        if column_offsets
            .windows(2)
            .any(|window| window[0] > window[1])
        {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must be non-decreasing"
            )));
        }
        let mut column = Vec::with_capacity(rows);
        for window in column_offsets.windows(2) {
            column.push(column_ids[window[0]..window[1]].to_vec());
        }
        columns.push(column);
    }
    Ok(columns)
}

#[derive(Clone)]
struct OverlayPoint {
    id: String,
    coordinates: (f64, f64),
    properties: serde_json::Map<String, Value>,
}

struct OverlayZone {
    id: String,
    priority: f64,
    bbox: (f64, f64, f64, f64),
    ring: Vec<(f64, f64)>,
}

#[pyfunction(signature = (points, zones, weights, origin=None, zone_priority_multiplier=true, kernel="none", bandwidth_meters=None, distance_alpha=0.0, precision=6, include_debug=false))]
#[allow(clippy::too_many_arguments)]
fn weighted_overlay(
    py: Python<'_>,
    points: Bound<'_, PyAny>,
    zones: Bound<'_, PyAny>,
    weights: Bound<'_, PyAny>,
    origin: Option<(f64, f64)>,
    zone_priority_multiplier: bool,
    kernel: &str,
    bandwidth_meters: Option<f64>,
    distance_alpha: f64,
    precision: usize,
    include_debug: bool,
) -> PyResult<Py<PyAny>> {
    let json_module = PyModule::import(py, "json")?;
    let points_payload = json_module
        .call_method1("dumps", (points,))?
        .extract::<String>()?;
    let zones_payload = json_module
        .call_method1("dumps", (zones,))?
        .extract::<String>()?;
    let weights_payload = json_module
        .call_method1("dumps", (weights,))?
        .extract::<String>()?;

    let kernel = kernel.to_string();
    let payload = py
        .allow_threads(|| {
            let points_value = serde_json::from_str::<Value>(&points_payload)
                .map_err(|err| format!("invalid points payload: {err}"))?;
            let zones_value = serde_json::from_str::<Value>(&zones_payload)
                .map_err(|err| format!("invalid zones payload: {err}"))?;
            let weights_value = serde_json::from_str::<Value>(&weights_payload)
                .map_err(|err| format!("invalid weights payload: {err}"))?;

            let result = weighted_overlay_impl(
                &points_value,
                &zones_value,
                &weights_value,
                origin,
                zone_priority_multiplier,
                &kernel,
                bandwidth_meters,
                distance_alpha,
                precision,
                include_debug,
            )?;

            serde_json::to_string(&result)
                .map_err(|err| format!("failed to serialize overlay result: {err}"))
        })
        .map_err(PyValueError::new_err)?;
    Ok(json_module.call_method1("loads", (payload,))?.unbind())
}

#[allow(clippy::too_many_arguments)]
fn weighted_overlay_impl(
    points: &Value,
    zones: &Value,
    weights: &Value,
    origin: Option<(f64, f64)>,
    zone_priority_multiplier: bool,
    kernel: &str,
    bandwidth_meters: Option<f64>,
    distance_alpha: f64,
    precision: usize,
    include_debug: bool,
) -> Result<Value, String> {
    let overlay_points = parse_overlay_points(points)?;
    let overlay_zones = parse_overlay_zones(zones)?;
    let weight_map = weights
        .as_object()
        .ok_or_else(|| "weights must be a JSON object".to_string())?;

    let mut features = Vec::with_capacity(overlay_points.len());
    for point in &overlay_points {
        let zone = locate_zone(&overlay_zones, point.coordinates)?;
        let linear_score = weight_map.iter().try_fold(0.0, |score, (name, weight)| {
            let weight_value = weight
                .as_f64()
                .ok_or_else(|| format!("weight {name:?} must be numeric"))?;
            let property_value = point
                .properties
                .get(name)
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            Ok::<f64, String>(score + weight_value * property_value)
        })?;

        let priority = if zone_priority_multiplier {
            zone.priority
        } else {
            1.0
        };

        let spatial_term = if let Some(origin) = (kernel != "none" && distance_alpha != 0.0)
            .then_some(origin)
            .flatten()
        {
            let bandwidth =
                resolve_bandwidth(bandwidth_meters, point.coordinates, &overlay_points)?;
            let distance = haversine_meters(origin, point.coordinates);
            distance_alpha * kernel_weight(distance, bandwidth, kernel)?
        } else {
            0.0
        };

        let mut feature = json!({
            "id": point.id,
            "zone_id": zone.id,
            "boost_score": round_half_even(linear_score * priority * (1.0 + spatial_term), precision),
        });
        if include_debug {
            feature["debug"] = json!({
                "linear": round_half_even(linear_score, precision),
                "priority": round_half_even(priority, precision),
                "spatial_term": round_half_even(spatial_term, precision),
            });
        }
        features.push(feature);
    }

    features.sort_by(|left, right| {
        let right_score = right["boost_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        let left_score = left["boost_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left["id"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(right["id"].as_str().unwrap_or(""))
            })
    });
    for (rank, feature) in features.iter_mut().enumerate() {
        feature["rank"] = json!(rank + 1);
    }

    let points_name = points
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| points.get("type").and_then(Value::as_str))
        .unwrap_or("points");
    let zones_name = zones
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| zones.get("type").and_then(Value::as_str))
        .unwrap_or("zones");

    let mut config = json!({
        "algorithm": "weighted_overlay",
        "weights": weights.clone(),
        "zone_priority_multiplier": zone_priority_multiplier,
        "rounding": {
            "places": precision,
            "mode": "half_even",
        },
    });
    if origin.is_some() || kernel != "none" || distance_alpha != 0.0 {
        config["distance_term"] = json!({
            "enabled": origin.is_some() && kernel != "none" && distance_alpha != 0.0,
            "source": if origin.is_some() { Value::String("origin".to_string()) } else { Value::Null },
            "kernel": kernel,
            "bandwidth_meters": bandwidth_meters,
            "distance_alpha": distance_alpha,
        });
    }

    Ok(json!({
        "schema_version": 1,
        "scenario": format!("{points_name}_x_{zones_name}"),
        "config": config,
        "features": features,
    }))
}

fn parse_overlay_points(points: &Value) -> Result<Vec<OverlayPoint>, String> {
    let features = points
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| "points must contain a features array".to_string())?;
    features
        .iter()
        .map(|feature| {
            let id = feature
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| "point features must provide an id".to_string())?;
            let cartometry = feature
                .get("cartometry")
                .ok_or_else(|| format!("point feature {id:?} is missing cartometry"))?;
            let cartometry_type = cartometry
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("point feature {id:?} cartometry is missing type"))?;
            if cartometry_type != "Point" {
                return Err(format!("point feature {id:?} must use Point cartometry"));
            }
            let coordinates = cartometry
                .get("coordinates")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("point feature {id:?} is missing coordinates"))?;
            if coordinates.len() < 2 {
                return Err(format!(
                    "point feature {id:?} must provide [x, y] coordinates"
                ));
            }
            let x = coordinates[0]
                .as_f64()
                .ok_or_else(|| format!("point feature {id:?} x coordinate must be numeric"))?;
            let y = coordinates[1]
                .as_f64()
                .ok_or_else(|| format!("point feature {id:?} y coordinate must be numeric"))?;
            let properties = feature
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            Ok(OverlayPoint {
                id: id.to_string(),
                coordinates: (x, y),
                properties,
            })
        })
        .collect()
}

fn parse_overlay_zones(zones: &Value) -> Result<Vec<OverlayZone>, String> {
    let features = zones
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| "zones must contain a features array".to_string())?;
    features
        .iter()
        .map(|feature| {
            let id = feature
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| "zone features must provide an id".to_string())?;
            let cartometry = feature
                .get("cartometry")
                .ok_or_else(|| format!("zone feature {id:?} is missing cartometry"))?;
            let cartometry_type = cartometry
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("zone feature {id:?} cartometry is missing type"))?;
            if cartometry_type != "Polygon" {
                return Err(format!("zone feature {id:?} must use Polygon cartometry"));
            }
            let rings = cartometry
                .get("coordinates")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("zone feature {id:?} is missing polygon coordinates"))?;
            let outer_ring = rings
                .first()
                .and_then(Value::as_array)
                .ok_or_else(|| format!("zone feature {id:?} is missing an outer ring"))?;
            let ring = outer_ring
                .iter()
                .map(|coordinate| {
                    let pair = coordinate.as_array().ok_or_else(|| {
                        format!("zone feature {id:?} ring coordinates must be arrays")
                    })?;
                    if pair.len() < 2 {
                        return Err(format!(
                            "zone feature {id:?} ring coordinates must have two values"
                        ));
                    }
                    let x = pair[0].as_f64().ok_or_else(|| {
                        format!("zone feature {id:?} x coordinate must be numeric")
                    })?;
                    let y = pair[1].as_f64().ok_or_else(|| {
                        format!("zone feature {id:?} y coordinate must be numeric")
                    })?;
                    Ok((x, y))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let bbox = bounding_box(&ring)?;
            let priority = feature
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|properties| properties.get("priority"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            Ok(OverlayZone {
                id: id.to_string(),
                priority,
                bbox,
                ring,
            })
        })
        .collect()
}

fn locate_zone(zones: &[OverlayZone], point: (f64, f64)) -> Result<&OverlayZone, String> {
    let (x, y) = point;
    zones
        .iter()
        .find(|zone| {
            let (min_x, min_y, max_x, max_y) = zone.bbox;
            min_x <= x
                && x <= max_x
                && min_y <= y
                && y <= max_y
                && point_in_polygon(point, &zone.ring)
        })
        .ok_or_else(|| format!("point ({x}, {y}) does not belong to any zone"))
}

fn bounding_box(ring: &[(f64, f64)]) -> Result<(f64, f64, f64, f64), String> {
    if ring.is_empty() {
        return Err("polygon ring must not be empty".to_string());
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in ring {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }
    Ok((min_x, min_y, max_x, max_y))
}

fn point_in_polygon(point: (f64, f64), ring: &[(f64, f64)]) -> bool {
    if ring.len() < 2 {
        return false;
    }
    let (x, y) = point;
    let mut inside = false;
    for index in 0..(ring.len() - 1) {
        let start = ring[index];
        let end = ring[index + 1];
        if point_on_segment(point, start, end) {
            return true;
        }
        let intersects = (start.1 > y) != (end.1 > y);
        if intersects {
            let slope_x =
                (end.0 - start.0) * (y - start.1) / ((end.1 - start.1).abs().max(1e-12)) + start.0;
            if x <= slope_x {
                inside = !inside;
            }
        }
    }
    inside
}

fn point_on_segment(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> bool {
    let cross = (point.0 - start.0) * (end.1 - start.1) - (point.1 - start.1) * (end.0 - start.0);
    if cross.abs() > 1e-9 {
        return false;
    }
    let min_x = start.0.min(end.0) - 1e-9;
    let max_x = start.0.max(end.0) + 1e-9;
    let min_y = start.1.min(end.1) - 1e-9;
    let max_y = start.1.max(end.1) + 1e-9;
    min_x <= point.0 && point.0 <= max_x && min_y <= point.1 && point.1 <= max_y
}

fn resolve_bandwidth(
    bandwidth_meters: Option<f64>,
    point: (f64, f64),
    points: &[OverlayPoint],
) -> Result<f64, String> {
    if let Some(bandwidth) = bandwidth_meters {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err("bandwidth_meters must be positive and finite".to_string());
        }
        return Ok(bandwidth);
    }
    let mut distances = points
        .iter()
        .filter(|candidate| candidate.coordinates != point)
        .map(|candidate| haversine_meters(point, candidate.coordinates))
        .collect::<Vec<_>>();
    if distances.is_empty() {
        return Ok(1.0);
    }
    distances.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Ok(distances[distances.len().min(3) - 1].max(1.0))
}

fn kernel_weight(distance_meters: f64, bandwidth_meters: f64, kernel: &str) -> Result<f64, String> {
    if !bandwidth_meters.is_finite() || bandwidth_meters <= 0.0 {
        return Err("bandwidth_meters must be positive and finite".to_string());
    }
    let ratio = distance_meters / bandwidth_meters;
    match kernel {
        "none" => Ok(0.0),
        "gaussian" => Ok((-0.5 * ratio * ratio).exp()),
        "bisquare" => {
            if ratio >= 1.0 {
                Ok(0.0)
            } else {
                Ok((1.0 - ratio * ratio).powi(2))
            }
        }
        "exponential" => Ok((-ratio).exp()),
        _ => Err(format!("unknown kernel {kernel:?}")),
    }
}

fn haversine_meters(origin: (f64, f64), destination: (f64, f64)) -> f64 {
    let lon1 = origin.0.to_radians();
    let lat1 = origin.1.to_radians();
    let lon2 = destination.0.to_radians();
    let lat2 = destination.1.to_radians();
    let dlon = lon2 - lon1;
    let dlat = lat2 - lat1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * 6_371_000.0 * a.sqrt().asin()
}

fn round_half_even(value: f64, precision: usize) -> f64 {
    let factor = 10_f64.powi(precision as i32);
    let scaled = value * factor;
    let sign = if scaled.is_sign_negative() { -1.0 } else { 1.0 };
    let scaled_abs = scaled.abs();
    let lower = scaled_abs.floor();
    let fraction = scaled_abs - lower;
    let rounded = if fraction > 0.5 + 1e-12 {
        lower + 1.0
    } else if fraction < 0.5 - 1e-12 || (lower as i64) % 2 == 0 {
        lower
    } else {
        lower + 1.0
    };
    sign * rounded / factor
}

fn to_py_value_error(err: CartoBoostError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_error(err: CartoBoostError) -> PyErr {
    match err {
        CartoBoostError::Io(_) => PyIOError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn to_py_neural_error(err: cartoboost_neural::NeuralError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeCartoBoostRegressor>()?;
    m.add_class::<NativeForecastFrame>()?;
    m.add_class::<NativeForecastResult>()?;
    m.add_class::<NativeForecastFold>()?;
    m.add_class::<NativeRollingOriginSplitter>()?;
    m.add_class::<NativeForecastMetricSet>()?;
    m.add_class::<NativeBacktestFoldResult>()?;
    m.add_class::<NativeBacktestResult>()?;
    m.add_class::<NativeRollingOriginBacktester>()?;
    m.add_class::<NativeNaiveForecaster>()?;
    m.add_class::<NativeSeasonalNaiveForecaster>()?;
    m.add_class::<NativeThetaForecaster>()?;
    m.add_class::<NativeOptimizedThetaForecaster>()?;
    m.add_class::<NativeETSForecaster>()?;
    m.add_class::<NativeArimaForecaster>()?;
    m.add_class::<NativeAutoARIMAForecaster>()?;
    m.add_class::<NativeKalmanForecaster>()?;
    m.add_class::<NativeLocalLevelKalmanForecaster>()?;
    m.add_class::<NativeAutoKalmanForecaster>()?;
    m.add_class::<NativeAutoLocalLevelKalmanForecaster>()?;
    m.add_class::<NativeKrigingForecaster>()?;
    m.add_class::<NativeCartoBoostLagForecaster>()?;
    m.add_class::<NativeWeightedEnsembleForecaster>()?;
    m.add_class::<NativeNeuralEmbeddingFeatures>()?;
    m.add_class::<NativeGraphSageEncoder>()?;
    m.add_class::<NativeNode2VecEncoder>()?;
    m.add_class::<NativeStandaloneNeuralEmbeddingRegressor>()?;
    m.add_class::<NativeStandaloneNode2VecRegressor>()?;
    m.add_class::<NativeStandaloneGraphSageRegressor>()?;
    m.add_class::<NativeStandaloneHeteroGraphSageRegressor>()?;
    m.add_class::<NativeStandaloneHinSageRegressor>()?;
    m.add_class::<NativeStandaloneNode2VecLinkPredictor>()?;
    m.add_class::<NativeStandaloneGraphSageLinkPredictor>()?;
    m.add_class::<NativeStandaloneHeteroGraphSageLinkPredictor>()?;
    m.add_class::<NativeStandaloneHinSageLinkPredictor>()?;
    m.add_class::<NativeHeteroGraphSageEncoder>()?;
    m.add_class::<NativeHinSageEncoder>()?;
    m.add_function(wrap_pyfunction!(utility_kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(utility_local_level_kalman_filter, m)?)?;
    m.add_function(wrap_pyfunction!(utility_intermittent_demand_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(utility_ordinary_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(
        utility_ordinary_kriging_predict_detailed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utility_ordinary_kriging_leave_one_out, m)?)?;
    m.add_function(wrap_pyfunction!(utility_empirical_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(utility_fit_ordinary_kriging_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(
        utility_ordinary_kriging_leave_one_out_diagnostics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(utility_series_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(graph_compute_directional_features, m)?)?;
    m.add_function(wrap_pyfunction!(graph_validate_directed_metapath, m)?)?;
    m.add_function(wrap_pyfunction!(
        graph_materialize_source_target_pair_nodes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(h3_normalize_id_text, m)?)?;
    m.add_function(wrap_pyfunction!(s2_normalize_id_text, m)?)?;
    m.add_function(wrap_pyfunction!(h3_normalize_resolution_value, m)?)?;
    m.add_function(wrap_pyfunction!(s2_normalize_level_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_normalize_coordinate_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_validate_parent_resolutions_value, m)?)?;
    m.add_function(wrap_pyfunction!(s2_validate_parent_levels_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_scaffold_parent_id_value, m)?)?;
    m.add_function(wrap_pyfunction!(h3_expand_sparse_set_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_assemble_sparse_row_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_assemble_sparse_column_value, m)?)?;
    m.add_function(wrap_pyfunction!(geo_validate_equal_row_count_value, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_overlay, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_parse_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_evaluate_metrics, m)?)?;
    Ok(())
}
