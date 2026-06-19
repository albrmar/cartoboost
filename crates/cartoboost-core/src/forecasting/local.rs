#![allow(dead_code)]

use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::utilities::{
    fit_local_level_kalman, fit_local_linear_kalman, ordinary_kriging_predict_many,
    KrigingObservation, LocalLevelKalmanConfig, LocalLinearKalmanConfig, OrdinaryKrigingConfig,
};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const MAX_ARIMA_ORDER: usize = 8;
const MAX_ARIMA_COLUMNS: usize = MAX_ARIMA_ORDER * 2 + 1;

#[derive(Debug, Clone, Default)]
pub struct NaiveForecaster {
    fitted: Option<FittedLocalState>,
}

#[derive(Debug, Clone)]
pub struct SeasonalNaiveForecaster {
    season_length: usize,
    fitted: Option<FittedLocalState>,
}

#[derive(Debug, Clone)]
pub struct ThetaForecaster {
    theta: f64,
    alpha: f64,
    seasonality: Option<ThetaSeasonality>,
    fitted: Option<FittedThetaState>,
}

#[derive(Debug, Clone)]
pub struct OptimizedThetaForecaster {
    theta_grid: Vec<f64>,
    alpha_grid: Vec<f64>,
    seasonality: Option<ThetaSeasonality>,
    selected_theta: Option<f64>,
    selected_alpha: Option<f64>,
    validation_scores: Vec<ThetaValidationScore>,
    fitted: Option<ThetaForecaster>,
}

#[derive(Debug, Clone)]
pub struct ETSForecaster {
    alpha: f64,
    beta: f64,
    gamma: Option<f64>,
    season_length: Option<usize>,
    fitted: Option<FittedETSState>,
}

#[derive(Debug, Clone)]
pub struct ArimaForecaster {
    p: usize,
    d: usize,
    q: usize,
    fitted: Option<FittedArimaState>,
}

#[derive(Debug, Clone)]
pub struct AutoARIMAForecaster {
    max_p: usize,
    max_d: usize,
    max_q: usize,
    selected_order: Option<(usize, usize, usize)>,
    validation_scores: Vec<ArimaValidationScore>,
    fitted: Option<ArimaForecaster>,
}

#[derive(Debug, Clone)]
pub struct KalmanForecaster {
    level_process_variance: f64,
    trend_process_variance: f64,
    observation_variance: f64,
    fitted: Option<FittedKalmanState>,
}

#[derive(Debug, Clone)]
pub struct LocalLevelKalmanForecaster {
    level_process_variance: f64,
    observation_variance: f64,
    fitted: Option<FittedLocalLevelKalmanState>,
}

#[derive(Debug, Clone)]
pub struct AutoKalmanForecaster {
    level_process_variance_grid: Vec<f64>,
    trend_process_variance_grid: Vec<f64>,
    observation_variance_grid: Vec<f64>,
    validation_window: Option<usize>,
    selected_params: Option<KalmanParameterSet>,
    validation_scores: Vec<KalmanValidationScore>,
    fitted: Option<KalmanForecaster>,
}

#[derive(Debug, Clone)]
pub struct AutoLocalLevelKalmanForecaster {
    level_process_variance_grid: Vec<f64>,
    observation_variance_grid: Vec<f64>,
    validation_window: Option<usize>,
    selected_params: Option<LocalLevelKalmanParameterSet>,
    validation_scores: Vec<LocalLevelKalmanValidationScore>,
    fitted: Option<LocalLevelKalmanForecaster>,
}

#[derive(Debug, Clone)]
pub struct KrigingForecaster {
    coordinates: BTreeMap<String, (f64, f64)>,
    config: OrdinaryKrigingConfig,
    fitted: Option<FittedKrigingState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaSeasonalityKind {
    Additive,
    Multiplicative,
}

#[derive(Debug, Clone, Copy)]
pub struct ThetaSeasonality {
    kind: ThetaSeasonalityKind,
    season_length: usize,
}

#[derive(Debug, Clone)]
struct FittedLocalState {
    frame: ForecastFrame,
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
}

#[derive(Debug, Clone)]
struct FittedThetaState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedThetaSeries>,
}

#[derive(Debug, Clone)]
struct FittedETSState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedETSSeries>,
}

#[derive(Debug, Clone)]
struct FittedETSSeries {
    last_timestamp: chrono::NaiveDateTime,
    n_obs: usize,
    level: f64,
    trend: f64,
    seasonals: Option<Vec<f64>>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
    level_values: Vec<f64>,
    trend_values: Vec<f64>,
    seasonal_values: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FittedArimaState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedArimaSeries>,
}

#[derive(Debug, Clone)]
struct FittedArimaSeries {
    last_timestamp: chrono::NaiveDateTime,
    intercept: f64,
    ar_coefficients: Vec<f64>,
    ma_coefficients: Vec<f64>,
    score_start: usize,
    differenced_history: Vec<f64>,
    residual_history: Vec<f64>,
    last_differences: Vec<f64>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FittedKalmanState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedKalmanSeries>,
}

#[derive(Debug, Clone)]
struct FittedLocalLevelKalmanState {
    frame: ForecastFrame,
    series: BTreeMap<String, FittedLocalLevelKalmanSeries>,
}

#[derive(Debug, Clone)]
struct FittedKalmanSeries {
    last_timestamp: chrono::NaiveDateTime,
    level: f64,
    trend: f64,
}

#[derive(Debug, Clone)]
struct FittedLocalLevelKalmanSeries {
    last_timestamp: chrono::NaiveDateTime,
    level: f64,
}

#[derive(Debug, Clone)]
struct FittedKrigingState {
    frame: ForecastFrame,
    levels: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
struct FittedThetaSeries {
    last_timestamp: chrono::NaiveDateTime,
    n_obs: usize,
    component: ThetaComponent,
    seasonal_pattern: Option<Vec<f64>>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ThetaComponent {
    last_level: f64,
    slope: f64,
    theta: f64,
}

#[derive(Debug, Clone)]
pub struct ThetaValidationScore {
    pub theta: f64,
    pub alpha: f64,
    pub mse: f64,
}

#[derive(Debug, Clone)]
pub struct ArimaValidationScore {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub mse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanParameterSet {
    pub level_process_variance: f64,
    pub trend_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone)]
pub struct KalmanValidationScore {
    pub params: KalmanParameterSet,
    pub mse: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLevelKalmanParameterSet {
    pub level_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone)]
pub struct LocalLevelKalmanValidationScore {
    pub params: LocalLevelKalmanParameterSet,
    pub mse: f64,
}

impl NaiveForecaster {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SeasonalNaiveForecaster {
    pub fn new(season_length: usize) -> Result<Self> {
        if season_length == 0 {
            return Err(CartoBoostError::InvalidInput(
                "season_length must be positive".to_string(),
            ));
        }
        Ok(Self {
            season_length,
            fitted: None,
        })
    }
}

impl ThetaSeasonality {
    pub fn additive(season_length: usize) -> Result<Self> {
        Self::new(ThetaSeasonalityKind::Additive, season_length)
    }

    pub fn multiplicative(season_length: usize) -> Result<Self> {
        Self::new(ThetaSeasonalityKind::Multiplicative, season_length)
    }

    fn new(kind: ThetaSeasonalityKind, season_length: usize) -> Result<Self> {
        if season_length <= 1 {
            return Err(CartoBoostError::InvalidInput(
                "season_length must be greater than 1 for theta seasonality".to_string(),
            ));
        }
        Ok(Self {
            kind,
            season_length,
        })
    }

    fn name(self) -> &'static str {
        match self.kind {
            ThetaSeasonalityKind::Additive => "additive",
            ThetaSeasonalityKind::Multiplicative => "multiplicative",
        }
    }
}

impl ThetaForecaster {
    pub fn new(theta: f64, alpha: f64) -> Result<Self> {
        Self::with_seasonality(theta, alpha, None)
    }

    pub fn with_seasonality(
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        validate_theta_params(theta, alpha)?;
        Ok(Self {
            theta,
            alpha,
            seasonality,
            fitted: None,
        })
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }
}

impl OptimizedThetaForecaster {
    pub fn new(theta_grid: Vec<f64>, alpha_grid: Vec<f64>) -> Result<Self> {
        Self::with_seasonality(theta_grid, alpha_grid, None)
    }

    pub fn with_seasonality(
        theta_grid: Vec<f64>,
        alpha_grid: Vec<f64>,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        if theta_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "theta_grid must not be empty".to_string(),
            ));
        }
        if alpha_grid.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "alpha_grid must not be empty".to_string(),
            ));
        }
        for &theta in &theta_grid {
            validate_theta_params(theta, 0.5)?;
        }
        for &alpha in &alpha_grid {
            validate_theta_params(1.0, alpha)?;
        }
        Ok(Self {
            theta_grid,
            alpha_grid,
            seasonality,
            selected_theta: None,
            selected_alpha: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_theta(&self) -> Option<f64> {
        self.selected_theta
    }

    pub fn selected_alpha(&self) -> Option<f64> {
        self.selected_alpha
    }

    pub fn validation_scores(&self) -> &[ThetaValidationScore] {
        &self.validation_scores
    }
}

impl ETSForecaster {
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        Self::with_additive_seasonality(alpha, beta, None, None)
    }

    pub fn with_additive_seasonality(
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
    ) -> Result<Self> {
        validate_ets_params(alpha, beta, gamma, season_length)?;
        Ok(Self {
            alpha,
            beta,
            gamma,
            season_length,
            fitted: None,
        })
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }

    pub fn level_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.level_values.as_slice())
    }

    pub fn trend_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.trend_values.as_slice())
    }

    pub fn seasonal_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.seasonal_values.as_slice())
    }
}

impl ArimaForecaster {
    pub fn new(p: usize, d: usize, q: usize) -> Result<Self> {
        validate_arima_order(p, d, q)?;
        Ok(Self {
            p,
            d,
            q,
            fitted: None,
        })
    }

    pub fn order(&self) -> (usize, usize, usize) {
        (self.p, self.d, self.q)
    }

    pub fn fitted_values(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.fitted_values.as_slice())
    }

    pub fn residuals(&self, series_id: &str) -> Option<&[f64]> {
        self.fitted
            .as_ref()
            .and_then(|state| state.series.get(series_id))
            .map(|series| series.residuals.as_slice())
    }
}

impl AutoARIMAForecaster {
    pub fn new(max_p: usize, max_d: usize) -> Result<Self> {
        Self::with_max_order(max_p, max_d, 2)
    }

    pub fn with_max_order(max_p: usize, max_d: usize, max_q: usize) -> Result<Self> {
        if max_p > 8 {
            return Err(CartoBoostError::InvalidInput(
                "max_p must be <= 8 for auto_arima".to_string(),
            ));
        }
        if max_d > 2 {
            return Err(CartoBoostError::InvalidInput(
                "max_d must be <= 2 for auto_arima".to_string(),
            ));
        }
        if max_q > 8 {
            return Err(CartoBoostError::InvalidInput(
                "max_q must be <= 8 for auto_arima".to_string(),
            ));
        }
        Ok(Self {
            max_p,
            max_d,
            max_q,
            selected_order: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_order(&self) -> Option<(usize, usize, usize)> {
        self.selected_order
    }

    pub fn validation_scores(&self) -> &[ArimaValidationScore] {
        &self.validation_scores
    }
}

impl KalmanForecaster {
    pub fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        LocalLinearKalmanConfig::new(
            level_process_variance,
            trend_process_variance,
            observation_variance,
        )?;
        Ok(Self {
            level_process_variance,
            trend_process_variance,
            observation_variance,
            fitted: None,
        })
    }
}

impl LocalLevelKalmanForecaster {
    pub fn new(level_process_variance: f64, observation_variance: f64) -> Result<Self> {
        LocalLevelKalmanConfig::new(level_process_variance, observation_variance)?;
        Ok(Self {
            level_process_variance,
            observation_variance,
            fitted: None,
        })
    }
}

impl AutoKalmanForecaster {
    pub fn new() -> Result<Self> {
        Self::with_grids(
            vec![0.001, 0.01, 0.05, 0.1],
            vec![0.0001, 0.001, 0.005, 0.01],
            vec![0.1, 0.5, 1.0, 2.0],
            None,
        )
    }

    pub fn with_grids(
        level_process_variance_grid: Vec<f64>,
        trend_process_variance_grid: Vec<f64>,
        observation_variance_grid: Vec<f64>,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        validate_kalman_grid("level_process_variance_grid", &level_process_variance_grid)?;
        validate_kalman_grid("trend_process_variance_grid", &trend_process_variance_grid)?;
        validate_kalman_grid("observation_variance_grid", &observation_variance_grid)?;
        if matches!(validation_window, Some(0)) {
            return Err(CartoBoostError::InvalidInput(
                "auto_kalman validation_window must be positive when provided".to_string(),
            ));
        }
        Ok(Self {
            level_process_variance_grid,
            trend_process_variance_grid,
            observation_variance_grid,
            validation_window,
            selected_params: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_params(&self) -> Option<KalmanParameterSet> {
        self.selected_params
    }

    pub fn validation_scores(&self) -> &[KalmanValidationScore] {
        &self.validation_scores
    }
}

impl AutoLocalLevelKalmanForecaster {
    pub fn new() -> Result<Self> {
        Self::with_grids(vec![0.001, 0.01, 0.05, 0.1], vec![0.1, 0.5, 1.0, 2.0], None)
    }

    pub fn with_grids(
        level_process_variance_grid: Vec<f64>,
        observation_variance_grid: Vec<f64>,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        validate_kalman_grid("level_process_variance_grid", &level_process_variance_grid)?;
        validate_kalman_grid("observation_variance_grid", &observation_variance_grid)?;
        if matches!(validation_window, Some(0)) {
            return Err(CartoBoostError::InvalidInput(
                "auto_local_level_kalman validation_window must be positive when provided"
                    .to_string(),
            ));
        }
        Ok(Self {
            level_process_variance_grid,
            observation_variance_grid,
            validation_window,
            selected_params: None,
            validation_scores: Vec::new(),
            fitted: None,
        })
    }

    pub fn selected_params(&self) -> Option<LocalLevelKalmanParameterSet> {
        self.selected_params
    }

    pub fn validation_scores(&self) -> &[LocalLevelKalmanValidationScore] {
        &self.validation_scores
    }
}

impl Default for AutoLocalLevelKalmanForecaster {
    fn default() -> Self {
        Self::new().expect("default auto_local_level_kalman grid is valid")
    }
}

impl Default for AutoKalmanForecaster {
    fn default() -> Self {
        Self::new().expect("default auto_kalman grid is valid")
    }
}

impl Default for KalmanForecaster {
    fn default() -> Self {
        Self {
            level_process_variance: 0.05,
            trend_process_variance: 0.005,
            observation_variance: 1.0,
            fitted: None,
        }
    }
}

impl Default for LocalLevelKalmanForecaster {
    fn default() -> Self {
        Self {
            level_process_variance: 0.05,
            observation_variance: 1.0,
            fitted: None,
        }
    }
}

impl KrigingForecaster {
    pub fn new(coordinates: BTreeMap<String, (f64, f64)>, range: f64, nugget: f64) -> Result<Self> {
        Self::with_config(coordinates, OrdinaryKrigingConfig::new(range, nugget)?)
    }

    pub fn with_config(
        coordinates: BTreeMap<String, (f64, f64)>,
        config: OrdinaryKrigingConfig,
    ) -> Result<Self> {
        if coordinates.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "kriging coordinates must not be empty".to_string(),
            ));
        }
        for (series_id, (x, y)) in &coordinates {
            if !x.is_finite() || !y.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "kriging coordinate for series {series_id} must be finite"
                )));
            }
        }
        Ok(Self {
            coordinates,
            config,
            fitted: None,
        })
    }
}

impl Forecaster for NaiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedLocalState::from_frame(frame));
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean: last.target,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "naive"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name()})
    }
}

impl Forecaster for SeasonalNaiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let fitted = FittedLocalState::from_frame(frame);
        for (series_id, history) in &fitted.history_by_series {
            if history.len() < self.season_length {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} has {} rows, but seasonal naive requires at least {}",
                    history.len(),
                    self.season_length
                )));
            }
        }
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let base = history.len() - self.season_length;
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let seasonal_index = base + ((step - 1) % self.season_length);
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean: history[seasonal_index].target,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "seasonal_naive"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name(), "season_length": self.season_length})
    }
}

impl Forecaster for ThetaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedThetaState::from_frame(
            frame,
            self.theta,
            self.alpha,
            self.seasonality,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "theta"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "theta": self.theta,
            "alpha": self.alpha,
            "seasonality": self.seasonality.map(ThetaSeasonality::name),
            "season_length": self.seasonality.map(|seasonality| seasonality.season_length),
        })
    }
}

impl Forecaster for OptimizedThetaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let candidates = self
            .theta_grid
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(theta_idx, theta)| {
                self.alpha_grid
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(alpha_idx, alpha)| (theta_idx, alpha_idx, theta, alpha))
            })
            .collect::<Vec<_>>();
        let scored = candidates
            .into_par_iter()
            .map(|(theta_idx, alpha_idx, theta, alpha)| {
                let fitted = FittedThetaState::from_frame(frame, theta, alpha, self.seasonality)?;
                let mse = fitted.mean_squared_residual();
                Ok((
                    theta_idx,
                    alpha_idx,
                    ThetaValidationScore { theta, alpha, mse },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(theta_idx, alpha_idx, _)| (*theta_idx, *alpha_idx));
        let scores = scored
            .into_iter()
            .map(|(_, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores
            .iter()
            .map(|score| {
                (
                    OrderedF64(score.mse),
                    OrderedF64(score.theta),
                    OrderedF64(score.alpha),
                )
            })
            .min();
        let (_, theta, alpha) = best.ok_or_else(|| {
            CartoBoostError::InvalidInput("theta validation grid must not be empty".to_string())
        })?;
        let mut fitted = ThetaForecaster::with_seasonality(theta.0, alpha.0, self.seasonality)?;
        fitted.fit(frame)?;
        self.selected_theta = Some(theta.0);
        self.selected_alpha = Some(alpha.0);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "optimized_theta"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "selected_theta": self.selected_theta,
            "selected_alpha": self.selected_alpha,
            "seasonality": self.seasonality.map(ThetaSeasonality::name),
            "season_length": self.seasonality.map(|seasonality| seasonality.season_length),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({"theta": score.theta, "alpha": score.alpha, "mse": score.mse})
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for ETSForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedETSState::from_frame(
            frame,
            self.alpha,
            self.beta,
            self.gamma,
            self.season_length,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        let seasonal = series
                            .seasonals
                            .as_ref()
                            .map(|seasonals| seasonals[(series.n_obs + step - 1) % seasonals.len()])
                            .unwrap_or(0.0);
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: self.model_name().to_string(),
                            mean: series.level + step as f64 * series.trend + seasonal,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "ets"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "season_length": self.season_length,
        })
    }
}

impl Forecaster for ArimaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedArimaState::from_frame(frame, self.p, self.d, self.q)?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "arima"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name(), "p": self.p, "d": self.d, "q": self.q})
    }
}

impl Forecaster for AutoARIMAForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let max_p = self.max_p;
        let max_d = self.max_d;
        let max_q = self.max_q;
        let mut scores = (0..=max_d)
            .into_par_iter()
            .flat_map_iter(|d| (0..=max_p).flat_map(move |p| (0..=max_q).map(move |q| (p, d, q))))
            .map(|(p, d, q)| {
                let fitted = FittedArimaState::from_frame(frame, p, d, q)?;
                let mse = fitted.mean_squared_residual();
                Ok(ArimaValidationScore { p, d, q, mse })
            })
            .collect::<Result<Vec<_>>>()?;
        scores.sort_by_key(|score| (score.d, score.p, score.q));
        let best = scores
            .iter()
            .map(|score| (OrderedF64(score.mse), score.p, score.d, score.q))
            .min();
        let (_, p, d, q) = best.ok_or_else(|| {
            CartoBoostError::InvalidInput("auto_arima candidate grid must not be empty".to_string())
        })?;
        let mut fitted = ArimaForecaster::new(p, d, q)?;
        fitted.fit(frame)?;
        self.selected_order = Some((p, d, q));
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_arima"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "max_p": self.max_p,
            "max_d": self.max_d,
            "max_q": self.max_q,
            "selected_order": self.selected_order.map(|(p, d, q)| json!({"p": p, "d": d, "q": q})),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({"p": score.p, "d": score.d, "q": score.q, "mse": score.mse})
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for KalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedKalmanState::from_frame(
            frame,
            self.level_process_variance,
            self.trend_process_variance,
            self.observation_variance,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "level_process_variance": self.level_process_variance,
            "trend_process_variance": self.trend_process_variance,
            "observation_variance": self.observation_variance,
        })
    }
}

impl Forecaster for LocalLevelKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedLocalLevelKalmanState::from_frame(
            frame,
            self.level_process_variance,
            self.observation_variance,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "local_level_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "level_process_variance": self.level_process_variance,
            "observation_variance": self.observation_variance,
        })
    }
}

impl Forecaster for AutoKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let local = FittedLocalState::from_frame(frame);
        let level_grid = self.level_process_variance_grid.clone();
        let trend_grid = self.trend_process_variance_grid.clone();
        let observation_grid = self.observation_variance_grid.clone();
        let validation_window = self.validation_window;
        let mut candidates = Vec::new();
        for (level_idx, level_process_variance) in level_grid.iter().copied().enumerate() {
            for (trend_idx, trend_process_variance) in trend_grid.iter().copied().enumerate() {
                for (observation_idx, observation_variance) in
                    observation_grid.iter().copied().enumerate()
                {
                    candidates.push((
                        level_idx,
                        trend_idx,
                        observation_idx,
                        KalmanParameterSet {
                            level_process_variance,
                            trend_process_variance,
                            observation_variance,
                        },
                    ));
                }
            }
        }
        let scored = candidates
            .into_par_iter()
            .map(|(level_idx, trend_idx, observation_idx, params)| {
                let mse = score_kalman_params(&local.history_by_series, params, validation_window)?;
                Ok((
                    level_idx,
                    trend_idx,
                    observation_idx,
                    KalmanValidationScore { params, mse },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(level_idx, trend_idx, observation_idx, _)| {
            (*level_idx, *trend_idx, *observation_idx)
        });
        let scores = scored
            .into_iter()
            .map(|(_, _, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores.iter().min_by_key(|score| {
            (
                OrderedF64(score.mse),
                OrderedF64(score.params.level_process_variance),
                OrderedF64(score.params.trend_process_variance),
                OrderedF64(score.params.observation_variance),
            )
        });
        let params = best
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "auto_kalman candidate grid must not be empty".to_string(),
                )
            })?
            .params;
        let mut fitted = KalmanForecaster::new(
            params.level_process_variance,
            params.trend_process_variance,
            params.observation_variance,
        )?;
        fitted.fit(frame)?;
        self.selected_params = Some(params);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "validation_window": self.validation_window,
            "selected_params": self.selected_params.map(|params| json!({
                "level_process_variance": params.level_process_variance,
                "trend_process_variance": params.trend_process_variance,
                "observation_variance": params.observation_variance,
            })),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "level_process_variance": score.params.level_process_variance,
                    "trend_process_variance": score.params.trend_process_variance,
                    "observation_variance": score.params.observation_variance,
                    "mse": score.mse,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for AutoLocalLevelKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let local = FittedLocalState::from_frame(frame);
        let candidates = self
            .level_process_variance_grid
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(level_idx, level_process_variance)| {
                self.observation_variance_grid
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(observation_idx, observation_variance)| {
                        (
                            level_idx,
                            observation_idx,
                            LocalLevelKalmanParameterSet {
                                level_process_variance,
                                observation_variance,
                            },
                        )
                    })
            })
            .collect::<Vec<_>>();
        let scored = candidates
            .into_par_iter()
            .map(|(level_idx, observation_idx, params)| {
                let mse = score_local_level_kalman_params(
                    &local.history_by_series,
                    params,
                    self.validation_window,
                )?;
                Ok((
                    level_idx,
                    observation_idx,
                    LocalLevelKalmanValidationScore { params, mse },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(level_idx, observation_idx, _)| (*level_idx, *observation_idx));
        let scores = scored
            .into_iter()
            .map(|(_, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores.iter().min_by_key(|score| {
            (
                OrderedF64(score.mse),
                OrderedF64(score.params.level_process_variance),
                OrderedF64(score.params.observation_variance),
            )
        });
        let params = best
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "auto_local_level_kalman candidate grid must not be empty".to_string(),
                )
            })?
            .params;
        let mut fitted = LocalLevelKalmanForecaster::new(
            params.level_process_variance,
            params.observation_variance,
        )?;
        fitted.fit(frame)?;
        self.selected_params = Some(params);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_local_level_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "validation_window": self.validation_window,
            "selected_params": self.selected_params.map(|params| json!({
                "level_process_variance": params.level_process_variance,
                "observation_variance": params.observation_variance,
            })),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "level_process_variance": score.params.level_process_variance,
                    "observation_variance": score.params.observation_variance,
                    "mse": score.mse,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for KrigingForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedKrigingState::from_frame(frame, &self.coordinates)?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let observations = fitted
            .levels
            .iter()
            .map(|(series_id, value)| {
                let coord = self.coordinates.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing kriging coordinate for series {series_id}"
                    ))
                })?;
                Ok(KrigingObservation {
                    x: coord.0,
                    y: coord.1,
                    value: *value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let series_ids = fitted.frame.series_ids();
        let targets = series_ids
            .iter()
            .map(|series_id| {
                self.coordinates.get(series_id).copied().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing kriging coordinate for series {series_id}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let means = ordinary_kriging_predict_many(&observations, &targets, self.config)?
            .into_iter()
            .map(|prediction| prediction.mean)
            .collect::<Vec<_>>();
        let predictions = series_ids
            .into_par_iter()
            .zip(means)
            .map(|(series_id, mean)| {
                let history = fitted.frame.rows_for_series(&series_id);
                let last_timestamp = history
                    .last()
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput("empty series history".to_string())
                    })?
                    .timestamp;
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted.frame.frequency().advance(last_timestamp, step)?,
                            horizon: step,
                            model: self.model_name().to_string(),
                            mean,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "kriging"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "range": self.config.range,
            "nugget": self.config.nugget,
            "sill": self.config.sill,
            "variogram_model": format!("{:?}", self.config.variogram_model).to_lowercase(),
            "drift": format!("{:?}", self.config.drift).to_lowercase(),
            "anisotropy_angle_degrees": self.config.anisotropy_angle_degrees,
            "anisotropy_scaling": self.config.anisotropy_scaling,
            "max_neighbors": self.config.max_neighbors,
            "min_neighbors": self.config.min_neighbors,
            "max_distance": self.config.max_distance,
            "series_count": self.coordinates.len(),
        })
    }
}

impl ThetaForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        let adjusted = forecast_theta_component(&series.component, step);
                        let mean = reseasonalize_value(
                            adjusted,
                            series.n_obs + step - 1,
                            self.seasonality,
                            series.seasonal_pattern.as_deref(),
                        )?;
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl ArimaForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                series
                    .forecast_values(horizon)
                    .into_iter()
                    .enumerate()
                    .map(|(idx, mean)| {
                        let step = idx + 1;
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl FittedLocalState {
    fn from_frame(frame: &ForecastFrame) -> Self {
        let mut history_by_series: BTreeMap<String, Vec<ForecastRow>> = BTreeMap::new();
        for row in frame.rows() {
            history_by_series
                .entry(row.series_id.clone())
                .or_default()
                .push(row.clone());
        }
        Self {
            frame: frame.clone(),
            history_by_series,
        }
    }
}

impl FittedETSState {
    fn from_frame(
        frame: &ForecastFrame,
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedETSSeries::fit(series_id, history, alpha, beta, gamma, season_length)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }
}

impl FittedArimaState {
    fn from_frame(frame: &ForecastFrame, p: usize, d: usize, q: usize) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedArimaSeries::fit(series_id, history, p, d, q)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .skip(series.score_start)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

impl KalmanForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean: series.level + step as f64 * series.trend,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl LocalLevelKalmanForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean: series.level,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl FittedKalmanState {
    fn from_frame(
        frame: &ForecastFrame,
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedKalmanSeries::fit(
                        series_id,
                        history,
                        level_process_variance,
                        trend_process_variance,
                        observation_variance,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }
}

impl FittedLocalLevelKalmanState {
    fn from_frame(
        frame: &ForecastFrame,
        level_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedLocalLevelKalmanSeries::fit(
                        series_id,
                        history,
                        level_process_variance,
                        observation_variance,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }
}

impl FittedKalmanSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for kalman forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let config = LocalLinearKalmanConfig::new(
            level_process_variance,
            trend_process_variance,
            observation_variance,
        )?;
        let result = fit_local_linear_kalman(&values, config)
            .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
        Ok(Self {
            last_timestamp: history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?
                .timestamp,
            level: result.final_state.level,
            trend: result.final_state.trend,
        })
    }
}

impl FittedLocalLevelKalmanSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        level_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        if history.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least one row for local level kalman forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let config = LocalLevelKalmanConfig::new(level_process_variance, observation_variance)?;
        let result = fit_local_level_kalman(&values, config)
            .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            level: result.final_level,
        })
    }
}

impl FittedKrigingState {
    fn from_frame(
        frame: &ForecastFrame,
        coordinates: &BTreeMap<String, (f64, f64)>,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let mut levels = BTreeMap::new();
        for (series_id, history) in &local.history_by_series {
            if !coordinates.contains_key(series_id) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "missing kriging coordinate for series {series_id}"
                )));
            }
            let last = history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?;
            levels.insert(series_id.clone(), last.target);
        }
        Ok(Self {
            frame: frame.clone(),
            levels,
        })
    }
}

impl FittedETSSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for ETS forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let mut seasonals = match season_length {
            Some(length) => {
                if values.len() < length * 2 {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "series {series_id} requires at least two full seasonal cycles for ETS seasonality"
                    )));
                }
                let (_, pattern) = deseasonalize(
                    series_id,
                    &values,
                    Some(ThetaSeasonality::additive(length)?),
                )?;
                pattern
            }
            None => None,
        };

        let mut level = values[0] - seasonals.as_ref().map(|s| s[0]).unwrap_or(0.0);
        let mut trend = initial_trend(&values, seasonals.as_deref());
        let mut fitted_values = Vec::with_capacity(values.len());
        let mut residuals = Vec::with_capacity(values.len());
        let mut level_values = Vec::with_capacity(values.len());
        let mut trend_values = Vec::with_capacity(values.len());
        let mut seasonal_values = Vec::with_capacity(values.len());
        fitted_values.push(values[0]);
        residuals.push(0.0);
        level_values.push(level);
        trend_values.push(trend);
        seasonal_values.push(seasonals.as_ref().map(|s| s[0]).unwrap_or(0.0));

        for (idx, value) in values.iter().enumerate().skip(1) {
            let seasonal_idx = seasonals.as_ref().map(|seasonals| idx % seasonals.len());
            let seasonal = seasonal_idx
                .and_then(|seasonal_idx| {
                    seasonals.as_ref().map(|seasonals| seasonals[seasonal_idx])
                })
                .unwrap_or(0.0);
            let fitted = level + trend + seasonal;
            fitted_values.push(fitted);
            residuals.push(*value - fitted);

            let previous_level = level;
            level = alpha * (*value - seasonal) + (1.0 - alpha) * (level + trend);
            trend = beta * (level - previous_level) + (1.0 - beta) * trend;
            if let (Some(gamma), Some(seasonal_idx), Some(seasonals)) =
                (gamma, seasonal_idx, seasonals.as_mut())
            {
                seasonals[seasonal_idx] =
                    gamma * (*value - level) + (1.0 - gamma) * seasonals[seasonal_idx];
            }
            level_values.push(level);
            trend_values.push(trend);
            seasonal_values.push(seasonal);
        }

        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            n_obs: history.len(),
            level,
            trend,
            seasonals,
            fitted_values,
            residuals,
            level_values,
            trend_values,
            seasonal_values,
        })
    }
}

impl FittedArimaSeries {
    fn fit(series_id: &str, history: &[ForecastRow], p: usize, d: usize, q: usize) -> Result<Self> {
        validate_arima_order(p, d, q)?;
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let differences = difference_series(&values, d)?;
        let required_lags = p.max(q);
        if differences.len() <= required_lags {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} has {} differenced rows, but ARIMA({p},{d},{q}) requires more than {required_lags}",
                differences.len(),
            )));
        }
        let (intercept, ar_coefficients, ma_coefficients, fitted_diff, residuals) =
            fit_arima_components(&differences, p, q);
        let fitted_values = undifference_fitted_values(&values, &fitted_diff, d);
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            intercept,
            ar_coefficients,
            ma_coefficients,
            score_start: required_lags,
            differenced_history: differences,
            residual_history: residuals.clone(),
            last_differences: last_differences(&values, d)?,
            fitted_values,
            residuals,
        })
    }

    fn forecast_values(&self, horizon: usize) -> Vec<f64> {
        let p = self.ar_coefficients.len();
        let q = self.ma_coefficients.len();
        let mut differenced = tail_values(&self.differenced_history, p);
        let mut residuals = tail_values(&self.residual_history, q);
        let mut levels = self.last_differences.clone();
        let mut forecasts = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let next_diff = forecast_arima_next(
                &differenced,
                &residuals,
                self.intercept,
                &self.ar_coefficients,
                &self.ma_coefficients,
            );
            push_tail(&mut differenced, p, next_diff);
            push_tail(&mut residuals, q, 0.0);
            let mut value = next_diff;
            for idx in (0..(levels.len() - 1)).rev() {
                levels[idx] += value;
                value = levels[idx];
            }
            forecasts.push(value);
        }
        forecasts
    }
}

impl FittedThetaState {
    fn from_frame(
        frame: &ForecastFrame,
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedThetaSeries::fit(series_id, history, theta, alpha, seasonality)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .skip(1)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

impl FittedThetaSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for theta forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let (adjusted, pattern) = deseasonalize(series_id, &values, seasonality)?;
        let component = fit_theta_component(&adjusted, theta, alpha);
        let fitted_adjusted = fitted_theta_values(&adjusted, alpha);
        let mut fitted_values = Vec::with_capacity(values.len());
        let mut residuals = Vec::with_capacity(values.len());
        for (idx, fitted) in fitted_adjusted.into_iter().enumerate() {
            let reseasonalized = reseasonalize_value(fitted, idx, seasonality, pattern.as_deref())?;
            fitted_values.push(reseasonalized);
            residuals.push(values[idx] - reseasonalized);
        }
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            n_obs: history.len(),
            component,
            seasonal_pattern: pattern,
            fitted_values,
            residuals,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("theta grid scores are finite")
    }
}

fn validate_theta_params(theta: f64, alpha: f64) -> Result<()> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "theta must be a positive finite value".to_string(),
        ));
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
        return Err(CartoBoostError::InvalidInput(
            "alpha must be finite and in (0, 1]".to_string(),
        ));
    }
    Ok(())
}

fn validate_kalman_grid(name: &str, values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "auto_kalman {name} must not be empty"
        )));
    }
    for &value in values {
        if !value.is_finite() || value <= 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "auto_kalman {name} values must be positive finite numbers"
            )));
        }
    }
    Ok(())
}

fn score_kalman_params(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    params: KalmanParameterSet,
    validation_window: Option<usize>,
) -> Result<f64> {
    let config = LocalLinearKalmanConfig::new(
        params.level_process_variance,
        params.trend_process_variance,
        params.observation_variance,
    )?;
    let (sum_squared_error, count) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            let window = validation_window.unwrap_or_else(|| {
                let suggested = history.len() / 5;
                suggested.clamp(1, 12)
            });
            if history.len() <= window + 1 {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} has {} rows, but auto_kalman requires at least {} rows for validation",
                    history.len(),
                    window + 2
                )));
            }
            let train_len = history.len() - window;
            let train = history[..train_len]
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>();
            let result = fit_local_linear_kalman(&train, config)
                .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
            let mut sum = 0.0;
            for (idx, row) in history[train_len..].iter().enumerate() {
                let step = idx + 1;
                let mean = result.final_state.level + step as f64 * result.final_state.trend;
                let residual = row.target - mean;
                sum += residual * residual;
            }
            Ok((sum, window))
        })
        .reduce(
            || Ok((0.0, 0usize)),
            |left: Result<(f64, usize)>, right: Result<(f64, usize)>| {
                let (left_sum, left_count) = left?;
                let (right_sum, right_count) = right?;
                Ok((left_sum + right_sum, left_count + right_count))
            },
        )?;
    if count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto_kalman validation produced no scored observations".to_string(),
        ));
    }
    let mse = sum_squared_error / count as f64;
    if !mse.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "auto_kalman validation score must be finite".to_string(),
        ));
    }
    Ok(mse)
}

fn score_local_level_kalman_params(
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    params: LocalLevelKalmanParameterSet,
    validation_window: Option<usize>,
) -> Result<f64> {
    let config =
        LocalLevelKalmanConfig::new(params.level_process_variance, params.observation_variance)?;
    let (sum_squared_error, count) = history_by_series
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(series_id, history)| {
            let window = validation_window.unwrap_or_else(|| {
                let suggested = history.len() / 5;
                suggested.clamp(1, 12)
            });
            if history.len() <= window {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} has {} rows, but auto_local_level_kalman requires at least {} rows for validation",
                    history.len(),
                    window + 1
                )));
            }
            let train_len = history.len() - window;
            let train = history[..train_len]
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>();
            let result = fit_local_level_kalman(&train, config)
                .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
            let mut sum = 0.0;
            for row in &history[train_len..] {
                let residual = row.target - result.final_level;
                sum += residual * residual;
            }
            Ok((sum, window))
        })
        .reduce(
            || Ok((0.0, 0usize)),
            |left: Result<(f64, usize)>, right: Result<(f64, usize)>| {
                let (left_sum, left_count) = left?;
                let (right_sum, right_count) = right?;
                Ok((left_sum + right_sum, left_count + right_count))
            },
        )?;
    if count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto_local_level_kalman validation produced no scored observations".to_string(),
        ));
    }
    let mse = sum_squared_error / count as f64;
    if !mse.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "auto_local_level_kalman validation score must be finite".to_string(),
        ));
    }
    Ok(mse)
}

fn validate_ets_params(
    alpha: f64,
    beta: f64,
    gamma: Option<f64>,
    season_length: Option<usize>,
) -> Result<()> {
    validate_unit_interval("alpha", alpha, false)?;
    validate_unit_interval("beta", beta, true)?;
    if let Some(gamma) = gamma {
        validate_unit_interval("gamma", gamma, true)?;
    }
    match (gamma, season_length) {
        (Some(_), Some(length)) if length > 1 => Ok(()),
        (None, None) => Ok(()),
        (Some(_), None) => Err(CartoBoostError::InvalidInput(
            "ETS gamma requires season_length".to_string(),
        )),
        (None, Some(_)) => Err(CartoBoostError::InvalidInput(
            "ETS season_length requires gamma".to_string(),
        )),
        (Some(_), Some(_)) => Err(CartoBoostError::InvalidInput(
            "ETS season_length must be greater than 1".to_string(),
        )),
    }
}

fn validate_unit_interval(name: &str, value: f64, allow_zero: bool) -> Result<()> {
    let lower_ok = if allow_zero {
        value >= 0.0
    } else {
        value > 0.0
    };
    if !value.is_finite() || !lower_ok || value > 1.0 {
        let range = if allow_zero { "[0, 1]" } else { "(0, 1]" };
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be finite and in {range}"
        )));
    }
    Ok(())
}

fn validate_arima_order(p: usize, d: usize, q: usize) -> Result<()> {
    if p > MAX_ARIMA_ORDER {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA p must be <= 8".to_string(),
        ));
    }
    if d > 2 {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        ));
    }
    if q > MAX_ARIMA_ORDER {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA q must be <= 8".to_string(),
        ));
    }
    Ok(())
}

fn initial_trend(values: &[f64], seasonals: Option<&[f64]>) -> f64 {
    match seasonals {
        Some(seasonals) if values.len() > seasonals.len() => {
            let length = seasonals.len();
            let mut sum = 0.0;
            for idx in 0..length {
                sum += (values[idx + length] - values[idx]) / length as f64;
            }
            sum / length as f64
        }
        Some(seasonals) => {
            (values[1] - seasonals[1 % seasonals.len()]) - (values[0] - seasonals[0])
        }
        None => values[1] - values[0],
    }
}

fn difference_series(values: &[f64], d: usize) -> Result<Vec<f64>> {
    if values.len() <= d {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA differencing order leaves no observations".to_string(),
        ));
    }
    match d {
        0 => Ok(values.to_vec()),
        1 => {
            let mut differences = Vec::with_capacity(values.len() - 1);
            for idx in 1..values.len() {
                differences.push(values[idx] - values[idx - 1]);
            }
            Ok(differences)
        }
        2 => {
            let mut differences = Vec::with_capacity(values.len() - 2);
            for idx in 2..values.len() {
                differences.push(values[idx] - 2.0 * values[idx - 1] + values[idx - 2]);
            }
            Ok(differences)
        }
        _ => Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        )),
    }
}

fn last_differences(values: &[f64], d: usize) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA requires at least one observation".to_string(),
        ));
    }
    if values.len() <= d {
        return Err(CartoBoostError::InvalidInput(
            "ARIMA differencing order leaves no observations".to_string(),
        ));
    }
    let last = values[values.len() - 1];
    match d {
        0 => Ok(vec![last]),
        1 => Ok(vec![last, last - values[values.len() - 2]]),
        2 => {
            let n = values.len();
            Ok(vec![
                last,
                last - values[n - 2],
                last - 2.0 * values[n - 2] + values[n - 3],
            ])
        }
        _ => Err(CartoBoostError::InvalidInput(
            "ARIMA d must be <= 2".to_string(),
        )),
    }
}

fn tail_values(values: &[f64], length: usize) -> Vec<f64> {
    if length == 0 {
        Vec::new()
    } else {
        values[values.len().saturating_sub(length)..].to_vec()
    }
}

fn push_tail(values: &mut Vec<f64>, max_len: usize, value: f64) {
    if max_len == 0 {
        return;
    }
    if values.len() == max_len {
        values.rotate_left(1);
        if let Some(last) = values.last_mut() {
            *last = value;
        }
    } else {
        values.push(value);
    }
}

fn arima_feature(values: &[f64], residuals: &[f64], idx: usize, p: usize, col: usize) -> f64 {
    if col == 0 {
        1.0
    } else if col <= p {
        values[idx - col]
    } else {
        residuals[idx - (col - p)]
    }
}

fn solve_arima_normal_equations(
    mut matrix: [[f64; MAX_ARIMA_COLUMNS]; MAX_ARIMA_COLUMNS],
    mut rhs: [f64; MAX_ARIMA_COLUMNS],
    n: usize,
) -> Option<Vec<f64>> {
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        for row in (pivot_idx + 1)..n {
            if matrix[row][pivot_idx].abs() > matrix[pivot_row][pivot_idx].abs() {
                pivot_row = row;
            }
        }
        if matrix[pivot_row][pivot_idx].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(pivot_idx, pivot_row);
        rhs.swap(pivot_idx, pivot_row);

        let pivot = matrix[pivot_idx][pivot_idx];
        for value in matrix[pivot_idx].iter_mut().take(n).skip(pivot_idx) {
            *value /= pivot;
        }
        rhs[pivot_idx] /= pivot;
        let pivot_tail = matrix[pivot_idx];
        let pivot_rhs = rhs[pivot_idx];

        for row in 0..n {
            if row == pivot_idx {
                continue;
            }
            let factor = matrix[row][pivot_idx];
            for (col, pivot_cell) in pivot_tail.iter().enumerate().take(n).skip(pivot_idx) {
                matrix[row][col] -= factor * pivot_cell;
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }
    Some(rhs[..n].to_vec())
}

fn fit_arima_components(
    values: &[f64],
    p: usize,
    q: usize,
) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut residuals = vec![0.0; values.len()];
    let mut intercept = values.iter().sum::<f64>() / values.len() as f64;
    let mut ar_coefficients = vec![0.0; p];
    let mut ma_coefficients = vec![0.0; q];
    let iterations = if q == 0 { 1 } else { 6 };
    for _ in 0..iterations {
        let (next_intercept, next_ar, next_ma) =
            fit_arima_coefficients_once(values, &residuals, p, q);
        intercept = next_intercept;
        ar_coefficients = next_ar;
        ma_coefficients = next_ma;
        let (fitted, next_residuals) = fitted_arima_values(
            values,
            intercept,
            &ar_coefficients,
            &ma_coefficients,
            &residuals,
        );
        let _ = fitted;
        residuals = next_residuals;
    }
    let (fitted, residuals) = fitted_arima_values(
        values,
        intercept,
        &ar_coefficients,
        &ma_coefficients,
        &residuals,
    );
    (
        intercept,
        ar_coefficients,
        ma_coefficients,
        fitted,
        residuals,
    )
}

fn fit_arima_coefficients_once(
    values: &[f64],
    residuals: &[f64],
    p: usize,
    q: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    if p == 0 && q == 0 {
        return (
            values.iter().sum::<f64>() / values.len() as f64,
            Vec::new(),
            Vec::new(),
        );
    }
    let cols = p + q + 1;
    let mut xtx = [[0.0; MAX_ARIMA_COLUMNS]; MAX_ARIMA_COLUMNS];
    let mut xty = [0.0; MAX_ARIMA_COLUMNS];
    let start = p.max(q);
    for idx in start..values.len() {
        for row in 0..cols {
            let row_value = arima_feature(values, residuals, idx, p, row);
            xty[row] += row_value * values[idx];
            for (col, cell) in xtx[row].iter_mut().enumerate().take(cols) {
                *cell += row_value * arima_feature(values, residuals, idx, p, col);
            }
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().take(cols) {
        row[idx] += 1.0e-8;
    }
    let solution = solve_arima_normal_equations(xtx, xty, cols).unwrap_or_else(|| vec![0.0; cols]);
    (
        solution[0],
        solution[1..=p].to_vec(),
        solution[(p + 1)..].to_vec(),
    )
}

fn fitted_arima_values(
    values: &[f64],
    intercept: f64,
    ar_coefficients: &[f64],
    ma_coefficients: &[f64],
    residual_history: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let p = ar_coefficients.len();
    let q = ma_coefficients.len();
    let mut fitted = Vec::with_capacity(values.len());
    let mut residuals = vec![0.0; values.len()];
    for idx in 0..values.len() {
        let start = p.max(q);
        let mean = if idx < start {
            values[idx]
        } else {
            let mut mean = intercept;
            for (coef_idx, coef) in ar_coefficients.iter().enumerate() {
                mean += coef * values[idx - coef_idx - 1];
            }
            for (coef_idx, coef) in ma_coefficients.iter().enumerate() {
                mean += coef * residual_history[idx - coef_idx - 1];
            }
            mean
        };
        fitted.push(mean);
        if idx >= start {
            residuals[idx] = values[idx] - mean;
        }
    }
    (fitted, residuals)
}

fn forecast_arima_next(
    history: &[f64],
    residuals: &[f64],
    intercept: f64,
    ar_coefficients: &[f64],
    ma_coefficients: &[f64],
) -> f64 {
    let mut forecast = intercept;
    for (idx, coef) in ar_coefficients.iter().enumerate() {
        forecast += coef * history[history.len() - idx - 1];
    }
    for (idx, coef) in ma_coefficients.iter().enumerate() {
        forecast += coef * residuals[residuals.len() - idx - 1];
    }
    forecast
}

fn undifference_fitted_values(values: &[f64], fitted_diff: &[f64], d: usize) -> Vec<f64> {
    match d {
        0 => fitted_diff.to_vec(),
        1 => {
            let mut fitted = vec![values[0]];
            for idx in 1..values.len() {
                fitted.push(values[idx - 1] + fitted_diff[idx - 1]);
            }
            fitted
        }
        2 => {
            let first_diff = values
                .windows(2)
                .map(|window| window[1] - window[0])
                .collect::<Vec<_>>();
            let mut fitted = vec![values[0], values[1]];
            for idx in 2..values.len() {
                fitted.push(values[idx - 1] + first_diff[idx - 2] + fitted_diff[idx - 2]);
            }
            fitted
        }
        _ => values.to_vec(),
    }
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        for row in (pivot_idx + 1)..n {
            if matrix[row][pivot_idx].abs() > matrix[pivot_row][pivot_idx].abs() {
                pivot_row = row;
            }
        }
        if matrix[pivot_row][pivot_idx].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(pivot_idx, pivot_row);
        rhs.swap(pivot_idx, pivot_row);

        let pivot = matrix[pivot_idx][pivot_idx];
        for cell in matrix[pivot_idx].iter_mut().take(n).skip(pivot_idx) {
            *cell /= pivot;
        }
        rhs[pivot_idx] /= pivot;
        let pivot_tail = matrix[pivot_idx][pivot_idx..n].to_vec();

        for row in 0..n {
            if row == pivot_idx {
                continue;
            }
            let factor = matrix[row][pivot_idx];
            for (cell, pivot_cell) in matrix[row]
                .iter_mut()
                .take(n)
                .skip(pivot_idx)
                .zip(pivot_tail.iter())
            {
                *cell -= factor * pivot_cell;
            }
            rhs[row] -= factor * rhs[pivot_idx];
        }
    }
    Some(rhs)
}

fn deseasonalize(
    series_id: &str,
    values: &[f64],
    seasonality: Option<ThetaSeasonality>,
) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let Some(seasonality) = seasonality else {
        return Ok((values.to_vec(), None));
    };
    if values.len() < seasonality.season_length * 2 {
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} requires at least two full seasonal cycles for theta seasonality"
        )));
    }
    if seasonality.kind == ThetaSeasonalityKind::Multiplicative
        && values.iter().any(|value| *value <= 0.0)
    {
        return Err(CartoBoostError::InvalidInput(format!(
            "series {series_id} uses multiplicative seasonality but contains non-positive values"
        )));
    }

    let mut pattern = vec![0.0; seasonality.season_length];
    let mut counts = vec![0usize; seasonality.season_length];
    for (idx, value) in values.iter().enumerate() {
        let season_idx = idx % seasonality.season_length;
        pattern[season_idx] += *value;
        counts[season_idx] += 1;
    }
    for (slot, count) in pattern.iter_mut().zip(counts) {
        *slot /= count as f64;
    }

    match seasonality.kind {
        ThetaSeasonalityKind::Additive => {
            let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
            for slot in &mut pattern {
                *slot -= mean;
            }
            let adjusted = values
                .iter()
                .enumerate()
                .map(|(idx, value)| value - pattern[idx % pattern.len()])
                .collect();
            Ok((adjusted, Some(pattern)))
        }
        ThetaSeasonalityKind::Multiplicative => {
            let series_mean = values.iter().sum::<f64>() / values.len() as f64;
            for slot in &mut pattern {
                *slot /= series_mean;
            }
            let pattern_mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
            for slot in &mut pattern {
                *slot /= pattern_mean;
            }
            let adjusted = values
                .iter()
                .enumerate()
                .map(|(idx, value)| value / pattern[idx % pattern.len()])
                .collect();
            Ok((adjusted, Some(pattern)))
        }
    }
}

fn fit_theta_component(values: &[f64], theta: f64, alpha: f64) -> ThetaComponent {
    let slope = linear_slope(values);
    let levels = ses_one_step_levels(values, alpha);
    let last_level = alpha * values[values.len() - 1] + (1.0 - alpha) * levels[values.len() - 1];
    ThetaComponent {
        last_level,
        slope,
        theta,
    }
}

fn fitted_theta_values(values: &[f64], alpha: f64) -> Vec<f64> {
    let levels = ses_one_step_levels(values, alpha);
    let slope = linear_slope(values);
    let intercept = values.iter().sum::<f64>() / values.len() as f64
        - slope * ((values.len() - 1) as f64 / 2.0);
    levels
        .iter()
        .enumerate()
        .map(|(idx, level)| 0.5 * (level + intercept + slope * idx as f64))
        .collect()
}

fn ses_one_step_levels(values: &[f64], alpha: f64) -> Vec<f64> {
    let mut levels = Vec::with_capacity(values.len());
    levels.push(values[0]);
    for idx in 1..values.len() {
        levels.push(alpha * values[idx - 1] + (1.0 - alpha) * levels[idx - 1]);
    }
    levels
}

fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let x_mean = (values.len() - 1) as f64 / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (idx, value) in values.iter().enumerate() {
        let x_delta = idx as f64 - x_mean;
        numerator += x_delta * (value - y_mean);
        denominator += x_delta * x_delta;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn forecast_theta_component(component: &ThetaComponent, step: usize) -> f64 {
    let drift = (1.0 - 1.0 / component.theta) * component.slope * step as f64;
    component.last_level + drift
}

fn reseasonalize_value(
    value: f64,
    position: usize,
    seasonality: Option<ThetaSeasonality>,
    pattern: Option<&[f64]>,
) -> Result<f64> {
    let Some(seasonality) = seasonality else {
        return Ok(value);
    };
    let pattern = pattern.ok_or_else(|| {
        CartoBoostError::InvalidInput("theta seasonal pattern is missing".to_string())
    })?;
    let seasonal = pattern[position % seasonality.season_length];
    match seasonality.kind {
        ThetaSeasonalityKind::Additive => Ok(value + seasonal),
        ThetaSeasonalityKind::Multiplicative => Ok(value * seasonal),
    }
}

fn validate_horizon(horizon: usize) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    Ok(())
}

fn not_fitted() -> CartoBoostError {
    CartoBoostError::InvalidInput("forecaster must be fitted before predict".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::ForecastFrequency;
    use chrono::{NaiveDate, NaiveDateTime};

    fn ts(day: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 1, day)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("valid fixture timestamp")
    }

    #[test]
    fn theta_forecasts_panel_series_without_bleeding() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 12.0),
                ForecastRow::new("PU1->DO2", ts(3), 15.0),
                ForecastRow::new("PU1->DO2", ts(4), 19.0),
                ForecastRow::new("PU9->DO8", ts(1), 30.0),
                ForecastRow::new("PU9->DO8", ts(2), 29.0),
                ForecastRow::new("PU9->DO8", ts(3), 27.0),
                ForecastRow::new("PU9->DO8", ts(4), 24.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ThetaForecaster::new(2.0, 0.4).expect("valid theta");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.series_id.as_str(), row.horizon, row.mean))
            .collect::<Vec<_>>();
        assert_eq!(means.len(), 4);
        assert_eq!(means[0].0, "PU1->DO2");
        assert_eq!(means[2].0, "PU9->DO8");
        assert!(means[0].2 > means[1].2 - 10.0);
        assert_ne!(means[0].2, means[2].2);
        assert_eq!(model.fitted_values("PU1->DO2").expect("fitted").len(), 4);
        assert_eq!(model.residuals("PU9->DO8").expect("residuals").len(), 4);
    }

    #[test]
    fn theta_additive_seasonality_reseasons_forecast() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| {
                    let base = f64::from(day);
                    let seasonal = if day % 2 == 0 { 5.0 } else { -5.0 };
                    ForecastRow::single(ts(day), 20.0 + base + seasonal)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let seasonality = ThetaSeasonality::additive(2).expect("valid season");
        let mut model =
            ThetaForecaster::with_seasonality(2.0, 0.5, Some(seasonality)).expect("valid theta");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| row.mean)
            .collect::<Vec<_>>();

        assert_eq!(means.len(), 2);
        assert!(means[1] > means[0]);
    }

    #[test]
    fn theta_multiplicative_rejects_non_positive_values() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
                ForecastRow::single(ts(3), 0.0),
                ForecastRow::single(ts(4), 4.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let seasonality = ThetaSeasonality::multiplicative(2).expect("valid season");
        let mut model =
            ThetaForecaster::with_seasonality(2.0, 0.5, Some(seasonality)).expect("valid theta");

        let err = model.fit(&frame).expect_err("non-positive values rejected");

        assert!(err.to_string().contains("non-positive"));
    }

    #[test]
    fn optimized_theta_selects_from_grid_deterministically() {
        let frame = ForecastFrame::new(
            (1..=6)
                .map(|day| ForecastRow::single(ts(day), f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            OptimizedThetaForecaster::new(vec![1.0, 2.0], vec![0.2, 0.8]).expect("valid grid");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(matches!(model.selected_theta(), Some(1.0 | 2.0)));
        assert!(matches!(model.selected_alpha(), Some(0.2 | 0.8)));
        assert_eq!(model.validation_scores().len(), 4);
        assert_eq!(forecast.predictions().len(), 2);
    }

    #[test]
    fn auto_kalman_selects_variances_and_predicts_with_auto_name() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 20.0 + 2.0 * f64::from(day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoKalmanForecaster::with_grids(
            vec![0.001, 0.01],
            vec![0.0001, 0.001],
            vec![0.1, 1.0],
            Some(2),
        )
        .expect("valid auto kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let metadata = model.metadata();

        assert!(model.selected_params().is_some());
        assert_eq!(model.validation_scores().len(), 8);
        assert_eq!(forecast.predictions().len(), 2);
        assert_eq!(forecast.predictions()[0].model, "auto_kalman");
        assert!(
            metadata["validation_scores"]
                .as_array()
                .expect("scores")
                .len()
                == 8
        );
    }

    #[test]
    fn auto_kalman_rejects_empty_grid_and_short_validation_history() {
        assert!(
            AutoKalmanForecaster::with_grids(Vec::new(), vec![0.001], vec![1.0], Some(1),).is_err()
        );

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            AutoKalmanForecaster::with_grids(vec![0.001], vec![0.0001], vec![1.0], Some(1))
                .expect("valid grid");

        let err = model.fit(&frame).expect_err("short history rejected");

        assert!(err.to_string().contains("auto_kalman requires"));
    }

    #[test]
    fn local_level_kalman_forecasts_flat_panel_levels() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PULocationID=1", ts(1), 10.0),
                ForecastRow::new("PULocationID=1", ts(2), 11.0),
                ForecastRow::new("PULocationID=2", ts(1), 30.0),
                ForecastRow::new("PULocationID=2", ts(2), 31.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = LocalLevelKalmanForecaster::new(0.01, 0.1).expect("valid kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert_eq!(forecast.predictions().len(), 4);
        assert!(forecast
            .predictions()
            .iter()
            .all(|prediction| prediction.model == "local_level_kalman"));
    }

    #[test]
    fn auto_local_level_kalman_selects_variances() {
        let frame = ForecastFrame::new(
            (1..=6)
                .map(|day| ForecastRow::single(ts(day), 20.0 + f64::from(day % 2)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model =
            AutoLocalLevelKalmanForecaster::with_grids(vec![0.001, 0.01], vec![0.1, 1.0], Some(2))
                .expect("valid auto kalman");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(model.selected_params().is_some());
        assert_eq!(model.validation_scores().len(), 4);
        assert_eq!(forecast.predictions()[0].model, "auto_local_level_kalman");
    }

    #[test]
    fn ets_forecasts_panel_series_with_daily_timestamps() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PULocationID=1", ts(1), 10.0),
                ForecastRow::new("PULocationID=1", ts(2), 12.0),
                ForecastRow::new("PULocationID=1", ts(3), 14.0),
                ForecastRow::new("PULocationID=1", ts(4), 16.0),
                ForecastRow::new("PULocationID=2", ts(1), 30.0),
                ForecastRow::new("PULocationID=2", ts(2), 29.0),
                ForecastRow::new("PULocationID=2", ts(3), 28.0),
                ForecastRow::new("PULocationID=2", ts(4), 27.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::new(0.6, 0.2).expect("valid ets");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        let predictions = forecast.predictions();
        assert_eq!(predictions.len(), 4);
        assert_eq!(predictions[0].series_id, "PULocationID=1");
        assert_eq!(predictions[0].timestamp, ts(5));
        assert_eq!(predictions[1].horizon, 2);
        assert_eq!(predictions[2].series_id, "PULocationID=2");
        assert!(predictions[0].mean > 16.0);
        assert!(predictions[2].mean < 27.0);
        assert_eq!(
            model.fitted_values("PULocationID=1").expect("fitted").len(),
            4
        );
        assert_eq!(
            model.level_values("PULocationID=1").expect("levels").len(),
            4
        );
        assert_eq!(
            model.trend_values("PULocationID=1").expect("trends").len(),
            4
        );
        assert_eq!(
            model
                .seasonal_values("PULocationID=1")
                .expect("seasonals")
                .len(),
            4
        );
    }

    #[test]
    fn ets_additive_seasonality_repeats_pattern() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| {
                    let seasonal = if day % 2 == 0 { 4.0 } else { -4.0 };
                    ForecastRow::single(ts(day), 50.0 + seasonal)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::with_additive_seasonality(0.5, 0.0, Some(0.5), Some(2))
            .expect("valid seasonal ets");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| row.mean)
            .collect::<Vec<_>>();

        assert_eq!(forecast.predictions()[0].timestamp, ts(9));
        assert!(means[1] > means[0]);
        let seasonals = model
            .seasonal_values("__single__")
            .expect("seasonal contributions");
        assert_eq!(seasonals.len(), 8);
        assert!(seasonals[1] > seasonals[0]);
    }

    #[test]
    fn ets_rejects_invalid_params_and_short_seasonal_history() {
        assert!(ETSForecaster::new(0.0, 0.2).is_err());
        assert!(ETSForecaster::with_additive_seasonality(0.5, 0.2, Some(0.5), None).is_err());

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
                ForecastRow::single(ts(3), 3.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ETSForecaster::with_additive_seasonality(0.5, 0.2, Some(0.5), Some(2))
            .expect("valid seasonal ets");
        let err = model.fit(&frame).expect_err("short seasonal history");
        assert!(err.to_string().contains("two full seasonal cycles"));
    }

    #[test]
    fn arima_forecasts_differenced_linear_series() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day) * 3.0))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(0, 1, 0).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(3).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.timestamp, row.horizon, row.mean))
            .collect::<Vec<_>>();

        assert_eq!(means[0].0, ts(9));
        assert_eq!(means[2].1, 3);
        assert!((means[0].2 - 37.0).abs() < 1.0e-6);
        assert!((means[2].2 - 43.0).abs() < 1.0e-6);
        assert_eq!(model.residuals("__single__").expect("residuals").len(), 7);
    }

    #[test]
    fn arima_forecasts_each_panel_series_without_bleeding() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::new("PU1->DO2", ts(1), 10.0),
                ForecastRow::new("PU1->DO2", ts(2), 13.0),
                ForecastRow::new("PU1->DO2", ts(3), 16.0),
                ForecastRow::new("PU1->DO2", ts(4), 19.0),
                ForecastRow::new("PU9->DO8", ts(1), 40.0),
                ForecastRow::new("PU9->DO8", ts(2), 38.0),
                ForecastRow::new("PU9->DO8", ts(3), 36.0),
                ForecastRow::new("PU9->DO8", ts(4), 34.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(0, 1, 0).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");
        let means = forecast
            .predictions()
            .iter()
            .map(|row| (row.series_id.as_str(), row.timestamp, row.mean))
            .collect::<Vec<_>>();

        assert_eq!(means.len(), 4);
        assert_eq!(means[0].0, "PU1->DO2");
        assert_eq!(means[0].1, ts(5));
        assert_eq!(means[2].0, "PU9->DO8");
        assert!(means[0].2 > 19.0);
        assert!(means[2].2 < 34.0);
    }

    #[test]
    fn arima_rejects_invalid_order_and_insufficient_history() {
        assert!(ArimaForecaster::new(9, 0, 0).is_err());
        assert!(ArimaForecaster::new(1, 0, 9).is_err());

        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(2, 0, 0).expect("valid arima");
        let err = model.fit(&frame).expect_err("insufficient history");
        assert!(err.to_string().contains("requires more than 2"));
    }

    #[test]
    fn arima_supports_moving_average_terms() {
        let frame = ForecastFrame::new(
            (1..=10)
                .map(|day| {
                    let shock = if day % 3 == 0 { 2.0 } else { -1.0 };
                    ForecastRow::single(ts(day), 20.0 + f64::from(day) + shock)
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = ArimaForecaster::new(1, 0, 1).expect("valid arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert_eq!(model.order(), (1, 0, 1));
        assert_eq!(forecast.predictions().len(), 2);
        assert!(forecast
            .predictions()
            .iter()
            .all(|row| row.mean.is_finite()));
    }

    #[test]
    fn arima_candidate_score_excludes_warmup_residuals() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), 20.0 + f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let state = FittedArimaState::from_frame(&frame, 2, 0, 1).expect("fit arima");
        let series = state.series.get("__single__").expect("single series");

        let expected = series
            .residuals
            .iter()
            .skip(2)
            .map(|residual| residual * residual)
            .sum::<f64>()
            / 6.0;

        assert_eq!(series.score_start, 2);
        assert!((state.mean_squared_residual() - expected).abs() < 1.0e-12);
    }

    #[test]
    fn auto_arima_selects_candidate_and_predicts_with_model_name() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), f64::from(day * day)))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("valid frame");
        let mut model = AutoARIMAForecaster::with_max_order(2, 1, 1).expect("valid auto arima");

        model.fit(&frame).expect("fit");
        let forecast = model.predict(2).expect("predict");

        assert!(matches!(
            model.selected_order(),
            Some((0..=2, 0..=1, 0..=1))
        ));
        assert_eq!(model.validation_scores().len(), 12);
        assert_eq!(forecast.predictions().len(), 2);
        assert_eq!(forecast.predictions()[0].model, "auto_arima");
        assert_eq!(forecast.predictions()[0].timestamp, ts(9));
    }
}
