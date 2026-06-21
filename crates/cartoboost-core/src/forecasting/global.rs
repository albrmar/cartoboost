use crate::booster::{Booster, BoosterConfig};
use crate::data::{Dataset, FeatureKind, FeatureSchema};
use crate::forecasting::lag_features::{history_by_series, LagFeatureBuilder, LagFeatureConfig};
use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::tree::Model;
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GlobalForecastTargetMode {
    Level,
    DeltaFromLast,
    SeasonalDelta { season_length: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GlobalForecastSampleWeightMode {
    Uniform,
    ExponentialRecency { half_life: usize },
}

#[derive(Debug, Clone)]
pub struct CartoBoostLagForecaster {
    lag_builder: LagFeatureBuilder,
    booster_config: BoosterConfig,
    target_mode: GlobalForecastTargetMode,
    sample_weight_mode: GlobalForecastSampleWeightMode,
    fitted: Option<FittedGlobalState>,
}

#[derive(Debug, Clone)]
struct FittedGlobalState {
    frame: ForecastFrame,
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
    model: Model,
    training_rows: usize,
}

impl CartoBoostLagForecaster {
    pub fn new(lag_config: LagFeatureConfig, booster_config: BoosterConfig) -> Result<Self> {
        Self::new_with_target_mode(lag_config, booster_config, GlobalForecastTargetMode::Level)
    }

    pub fn new_with_target_mode(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        target_mode: GlobalForecastTargetMode,
    ) -> Result<Self> {
        Self::new_with_target_mode_and_sample_weight(
            lag_config,
            booster_config,
            target_mode,
            GlobalForecastSampleWeightMode::Uniform,
        )
    }

    pub fn new_with_target_mode_and_sample_weight(
        lag_config: LagFeatureConfig,
        booster_config: BoosterConfig,
        target_mode: GlobalForecastTargetMode,
        sample_weight_mode: GlobalForecastSampleWeightMode,
    ) -> Result<Self> {
        validate_target_mode(target_mode)?;
        validate_sample_weight_mode(sample_weight_mode)?;
        Ok(Self {
            lag_builder: LagFeatureBuilder::new(lag_config)?,
            booster_config,
            target_mode,
            sample_weight_mode,
            fitted: None,
        })
    }

    pub fn lag_builder(&self) -> &LagFeatureBuilder {
        &self.lag_builder
    }

    pub fn booster_config(&self) -> &BoosterConfig {
        &self.booster_config
    }

    pub fn target_mode(&self) -> GlobalForecastTargetMode {
        self.target_mode
    }

    pub fn sample_weight_mode(&self) -> GlobalForecastSampleWeightMode {
        self.sample_weight_mode
    }

    pub fn model(&self) -> Option<&Model> {
        self.fitted.as_ref().map(|state| &state.model)
    }

    pub fn training_rows(&self) -> Option<usize> {
        self.fitted.as_ref().map(|state| state.training_rows)
    }
}

impl Forecaster for CartoBoostLagForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let feature_rows = self.lag_builder.transform_frame(frame)?;
        if feature_rows.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "not enough history to build lag training rows".to_string(),
            ));
        }
        let feature_count = self.lag_builder.feature_names().len();
        let x = Dataset::from_rows(
            feature_rows
                .iter()
                .map(|row| row.features.clone())
                .collect::<Vec<_>>(),
        )?
        .with_schema(FeatureSchema {
            names: self.lag_builder.feature_names().to_vec(),
            kinds: vec![FeatureKind::Numeric; feature_count],
        })?;
        let history_by_series = history_by_series(frame.rows());
        let y = feature_rows
            .iter()
            .map(|row| target_for_mode(&self.target_mode, &history_by_series, row))
            .collect::<Result<Vec<_>>>()?;
        let sample_weights = sample_weights_for_feature_rows(
            &feature_rows,
            &history_by_series,
            self.sample_weight_mode,
        )?;
        let model =
            Booster::new(self.booster_config.clone()).fit(&x, &y, sample_weights.as_deref())?;
        self.fitted = Some(FittedGlobalState {
            frame: frame.clone(),
            history_by_series,
            model,
            training_rows: feature_rows.len(),
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, fitted_history)| {
                let mut history = fitted_history.clone();
                let last = history
                    .last()
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput("empty series history".to_string())
                    })?
                    .clone();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let timestamp = fitted.frame.frequency().advance(last.timestamp, step)?;
                    let features = self
                        .lag_builder
                        .transform_next_sorted_prior(series_id, &history, timestamp)?;
                    let raw_prediction = fitted.model.predict_one(&features);
                    let mean = match self.target_mode {
                        GlobalForecastTargetMode::Level => raw_prediction,
                        GlobalForecastTargetMode::DeltaFromLast => {
                            let previous = history.last().ok_or_else(|| {
                                CartoBoostError::InvalidInput(format!(
                                    "series {series_id} has no previous target for delta forecast"
                                ))
                            })?;
                            previous.target + raw_prediction
                        }
                        GlobalForecastTargetMode::SeasonalDelta { season_length } => {
                            let seasonal = seasonal_target_before(&history, season_length)
                                .ok_or_else(|| {
                                    CartoBoostError::InvalidInput(format!(
                                        "series {series_id} does not have enough seasonal history"
                                    ))
                                })?;
                            seasonal + raw_prediction
                        }
                    };
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp,
                        horizon: step,
                        model: self.model_name().to_string(),
                        mean,
                    });
                    let covariates = history
                        .last()
                        .map(|row| row.covariates.clone())
                        .unwrap_or_default();
                    history.push(ForecastRow::with_covariates(
                        series_id.clone(),
                        timestamp,
                        mean,
                        covariates,
                    ));
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "cartoboost_lag"
    }

    fn metadata(&self) -> Value {
        let mut payload = json!({
            "model": self.model_name(),
            "feature_names": self.lag_builder.feature_names(),
            "lag_config": self.lag_builder.config(),
            "booster_config": self.booster_config,
            "target_mode": target_mode_name(self.target_mode),
            "sample_weight_mode": sample_weight_mode_name(self.sample_weight_mode),
        });
        if let Some(fitted) = &self.fitted {
            payload["training_rows"] = json!(fitted.training_rows);
            payload["series_count"] = json!(fitted.history_by_series.len());
            payload["native_model_metadata"] = json!(fitted.model.metadata);
        }
        payload
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

fn validate_target_mode(target_mode: GlobalForecastTargetMode) -> Result<()> {
    if matches!(
        target_mode,
        GlobalForecastTargetMode::SeasonalDelta { season_length: 0 }
    ) {
        return Err(CartoBoostError::InvalidInput(
            "seasonal_delta target mode requires a positive season length".to_string(),
        ));
    }
    Ok(())
}

fn validate_sample_weight_mode(mode: GlobalForecastSampleWeightMode) -> Result<()> {
    match mode {
        GlobalForecastSampleWeightMode::Uniform => Ok(()),
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life } => {
            if half_life == 0 {
                return Err(CartoBoostError::InvalidInput(
                    "exponential recency half_life must be positive".to_string(),
                ));
            }
            Ok(())
        }
    }
}

fn not_fitted() -> CartoBoostError {
    CartoBoostError::InvalidInput("forecaster must be fitted before predict".to_string())
}

fn sample_weights_for_feature_rows(
    feature_rows: &[crate::forecasting::LagFeatureRow],
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    mode: GlobalForecastSampleWeightMode,
) -> Result<Option<Vec<f64>>> {
    match mode {
        GlobalForecastSampleWeightMode::Uniform => Ok(None),
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life } => {
            let mut weights = Vec::with_capacity(feature_rows.len());
            for row in feature_rows {
                let history = history_by_series.get(&row.series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing history for series {}",
                        row.series_id
                    ))
                })?;
                let row_index = history
                    .iter()
                    .position(|history_row| history_row.timestamp == row.timestamp)
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing history timestamp {} for series {}",
                            row.timestamp, row.series_id
                        ))
                    })?;
                let age = history.len().saturating_sub(row_index + 1);
                let exponent = -(age as f64) / half_life as f64;
                weights.push(2.0_f64.powf(exponent));
            }
            normalize_sample_weights(weights).map(Some)
        }
    }
}

fn normalize_sample_weights(mut weights: Vec<f64>) -> Result<Vec<f64>> {
    if weights.is_empty() {
        return Ok(weights);
    }
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "sample weights must have positive finite mean".to_string(),
        ));
    }
    for weight in &mut weights {
        *weight /= mean;
    }
    Ok(weights)
}

fn target_for_mode(
    target_mode: &GlobalForecastTargetMode,
    history_by_series: &BTreeMap<String, Vec<ForecastRow>>,
    row: &crate::forecasting::LagFeatureRow,
) -> Result<f64> {
    match target_mode {
        GlobalForecastTargetMode::Level => Ok(row.target),
        GlobalForecastTargetMode::DeltaFromLast => {
            let history = history_by_series.get(&row.series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing history for series {}",
                    row.series_id
                ))
            })?;
            let prior_target = prior_target_before(history, row.timestamp).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing prior target for series {} at {}",
                    row.series_id, row.timestamp
                ))
            })?;
            Ok(row.target - prior_target)
        }
        GlobalForecastTargetMode::SeasonalDelta { season_length } => {
            let history = history_by_series.get(&row.series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing history for series {}",
                    row.series_id
                ))
            })?;
            let seasonal_target =
                seasonal_target_before_timestamp(history, row.timestamp, *season_length)
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing seasonal target for series {} at {}",
                            row.series_id, row.timestamp
                        ))
                    })?;
            Ok(row.target - seasonal_target)
        }
    }
}

fn prior_target_before(history: &[ForecastRow], timestamp: chrono::NaiveDateTime) -> Option<f64> {
    history
        .iter()
        .rev()
        .find(|row| row.timestamp < timestamp)
        .map(|row| row.target)
}

fn seasonal_target_before_timestamp(
    history: &[ForecastRow],
    timestamp: chrono::NaiveDateTime,
    season_length: usize,
) -> Option<f64> {
    if season_length == 0 {
        return None;
    }
    let prior = history
        .iter()
        .filter(|row| row.timestamp < timestamp)
        .collect::<Vec<_>>();
    if prior.len() < season_length {
        return None;
    }
    Some(prior[prior.len() - season_length].target)
}

fn seasonal_target_before(history: &[ForecastRow], season_length: usize) -> Option<f64> {
    if season_length == 0 || history.len() < season_length {
        return None;
    }
    Some(history[history.len() - season_length].target)
}

pub(crate) fn target_mode_name(target_mode: GlobalForecastTargetMode) -> String {
    match target_mode {
        GlobalForecastTargetMode::Level => "level".to_string(),
        GlobalForecastTargetMode::DeltaFromLast => "delta_from_last".to_string(),
        GlobalForecastTargetMode::SeasonalDelta { season_length } => {
            format!("seasonal_delta_{season_length}")
        }
    }
}

pub(crate) fn sample_weight_mode_name(mode: GlobalForecastSampleWeightMode) -> String {
    match mode {
        GlobalForecastSampleWeightMode::Uniform => "uniform".to_string(),
        GlobalForecastSampleWeightMode::ExponentialRecency { half_life } => {
            format!("exponential_recency_half_life_{half_life}")
        }
    }
}
