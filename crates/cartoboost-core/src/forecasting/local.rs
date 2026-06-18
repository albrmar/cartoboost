#![allow(dead_code)]

use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

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

impl Forecaster for NaiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = Some(FittedLocalState::from_frame(frame));
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mut predictions = Vec::new();
        for (series_id, history) in &fitted.history_by_series {
            let last = history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?;
            for step in 1..=horizon {
                predictions.push(ForecastPrediction {
                    series_id: series_id.clone(),
                    timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean: last.target,
                });
            }
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
        let mut predictions = Vec::new();
        for (series_id, history) in &fitted.history_by_series {
            let last = history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?;
            let base = history.len() - self.season_length;
            for step in 1..=horizon {
                let seasonal_index = base + ((step - 1) % self.season_length);
                predictions.push(ForecastPrediction {
                    series_id: series_id.clone(),
                    timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean: history[seasonal_index].target,
                });
            }
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
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mut predictions = Vec::new();
        for (series_id, series) in &fitted.series {
            for step in 1..=horizon {
                let adjusted = forecast_theta_component(&series.component, step);
                let mean = reseasonalize_value(
                    adjusted,
                    series.n_obs + step - 1,
                    self.seasonality,
                    series.seasonal_pattern.as_deref(),
                )?;
                predictions.push(ForecastPrediction {
                    series_id: series_id.clone(),
                    timestamp: fitted
                        .frame
                        .frequency()
                        .advance(series.last_timestamp, step)?,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean,
                });
            }
        }
        ForecastResult::new(predictions)
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
        let mut best: Option<(OrderedF64, OrderedF64, OrderedF64)> = None;
        let mut scores = Vec::new();
        for &theta in &self.theta_grid {
            for &alpha in &self.alpha_grid {
                let fitted = FittedThetaState::from_frame(frame, theta, alpha, self.seasonality)?;
                let mse = fitted.mean_squared_residual();
                scores.push(ThetaValidationScore { theta, alpha, mse });
                let candidate = (OrderedF64(mse), OrderedF64(theta), OrderedF64(alpha));
                if match best {
                    Some(current) => candidate < current,
                    None => true,
                } {
                    best = Some(candidate);
                }
            }
        }
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

impl ThetaForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mut predictions = Vec::new();
        for (series_id, series) in &fitted.series {
            for step in 1..=horizon {
                let adjusted = forecast_theta_component(&series.component, step);
                let mean = reseasonalize_value(
                    adjusted,
                    series.n_obs + step - 1,
                    self.seasonality,
                    series.seasonal_pattern.as_deref(),
                )?;
                predictions.push(ForecastPrediction {
                    series_id: series_id.clone(),
                    timestamp: fitted
                        .frame
                        .frequency()
                        .advance(series.last_timestamp, step)?,
                    horizon: step,
                    model: model_name.to_string(),
                    mean,
                });
            }
        }
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

impl FittedThetaState {
    fn from_frame(
        frame: &ForecastFrame,
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let mut series = BTreeMap::new();
        for (series_id, history) in &local.history_by_series {
            series.insert(
                series_id.clone(),
                FittedThetaSeries::fit(series_id, history, theta, alpha, seasonality)?,
            );
        }
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for series in self.series.values() {
            for residual in series.residuals.iter().skip(1) {
                sum += residual * residual;
                count += 1;
            }
        }
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
}
