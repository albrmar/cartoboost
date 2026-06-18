use crate::forecasting::{ForecastFrame, ForecastRow};
use crate::{CartoBoostError, Result};
use chrono::{Datelike, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalendarFeature {
    DayOfWeek,
    Month,
    Day,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LagFeatureConfig {
    pub lags: Vec<usize>,
    pub rolling_mean_windows: Vec<usize>,
    pub calendar_features: Vec<CalendarFeature>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LagFeatureRow {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub target: f64,
    pub features: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LagFeatureBuilder {
    config: LagFeatureConfig,
    feature_names: Vec<String>,
}

impl Default for LagFeatureConfig {
    fn default() -> Self {
        Self {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            calendar_features: Vec::new(),
        }
    }
}

impl LagFeatureBuilder {
    pub fn new(config: LagFeatureConfig) -> Result<Self> {
        validate_config(&config)?;
        let mut feature_names = Vec::with_capacity(
            config.lags.len() + config.rolling_mean_windows.len() + config.calendar_features.len(),
        );
        feature_names.extend(config.lags.iter().map(|lag| format!("target_lag_{lag}")));
        feature_names.extend(
            config
                .rolling_mean_windows
                .iter()
                .map(|window| format!("target_roll_mean_{window}")),
        );
        feature_names.extend(config.calendar_features.iter().map(calendar_feature_name));
        Ok(Self {
            config,
            feature_names,
        })
    }

    pub fn config(&self) -> &LagFeatureConfig {
        &self.config
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn transform_frame(&self, frame: &ForecastFrame) -> Result<Vec<LagFeatureRow>> {
        let mut rows = Vec::new();
        for (series_id, history) in history_by_series(frame.rows()) {
            for row_idx in 0..history.len() {
                if let Some(features) = self.features_for_position(&history, row_idx)? {
                    rows.push(LagFeatureRow {
                        series_id: series_id.clone(),
                        timestamp: history[row_idx].timestamp,
                        target: history[row_idx].target,
                        features,
                    });
                }
            }
        }
        Ok(rows)
    }

    pub fn transform_next(
        &self,
        series_id: &str,
        history: &[ForecastRow],
        timestamp: NaiveDateTime,
    ) -> Result<Vec<f64>> {
        if history.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} has no history for lag features"
            )));
        }
        if history.iter().any(|row| row.series_id != series_id) {
            return Err(CartoBoostError::InvalidInput(format!(
                "history for series {series_id} contains another series"
            )));
        }
        let mut sorted = history.to_vec();
        sorted.sort_by_key(|row| row.timestamp);
        if sorted
            .windows(2)
            .any(|pair| pair[0].timestamp >= pair[1].timestamp)
        {
            return Err(CartoBoostError::InvalidInput(format!(
                "history for series {series_id} contains duplicate timestamps"
            )));
        }
        let prior = sorted
            .into_iter()
            .filter(|row| row.timestamp < timestamp)
            .collect::<Vec<_>>();
        self.features_from_prior(series_id, &prior, timestamp)?
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "series {series_id} does not have enough prior history for lag features"
                ))
            })
    }

    fn features_for_position(
        &self,
        history: &[ForecastRow],
        row_idx: usize,
    ) -> Result<Option<Vec<f64>>> {
        let series_id = history
            .get(row_idx)
            .map(|row| row.series_id.as_str())
            .unwrap_or("<unknown>");
        self.features_from_prior(series_id, &history[..row_idx], history[row_idx].timestamp)
    }

    fn features_from_prior(
        &self,
        series_id: &str,
        prior: &[ForecastRow],
        timestamp: NaiveDateTime,
    ) -> Result<Option<Vec<f64>>> {
        let mut features = Vec::with_capacity(self.feature_names.len());
        for lag in &self.config.lags {
            if prior.len() < *lag {
                return Ok(None);
            }
            let row = &prior[prior.len() - *lag];
            features.push(row.target);
        }
        for window in &self.config.rolling_mean_windows {
            if prior.len() < *window {
                return Ok(None);
            }
            let start = prior.len() - *window;
            let sum = prior[start..].iter().map(|row| row.target).sum::<f64>();
            let mean = sum / *window as f64;
            if !mean.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling mean for series {series_id} is not finite"
                )));
            }
            features.push(mean);
        }
        features.extend(
            self.config
                .calendar_features
                .iter()
                .map(|feature| calendar_feature_value(feature, timestamp)),
        );
        Ok(Some(features))
    }
}

pub(crate) fn history_by_series(rows: &[ForecastRow]) -> BTreeMap<String, Vec<ForecastRow>> {
    let mut history_by_series: BTreeMap<String, Vec<ForecastRow>> = BTreeMap::new();
    for row in rows {
        history_by_series
            .entry(row.series_id.clone())
            .or_default()
            .push(row.clone());
    }
    history_by_series
}

fn validate_config(config: &LagFeatureConfig) -> Result<()> {
    if config.lags.is_empty()
        && config.rolling_mean_windows.is_empty()
        && config.calendar_features.is_empty()
    {
        return Err(CartoBoostError::InvalidInput(
            "lag feature config must contain at least one feature".to_string(),
        ));
    }
    if config.lags.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "target lags must be positive".to_string(),
        ));
    }
    if config.rolling_mean_windows.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "rolling mean windows must be positive".to_string(),
        ));
    }
    Ok(())
}

fn calendar_feature_name(feature: &CalendarFeature) -> String {
    match feature {
        CalendarFeature::DayOfWeek => "calendar_day_of_week".to_string(),
        CalendarFeature::Month => "calendar_month".to_string(),
        CalendarFeature::Day => "calendar_day".to_string(),
    }
}

fn calendar_feature_value(feature: &CalendarFeature, timestamp: NaiveDateTime) -> f64 {
    match feature {
        CalendarFeature::DayOfWeek => f64::from(timestamp.weekday().num_days_from_monday()),
        CalendarFeature::Month => f64::from(timestamp.month()),
        CalendarFeature::Day => f64::from(timestamp.day()),
    }
}
