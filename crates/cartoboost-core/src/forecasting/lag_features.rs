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
    #[serde(default)]
    pub difference_lags: Vec<usize>,
    #[serde(default)]
    pub rolling_trend_windows: Vec<usize>,
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
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
        }
    }
}

impl LagFeatureBuilder {
    pub fn new(config: LagFeatureConfig) -> Result<Self> {
        validate_config(&config)?;
        let mut feature_names = Vec::with_capacity(
            config.lags.len()
                + config.rolling_mean_windows.len()
                + config.difference_lags.len()
                + config.rolling_trend_windows.len()
                + config.calendar_features.len(),
        );
        feature_names.extend(config.lags.iter().map(|lag| format!("target_lag_{lag}")));
        feature_names.extend(
            config
                .rolling_mean_windows
                .iter()
                .map(|window| format!("target_roll_mean_{window}")),
        );
        feature_names.extend(
            config
                .difference_lags
                .iter()
                .map(|lag| format!("target_delta_lag_{lag}")),
        );
        feature_names.extend(
            config
                .rolling_trend_windows
                .iter()
                .map(|window| format!("target_roll_trend_{window}")),
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
        for lag in &self.config.difference_lags {
            if prior.len() <= *lag {
                return Ok(None);
            }
            let last = prior
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty prior history".to_string()))?;
            let row = &prior[prior.len() - 1 - *lag];
            let delta = last.target - row.target;
            if !delta.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "lag delta for series {series_id} is not finite"
                )));
            }
            features.push(delta);
        }
        for window in &self.config.rolling_trend_windows {
            if *window < 2 || prior.len() < *window {
                return Ok(None);
            }
            let start = prior.len() - *window;
            let first = prior[start].target;
            let last = prior
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty prior history".to_string()))?
                .target;
            let trend = (last - first) / (*window - 1) as f64;
            if !trend.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "rolling trend for series {series_id} is not finite"
                )));
            }
            features.push(trend);
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
        && config.difference_lags.is_empty()
        && config.rolling_trend_windows.is_empty()
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
    if config.difference_lags.contains(&0) {
        return Err(CartoBoostError::InvalidInput(
            "difference lags must be positive".to_string(),
        ));
    }
    if config
        .rolling_trend_windows
        .iter()
        .any(|window| *window < 2)
    {
        return Err(CartoBoostError::InvalidInput(
            "rolling trend windows must be at least 2".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::parse_forecast_timestamp;

    fn ts(day: u32) -> NaiveDateTime {
        parse_forecast_timestamp(&format!("2026-01-{day:02}")).expect("timestamp")
    }

    #[test]
    fn delta_and_trend_features_use_only_prior_history() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 10.0),
                ForecastRow::single(ts(2), 12.0),
                ForecastRow::single(ts(3), 15.0),
                ForecastRow::single(ts(4), 19.0),
            ],
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let builder = LagFeatureBuilder::new(LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: vec![2],
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![3],
        })
        .expect("builder");

        let rows = builder.transform_frame(&frame).expect("features");
        let last = rows.last().expect("last row");

        assert_eq!(
            builder.feature_names(),
            &[
                "target_lag_1".to_string(),
                "target_roll_mean_2".to_string(),
                "target_delta_lag_2".to_string(),
                "target_roll_trend_3".to_string(),
            ]
        );
        assert_eq!(last.timestamp, ts(4));
        assert_eq!(last.features, vec![15.0, 13.5, 5.0, 2.5]);
    }

    #[test]
    fn trend_feature_windows_are_validated() {
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![0],
            rolling_trend_windows: Vec::new(),
        })
        .is_err());
        assert!(LagFeatureBuilder::new(LagFeatureConfig {
            lags: Vec::new(),
            rolling_mean_windows: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: vec![1],
        })
        .is_err());
    }
}
