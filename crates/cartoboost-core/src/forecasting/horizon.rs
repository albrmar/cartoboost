use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastStrategy {
    Recursive,
    Direct,
    RectifiedRecursive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastRequest {
    pub horizon: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub series_ids: Option<Vec<String>>,
}

impl ForecastRequest {
    pub fn new(horizon: usize) -> Result<Self> {
        Self::with_series_ids(horizon, None)
    }

    pub fn with_series_ids(horizon: usize, series_ids: Option<Vec<String>>) -> Result<Self> {
        validate_horizon(horizon)?;
        if let Some(ids) = &series_ids {
            if ids.is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "forecast request series_ids must not be empty".to_string(),
                ));
            }
            if ids.iter().any(|series_id| series_id.is_empty()) {
                return Err(CartoBoostError::InvalidInput(
                    "forecast request series_ids must not contain empty ids".to_string(),
                ));
            }
        }
        Ok(Self {
            horizon,
            series_ids,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastOutput {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub horizon: usize,
    pub prediction: f64,
}

impl ForecastOutput {
    pub fn new(
        series_id: impl Into<String>,
        timestamp: NaiveDateTime,
        horizon: usize,
        prediction: f64,
    ) -> Result<Self> {
        validate_horizon(horizon)?;
        if !prediction.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "forecast output prediction must be finite".to_string(),
            ));
        }
        let series_id = series_id.into();
        if series_id.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "forecast output series_id must not be empty".to_string(),
            ));
        }
        Ok(Self {
            series_id,
            timestamp,
            horizon,
            prediction,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantileForecastOutput {
    pub point: ForecastOutput,
    pub quantiles: BTreeMap<String, f64>,
}

impl QuantileForecastOutput {
    pub fn new(point: ForecastOutput, quantiles: BTreeMap<String, f64>) -> Result<Self> {
        if quantiles.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "quantile forecast output requires at least one quantile".to_string(),
            ));
        }
        for (label, value) in &quantiles {
            if label.is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "quantile forecast labels must not be empty".to_string(),
                ));
            }
            if !value.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "quantile forecast {label:?} must be finite"
                )));
            }
        }
        Ok(Self { point, quantiles })
    }
}

pub(crate) fn validate_horizon(horizon: usize) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    Ok(())
}
