use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastPrediction {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub horizon: usize,
    pub model: String,
    pub mean: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastResult {
    predictions: Vec<ForecastPrediction>,
}

impl ForecastResult {
    pub fn new(mut predictions: Vec<ForecastPrediction>) -> Result<Self> {
        if predictions.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "forecast result must contain at least one prediction".to_string(),
            ));
        }
        for prediction in &predictions {
            if prediction.horizon == 0 {
                return Err(CartoBoostError::InvalidInput(
                    "forecast horizon values must be positive".to_string(),
                ));
            }
            if !prediction.mean.is_finite() {
                return Err(CartoBoostError::InvalidInput(
                    "forecast means must be finite".to_string(),
                ));
            }
        }
        predictions.sort_by(|a, b| {
            a.series_id
                .cmp(&b.series_id)
                .then_with(|| a.timestamp.cmp(&b.timestamp))
                .then_with(|| a.horizon.cmp(&b.horizon))
                .then_with(|| a.model.cmp(&b.model))
        });
        Ok(Self { predictions })
    }

    pub fn predictions(&self) -> &[ForecastPrediction] {
        &self.predictions
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(CartoBoostError::from)
    }

    pub fn from_json_string(value: &str) -> Result<Self> {
        let result: Self = serde_json::from_str(value)?;
        Self::new(result.predictions)
    }
}
