use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
        serde_json::to_string_pretty(&self.to_json_value()).map_err(CartoBoostError::from)
    }

    pub fn from_json_string(value: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(value)?;
        if let Some(records) = value.get("records").and_then(Value::as_array) {
            return Self::from_record_values(records);
        }
        let result: Self = serde_json::from_value(value)?;
        Self::new(result.predictions)
    }

    pub fn prediction_columns() -> Vec<&'static str> {
        vec!["series_id", "timestamp", "horizon", "model", "prediction"]
    }

    pub fn to_json_value(&self) -> Value {
        let records = self
            .predictions
            .iter()
            .map(|prediction| {
                json!({
                    "series_id": prediction.series_id,
                    "timestamp": prediction.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    "horizon": prediction.horizon,
                    "model": prediction.model,
                    "prediction": prediction.mean,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "columns": Self::prediction_columns(),
            "records": records,
        })
    }

    fn from_record_values(records: &[Value]) -> Result<Self> {
        let mut predictions = Vec::with_capacity(records.len());
        for record in records {
            let series_id = required_string(record, "series_id")?.to_string();
            let timestamp = crate::forecasting::parse_forecast_timestamp(required_string(
                record,
                "timestamp",
            )?)?;
            let horizon = required_usize(record, "horizon")?;
            let model = required_string(record, "model")?.to_string();
            let mean = required_f64(record, "prediction")?;
            predictions.push(ForecastPrediction {
                series_id,
                timestamp,
                horizon,
                model,
                mean,
            });
        }
        Self::new(predictions)
    }
}

fn required_string<'a>(record: &'a Value, field: &str) -> Result<&'a str> {
    record
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| CartoBoostError::InvalidInput(format!("forecast JSON missing {field:?}")))
}

fn required_usize(record: &Value, field: &str) -> Result<usize> {
    let value = record
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| CartoBoostError::InvalidInput(format!("forecast JSON missing {field:?}")))?;
    usize::try_from(value)
        .map_err(|_| CartoBoostError::InvalidInput(format!("forecast JSON {field:?} is too large")))
}

fn required_f64(record: &Value, field: &str) -> Result<f64> {
    let value = record
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| CartoBoostError::InvalidInput(format!("forecast JSON missing {field:?}")))?;
    if !value.is_finite() {
        return Err(CartoBoostError::InvalidInput(format!(
            "forecast JSON {field:?} must be finite"
        )));
    }
    Ok(value)
}
