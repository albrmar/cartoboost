use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

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
    #[serde(default)]
    intervals: Vec<ForecastIntervalPrediction>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastIntervalPrediction {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub horizon: usize,
    pub model: String,
    pub level: f64,
    pub lower: f64,
    pub upper: f64,
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
        Ok(Self {
            predictions,
            intervals: Vec::new(),
        })
    }

    pub fn new_with_intervals(
        predictions: Vec<ForecastPrediction>,
        mut intervals: Vec<ForecastIntervalPrediction>,
    ) -> Result<Self> {
        let result = Self::new(predictions)?;
        for interval in &intervals {
            validate_interval_prediction(interval)?;
        }
        intervals.sort_by(|a, b| {
            a.series_id
                .cmp(&b.series_id)
                .then_with(|| a.timestamp.cmp(&b.timestamp))
                .then_with(|| a.horizon.cmp(&b.horizon))
                .then_with(|| a.model.cmp(&b.model))
                .then_with(|| {
                    a.level
                        .partial_cmp(&b.level)
                        .expect("interval levels are finite")
                })
        });
        Ok(Self {
            predictions: result.predictions,
            intervals,
        })
    }

    pub fn predictions(&self) -> &[ForecastPrediction] {
        &self.predictions
    }

    pub fn intervals(&self) -> &[ForecastIntervalPrediction] {
        &self.intervals
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
        Self::new_with_intervals(result.predictions, result.intervals)
    }

    pub fn prediction_columns() -> Vec<&'static str> {
        vec!["series_id", "timestamp", "horizon", "model", "prediction"]
    }

    pub fn result_columns(&self) -> Vec<String> {
        let mut columns = vec![
            "series_id".to_string(),
            "timestamp".to_string(),
            "horizon".to_string(),
            "model".to_string(),
            "prediction".to_string(),
        ];
        for level in self.interval_levels() {
            let suffix = interval_suffix(level);
            columns.push(format!("prediction_lower_{suffix}"));
            columns.push(format!("prediction_upper_{suffix}"));
        }
        columns
    }

    pub fn to_json_value(&self) -> Value {
        let intervals_by_key = self.intervals_by_prediction_key();
        let records = self
            .predictions
            .iter()
            .map(|prediction| {
                let mut record = json!({
                    "series_id": prediction.series_id,
                    "timestamp": prediction.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    "horizon": prediction.horizon,
                    "model": prediction.model,
                    "prediction": prediction.mean,
                });
                if let Some(intervals) = intervals_by_key.get(&prediction_key(prediction)) {
                    for interval in intervals {
                        let suffix = interval_suffix(interval.level);
                        record[format!("prediction_lower_{suffix}")] = json!(interval.lower);
                        record[format!("prediction_upper_{suffix}")] = json!(interval.upper);
                    }
                }
                record
            })
            .collect::<Vec<_>>();
        json!({
            "columns": self.result_columns(),
            "records": records,
        })
    }

    fn interval_levels(&self) -> Vec<f64> {
        let mut levels = self
            .intervals
            .iter()
            .map(|interval| interval.level)
            .collect::<Vec<_>>();
        levels.sort_by(|a, b| a.partial_cmp(b).expect("interval levels are finite"));
        levels.dedup_by(|left, right| (*left - *right).abs() < 1.0e-12);
        levels
    }

    fn intervals_by_prediction_key(&self) -> BTreeMap<String, Vec<&ForecastIntervalPrediction>> {
        let mut by_key: BTreeMap<String, Vec<&ForecastIntervalPrediction>> = BTreeMap::new();
        for interval in &self.intervals {
            by_key
                .entry(interval_prediction_key(interval))
                .or_default()
                .push(interval);
        }
        by_key
    }

    fn from_record_values(records: &[Value]) -> Result<Self> {
        let mut predictions = Vec::with_capacity(records.len());
        let mut intervals = Vec::new();
        for record in records {
            let series_id = required_string(record, "series_id")?.to_string();
            let timestamp = crate::forecasting::parse_forecast_timestamp(required_string(
                record,
                "timestamp",
            )?)?;
            let horizon = required_usize(record, "horizon")?;
            let model = required_string(record, "model")?.to_string();
            let mean = required_f64(record, "prediction")?;
            intervals.extend(intervals_from_record(
                record, &series_id, timestamp, horizon, &model,
            )?);
            predictions.push(ForecastPrediction {
                series_id,
                timestamp,
                horizon,
                model,
                mean,
            });
        }
        Self::new_with_intervals(predictions, intervals)
    }
}

fn validate_interval_prediction(interval: &ForecastIntervalPrediction) -> Result<()> {
    if interval.horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast interval horizon values must be positive".to_string(),
        ));
    }
    if !interval.level.is_finite() || interval.level <= 0.0 || interval.level >= 1.0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast interval levels must be finite and in (0, 1)".to_string(),
        ));
    }
    if !interval.lower.is_finite() || !interval.upper.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "forecast interval bounds must be finite".to_string(),
        ));
    }
    if interval.lower > interval.upper {
        return Err(CartoBoostError::InvalidInput(
            "forecast interval lower bounds must not exceed upper bounds".to_string(),
        ));
    }
    Ok(())
}

fn prediction_key(prediction: &ForecastPrediction) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}",
        prediction.series_id,
        prediction.timestamp.format("%Y-%m-%dT%H:%M:%S"),
        prediction.horizon,
        prediction.model
    )
}

fn interval_prediction_key(interval: &ForecastIntervalPrediction) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}",
        interval.series_id,
        interval.timestamp.format("%Y-%m-%dT%H:%M:%S"),
        interval.horizon,
        interval.model
    )
}

fn interval_suffix(level: f64) -> String {
    let rounded = (level * 100.0).round();
    if (level * 100.0 - rounded).abs() < 1.0e-9 {
        format!("p{:02}", rounded as usize)
    } else {
        format!("{level:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .replace('.', "_")
    }
}

fn intervals_from_record(
    record: &Value,
    series_id: &str,
    timestamp: NaiveDateTime,
    horizon: usize,
    model: &str,
) -> Result<Vec<ForecastIntervalPrediction>> {
    let object = record.as_object().ok_or_else(|| {
        CartoBoostError::InvalidInput("forecast record JSON must be an object".to_string())
    })?;
    let mut levels = BTreeMap::new();
    for key in object.keys() {
        if let Some(suffix) = key.strip_prefix("prediction_lower_") {
            levels.insert(suffix.to_string(), ());
        }
        if let Some(suffix) = key.strip_prefix("prediction_upper_") {
            levels.insert(suffix.to_string(), ());
        }
    }

    let mut intervals = Vec::with_capacity(levels.len());
    for suffix in levels.keys() {
        let lower_key = format!("prediction_lower_{suffix}");
        let upper_key = format!("prediction_upper_{suffix}");
        if record.get(&lower_key).is_none() || record.get(&upper_key).is_none() {
            return Err(CartoBoostError::InvalidInput(format!(
                "forecast JSON interval {suffix:?} must include both lower and upper bounds"
            )));
        }
        intervals.push(ForecastIntervalPrediction {
            series_id: series_id.to_string(),
            timestamp,
            horizon,
            model: model.to_string(),
            level: interval_level_from_suffix(suffix)?,
            lower: required_f64(record, &lower_key)?,
            upper: required_f64(record, &upper_key)?,
        });
    }
    Ok(intervals)
}

fn interval_level_from_suffix(suffix: &str) -> Result<f64> {
    let level = if let Some(percent) = suffix.strip_prefix('p') {
        percent.parse::<f64>().map(|value| value / 100.0)
    } else {
        suffix.replace('_', ".").parse::<f64>()
    }
    .map_err(|_| {
        CartoBoostError::InvalidInput(format!(
            "forecast JSON interval suffix {suffix:?} is invalid"
        ))
    })?;
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "forecast JSON interval suffix {suffix:?} must encode a level in (0, 1)"
        )));
    }
    Ok(level)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(day: u32) -> NaiveDateTime {
        chrono::NaiveDate::from_ymd_opt(2026, 1, day)
            .expect("valid date")
            .and_hms_opt(0, 0, 0)
            .expect("valid timestamp")
    }

    #[test]
    fn forecast_result_record_json_round_trips_intervals() {
        let result = ForecastResult::new_with_intervals(
            vec![
                ForecastPrediction {
                    series_id: "pickup_zone_1".to_string(),
                    timestamp: ts(1),
                    horizon: 1,
                    model: "piecewise_linear_seasonal".to_string(),
                    mean: 42.0,
                },
                ForecastPrediction {
                    series_id: "pickup_zone_1".to_string(),
                    timestamp: ts(2),
                    horizon: 2,
                    model: "piecewise_linear_seasonal".to_string(),
                    mean: 44.0,
                },
            ],
            vec![
                ForecastIntervalPrediction {
                    series_id: "pickup_zone_1".to_string(),
                    timestamp: ts(1),
                    horizon: 1,
                    model: "piecewise_linear_seasonal".to_string(),
                    level: 0.8,
                    lower: 38.0,
                    upper: 46.0,
                },
                ForecastIntervalPrediction {
                    series_id: "pickup_zone_1".to_string(),
                    timestamp: ts(2),
                    horizon: 2,
                    model: "piecewise_linear_seasonal".to_string(),
                    level: 0.8,
                    lower: 39.0,
                    upper: 49.0,
                },
            ],
        )
        .expect("valid result");

        let restored = ForecastResult::from_json_string(&result.to_json_string().expect("json"))
            .expect("record JSON round-trip");

        assert_eq!(restored.intervals(), result.intervals());
        assert_eq!(restored.to_json_value(), result.to_json_value());
    }
}
