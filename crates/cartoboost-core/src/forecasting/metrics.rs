use crate::forecasting::ForecastResult;
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastMetricSet {
    pub mae: f64,
    pub rmse: f64,
    pub wape: f64,
    pub smape: f64,
    pub bias: f64,
}

#[derive(Debug, Clone)]
pub struct ForecastActual {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub horizon: usize,
    pub actual: f64,
}

pub fn evaluate_forecast(
    forecast: &ForecastResult,
    actuals: &[ForecastActual],
) -> Result<ForecastMetricSet> {
    if actuals.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast metrics require at least one actual value".to_string(),
        ));
    }
    let mut predictions = HashMap::with_capacity(forecast.predictions().len());
    for prediction in forecast.predictions() {
        predictions.insert(
            (
                prediction.series_id.as_str(),
                prediction.timestamp,
                prediction.horizon,
            ),
            prediction.mean,
        );
    }

    let mut abs_sum = 0.0;
    let mut squared_sum = 0.0;
    let mut actual_abs_sum = 0.0;
    let mut smape_sum = 0.0;
    let mut smape_count = 0usize;
    let mut error_sum = 0.0;

    for actual in actuals {
        if !actual.actual.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "actual forecast values must be finite".to_string(),
            ));
        }
        let key = (actual.series_id.as_str(), actual.timestamp, actual.horizon);
        let Some(predicted) = predictions.get(&key) else {
            return Err(CartoBoostError::InvalidInput(format!(
                "missing forecast for series {}, timestamp {}, horizon {}",
                actual.series_id, actual.timestamp, actual.horizon
            )));
        };
        let error = predicted - actual.actual;
        let abs_error = error.abs();
        abs_sum += abs_error;
        squared_sum += error * error;
        actual_abs_sum += actual.actual.abs();
        error_sum += error;
        let denominator = predicted.abs() + actual.actual.abs();
        if denominator > 0.0 {
            smape_sum += 2.0 * abs_error / denominator;
            smape_count += 1;
        }
    }

    let n = actuals.len() as f64;
    let smape = if smape_count == 0 {
        0.0
    } else {
        smape_sum / smape_count as f64
    };
    Ok(ForecastMetricSet {
        mae: abs_sum / n,
        rmse: (squared_sum / n).sqrt(),
        wape: if actual_abs_sum > 0.0 {
            abs_sum / actual_abs_sum
        } else {
            0.0
        },
        smape,
        bias: error_sum / n,
    })
}
