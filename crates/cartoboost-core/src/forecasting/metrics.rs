use crate::forecasting::ForecastResult;
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastMetricSet {
    pub mae: f64,
    pub rmse: f64,
    pub wape: f64,
    pub smape: f64,
    pub bias: f64,
    pub mase: Option<f64>,
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
    evaluate_forecast_with_training(forecast, actuals, &[], None)
}

pub fn evaluate_forecast_with_training(
    forecast: &ForecastResult,
    actuals: &[ForecastActual],
    training_actuals: &[ForecastActual],
    mase_seasonality: Option<usize>,
) -> Result<ForecastMetricSet> {
    if actuals.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast metrics require at least one actual value".to_string(),
        ));
    }
    let mut predictions = HashMap::with_capacity(forecast.predictions().len());
    for prediction in forecast.predictions() {
        let previous = predictions.insert(
            (
                prediction.series_id.clone(),
                prediction.timestamp,
                prediction.horizon,
            ),
            prediction.mean,
        );
        if previous.is_some() {
            return Err(CartoBoostError::InvalidInput(format!(
                "duplicate forecast for series {}, timestamp {}, horizon {}",
                prediction.series_id, prediction.timestamp, prediction.horizon
            )));
        }
    }

    let mut abs_sum = 0.0;
    let mut squared_sum = 0.0;
    let mut actual_abs_sum = 0.0;
    let mut smape_sum = 0.0;
    let mut smape_count = 0usize;
    let mut error_sum = 0.0;
    let mut actual_keys = HashSet::with_capacity(actuals.len());

    for actual in actuals {
        if !actual.actual.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "actual forecast values must be finite".to_string(),
            ));
        }
        let key = (actual.series_id.clone(), actual.timestamp, actual.horizon);
        if !actual_keys.insert(key.clone()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "duplicate actual for series {}, timestamp {}, horizon {}",
                actual.series_id, actual.timestamp, actual.horizon
            )));
        }
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
    for key in predictions.keys() {
        if !actual_keys.contains(key) {
            return Err(CartoBoostError::InvalidInput(format!(
                "forecast has no matching actual for series {}, timestamp {}, horizon {}",
                key.0, key.1, key.2
            )));
        }
    }

    let n = actuals.len() as f64;
    let smape = if smape_count == 0 {
        0.0
    } else {
        smape_sum / smape_count as f64
    };
    let mase = if let Some(seasonality) = mase_seasonality {
        Some(evaluate_mase(abs_sum / n, training_actuals, seasonality)?)
    } else {
        None
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
        mase,
    })
}

fn evaluate_mase(mae: f64, training_actuals: &[ForecastActual], seasonality: usize) -> Result<f64> {
    if seasonality == 0 {
        return Err(CartoBoostError::InvalidInput(
            "MASE seasonality must be positive".to_string(),
        ));
    }
    if training_actuals.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "MASE requires training actuals".to_string(),
        ));
    }
    let mut by_series: HashMap<&str, Vec<&ForecastActual>> = HashMap::new();
    for actual in training_actuals {
        if !actual.actual.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "training actual values must be finite".to_string(),
            ));
        }
        by_series
            .entry(actual.series_id.as_str())
            .or_default()
            .push(actual);
    }
    let mut abs_naive_sum = 0.0;
    let mut count = 0usize;
    for values in by_series.values_mut() {
        values.sort_by_key(|actual| actual.timestamp);
        if values.len() <= seasonality {
            continue;
        }
        for index in seasonality..values.len() {
            abs_naive_sum += (values[index].actual - values[index - seasonality].actual).abs();
            count += 1;
        }
    }
    if count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "MASE requires more training rows than the seasonality".to_string(),
        ));
    }
    let scale = abs_naive_sum / count as f64;
    if scale == 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "MASE scale must be non-zero".to_string(),
        ));
    }
    Ok(mae / scale)
}
