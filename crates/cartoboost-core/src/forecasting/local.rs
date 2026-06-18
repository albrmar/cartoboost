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
struct FittedLocalState {
    frame: ForecastFrame,
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
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
