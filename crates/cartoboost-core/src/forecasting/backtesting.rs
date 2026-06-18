use crate::forecasting::{
    evaluate_forecast_with_training, ForecastActual, ForecastFold, ForecastFrame,
    ForecastMetricSet, ForecastPrediction, Forecaster, RollingOriginSplitter,
};
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BacktestFoldResult {
    pub fold: ForecastFold,
    pub metrics: ForecastMetricSet,
    pub predictions: Vec<ForecastPrediction>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BacktestResult {
    pub folds: Vec<BacktestFoldResult>,
    pub metrics: Option<ForecastMetricSet>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollingOriginBacktester {
    pub splitter: RollingOriginSplitter,
    pub mase_seasonality: Option<usize>,
}

impl RollingOriginBacktester {
    pub fn new(splitter: RollingOriginSplitter) -> Self {
        Self {
            splitter,
            mase_seasonality: None,
        }
    }

    pub fn with_mase_seasonality(mut self, seasonality: usize) -> Result<Self> {
        if seasonality == 0 {
            return Err(CartoBoostError::InvalidInput(
                "MASE seasonality must be positive".to_string(),
            ));
        }
        self.mase_seasonality = Some(seasonality);
        Ok(self)
    }

    pub fn run<M>(&self, model: M, frame: &ForecastFrame) -> Result<BacktestResult>
    where
        M: Forecaster + Clone,
    {
        let mut folds = Vec::new();
        for fold in self.splitter.split(frame)? {
            let train =
                crate::forecasting::splitters::frame_from_indices(frame, &fold.train_indices)?;
            let validation_actuals = actuals_for_indices(frame, &fold.validation_indices);
            let training_actuals = training_actuals_for_indices(frame, &fold.train_indices);
            let mut candidate = model.clone();
            candidate.fit(&train)?;
            let forecast = candidate.predict(fold.horizon)?;
            if forecast.predictions().len() != validation_actuals.len() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "fold {} produced {} predictions for {} validation rows",
                    fold.fold_id,
                    forecast.predictions().len(),
                    validation_actuals.len()
                )));
            }
            let metrics = evaluate_forecast_with_training(
                &forecast,
                &validation_actuals,
                &training_actuals,
                self.mase_seasonality,
            )?;
            folds.push(BacktestFoldResult {
                fold,
                metrics,
                predictions: forecast.predictions().to_vec(),
            });
        }
        let metrics = aggregate_metrics(&folds);
        Ok(BacktestResult { folds, metrics })
    }
}

fn actuals_for_indices(frame: &ForecastFrame, indices: &[usize]) -> Vec<ForecastActual> {
    let horizons = indices
        .iter()
        .map(|index| frame.rows()[*index].timestamp)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(index, timestamp)| (timestamp, index + 1))
        .collect::<BTreeMap<_, _>>();
    indices
        .iter()
        .map(|index| {
            let row = &frame.rows()[*index];
            ForecastActual {
                series_id: row.series_id.clone(),
                timestamp: row.timestamp,
                horizon: horizons[&row.timestamp],
                actual: row.target,
            }
        })
        .collect()
}

fn training_actuals_for_indices(frame: &ForecastFrame, indices: &[usize]) -> Vec<ForecastActual> {
    indices
        .iter()
        .map(|index| {
            let row = &frame.rows()[*index];
            ForecastActual {
                series_id: row.series_id.clone(),
                timestamp: row.timestamp,
                horizon: 1,
                actual: row.target,
            }
        })
        .collect()
}

fn aggregate_metrics(folds: &[BacktestFoldResult]) -> Option<ForecastMetricSet> {
    if folds.is_empty() {
        return None;
    }
    let n = folds.len() as f64;
    let mase_values = folds
        .iter()
        .filter_map(|fold| fold.metrics.mase)
        .collect::<Vec<_>>();
    Some(ForecastMetricSet {
        mae: folds.iter().map(|fold| fold.metrics.mae).sum::<f64>() / n,
        rmse: folds.iter().map(|fold| fold.metrics.rmse).sum::<f64>() / n,
        wape: folds.iter().map(|fold| fold.metrics.wape).sum::<f64>() / n,
        smape: folds.iter().map(|fold| fold.metrics.smape).sum::<f64>() / n,
        bias: folds.iter().map(|fold| fold.metrics.bias).sum::<f64>() / n,
        mase: (!mase_values.is_empty())
            .then(|| mase_values.iter().sum::<f64>() / mase_values.len() as f64),
    })
}
