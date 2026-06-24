use cartoboost_core::forecasting::{
    ForecastFrame, ForecastFrequency, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use cartoboost_core::{CartoBoostError, Result as CoreResult};
use serde_json::{json, Value};
use std::collections::BTreeMap;

use super::dataloader::WindowDataset;
use super::scaler::StandardScaler;
use super::validate_window_config;

#[derive(Debug, Clone, PartialEq)]
pub struct NBeatsConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
}

impl Default for NBeatsConfig {
    fn default() -> Self {
        Self {
            input_size: 8,
            hidden_size: 16,
            epochs: 80,
            learning_rate: 0.01,
        }
    }
}

pub struct NBeatsForecaster {
    config: NBeatsConfig,
    model: DeterministicMlp,
    scaler: Option<StandardScaler>,
    frequency: Option<ForecastFrequency>,
    tails: BTreeMap<String, Vec<f64>>,
    last_rows: BTreeMap<String, ForecastRow>,
}

impl NBeatsForecaster {
    pub fn new(config: NBeatsConfig) -> crate::Result<Self> {
        validate_window_config(config.input_size, config.hidden_size, config.epochs)?;
        if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
            return Err(crate::NeuralError::InvalidArgument(
                "learning_rate must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            model: DeterministicMlp::new(config.input_size, config.hidden_size, 0.017),
            config,
            scaler: None,
            frequency: None,
            tails: BTreeMap::new(),
            last_rows: BTreeMap::new(),
        })
    }

    pub fn config(&self) -> &NBeatsConfig {
        &self.config
    }
}

impl Forecaster for NBeatsForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> CoreResult<()> {
        let dataset = WindowDataset::from_frame(frame, self.config.input_size)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        let scaler = StandardScaler::fit(
            &frame
                .rows()
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>(),
        )
        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        let examples = dataset
            .windows()
            .iter()
            .map(|window| {
                (
                    scaler.transform_slice(&window.inputs),
                    scaler.transform(window.target),
                )
            })
            .collect::<Vec<_>>();
        self.model = DeterministicMlp::new(self.config.input_size, self.config.hidden_size, 0.017);
        self.model
            .fit(&examples, self.config.epochs, self.config.learning_rate);
        self.scaler = Some(scaler);
        self.frequency = Some(dataset.frequency());
        self.tails = dataset.tails().clone();
        self.last_rows = frame
            .series_ids()
            .into_iter()
            .filter_map(|series_id| {
                frame
                    .rows_for_series(&series_id)
                    .last()
                    .map(|row| (series_id, (*row).clone()))
            })
            .collect();
        Ok(())
    }

    fn predict(&self, horizon: usize) -> CoreResult<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let scaler = self.scaler.ok_or_else(|| {
            CartoBoostError::InvalidInput("NBeatsForecaster must be fit before predict".to_string())
        })?;
        let frequency = self.frequency.ok_or_else(|| {
            CartoBoostError::InvalidInput("NBeatsForecaster must be fit before predict".to_string())
        })?;
        let mut predictions = Vec::new();
        for (series_id, tail) in &self.tails {
            let last_row = self.last_rows.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted timestamp tail for series '{series_id}'"
                ))
            })?;
            let mut history = tail.clone();
            for step in 1..=horizon {
                let scaled_input = scaler.transform_slice(&history);
                let scaled_prediction = self.model.predict(&scaled_input);
                let mean = scaler.inverse_transform(scaled_prediction);
                let timestamp = frequency.advance(last_row.timestamp, step)?;
                predictions.push(ForecastPrediction {
                    series_id: series_id.clone(),
                    timestamp,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean,
                });
                history.remove(0);
                history.push(mean);
            }
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "nbeats"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "input_size": self.config.input_size,
            "hidden_size": self.config.hidden_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DeterministicMlp {
    input_size: usize,
    hidden_size: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
}

impl DeterministicMlp {
    pub(crate) fn new(input_size: usize, hidden_size: usize, phase: f64) -> Self {
        let mut w1 = vec![0.0; input_size * hidden_size];
        for hidden in 0..hidden_size {
            for input in 0..input_size {
                let idx = hidden * input_size + input;
                w1[idx] = (((idx + 1) as f64 * phase).sin()) / input_size as f64;
            }
        }
        let b1 = vec![0.0; hidden_size];
        let w2 = (0..hidden_size)
            .map(|idx| (((idx + 3) as f64 * phase).cos()) / hidden_size as f64)
            .collect();
        Self {
            input_size,
            hidden_size,
            w1,
            b1,
            w2,
            b2: 0.0,
        }
    }

    pub(crate) fn fit(&mut self, examples: &[(Vec<f64>, f64)], epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            for (input, target) in examples {
                self.train_one(input, *target, learning_rate);
            }
        }
    }

    pub(crate) fn predict(&self, input: &[f64]) -> f64 {
        let hidden = self.hidden(input);
        self.b2
            + hidden
                .iter()
                .zip(&self.w2)
                .map(|(activation, weight)| activation * weight)
                .sum::<f64>()
    }

    fn train_one(&mut self, input: &[f64], target: f64, learning_rate: f64) {
        let hidden = self.hidden(input);
        let prediction = self.b2
            + hidden
                .iter()
                .zip(&self.w2)
                .map(|(activation, weight)| activation * weight)
                .sum::<f64>();
        let error_grad = 2.0 * (prediction - target);
        let old_w2 = self.w2.clone();
        for (weight, activation) in self.w2.iter_mut().zip(&hidden) {
            *weight -= learning_rate * error_grad * activation;
        }
        self.b2 -= learning_rate * error_grad;
        for hidden_idx in 0..self.hidden_size {
            let tanh_derivative = 1.0 - hidden[hidden_idx] * hidden[hidden_idx];
            let grad_hidden = error_grad * old_w2[hidden_idx] * tanh_derivative;
            self.b1[hidden_idx] -= learning_rate * grad_hidden;
            for (input_idx, input_value) in input.iter().enumerate().take(self.input_size) {
                let idx = hidden_idx * self.input_size + input_idx;
                self.w1[idx] -= learning_rate * grad_hidden * input_value;
            }
        }
    }

    fn hidden(&self, input: &[f64]) -> Vec<f64> {
        (0..self.hidden_size)
            .map(|hidden_idx| {
                let start = hidden_idx * self.input_size;
                let linear = self.b1[hidden_idx]
                    + input
                        .iter()
                        .take(self.input_size)
                        .enumerate()
                        .map(|(input_idx, value)| self.w1[start + input_idx] * value)
                        .sum::<f64>();
                linear.tanh()
            })
            .collect()
    }
}
