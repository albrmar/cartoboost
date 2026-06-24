use crate::Result;

use super::quantiles::{repair_non_crossing_quantiles, validate_quantile_grid, QuantileForecast};

pub trait ProbabilisticForecaster {
    fn predict_quantiles(&self, horizon: usize, quantiles: &[f64])
        -> Result<Vec<QuantileForecast>>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilisticDirectForecaster {
    quantiles: Vec<f64>,
}

impl ProbabilisticDirectForecaster {
    pub fn new(quantiles: Vec<f64>) -> Result<Self> {
        validate_quantile_grid(&quantiles)?;
        Ok(Self { quantiles })
    }

    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    pub fn repair_horizon(&self, values: &[f64]) -> Result<QuantileForecast> {
        QuantileForecast::new(
            self.quantiles.clone(),
            repair_non_crossing_quantiles(values)?,
        )
    }

    pub fn repair_matrix(&self, horizon_values: &[Vec<f64>]) -> Result<Vec<QuantileForecast>> {
        horizon_values
            .iter()
            .map(|values| self.repair_horizon(values))
            .collect()
    }
}
