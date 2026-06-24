use cartoboost_core::forecasting::{ForecastFrame, ForecastFrequency};
use std::collections::BTreeMap;

use crate::{NeuralError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ForecastWindow {
    pub series_id: String,
    pub inputs: Vec<f64>,
    pub target: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowDataset {
    windows: Vec<ForecastWindow>,
    tails: BTreeMap<String, Vec<f64>>,
    frequency: ForecastFrequency,
}

impl WindowDataset {
    pub fn from_frame(frame: &ForecastFrame, input_size: usize) -> Result<Self> {
        if input_size == 0 {
            return Err(NeuralError::InvalidArgument(
                "input_size must be positive".to_string(),
            ));
        }
        let mut windows = Vec::new();
        let mut tails = BTreeMap::new();
        for series_id in frame.series_ids() {
            let rows = frame.rows_for_series(&series_id);
            if rows.len() <= input_size {
                return Err(NeuralError::InvalidArgument(format!(
                    "series '{series_id}' needs more than {input_size} rows"
                )));
            }
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            for end in input_size..values.len() {
                windows.push(ForecastWindow {
                    series_id: series_id.clone(),
                    inputs: values[end - input_size..end].to_vec(),
                    target: values[end],
                });
            }
            tails.insert(series_id, values[values.len() - input_size..].to_vec());
        }
        Ok(Self {
            windows,
            tails,
            frequency: frame.frequency(),
        })
    }

    pub fn windows(&self) -> &[ForecastWindow] {
        &self.windows
    }

    pub fn tails(&self) -> &BTreeMap<String, Vec<f64>> {
        &self.tails
    }

    pub fn frequency(&self) -> ForecastFrequency {
        self.frequency
    }
}
