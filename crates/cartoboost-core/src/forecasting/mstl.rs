use crate::forecasting::stl::{
    moving_average_trend, normalize_window, seasonal_pattern, validate_season_length,
    validate_trend_window, validate_values,
};
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct MSTLDecomposition {
    season_lengths: Vec<usize>,
    trend_window: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MSTLSeasonalComponent {
    pub season_length: usize,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MSTLDecompositionResult {
    pub observed: Vec<f64>,
    pub trend: Vec<f64>,
    pub seasonal_components: Vec<MSTLSeasonalComponent>,
    pub remainder: Vec<f64>,
}

impl MSTLDecomposition {
    pub fn new(season_lengths: Vec<usize>) -> Result<Self> {
        Self::with_trend_window(season_lengths, None)
    }

    pub fn with_trend_window(
        mut season_lengths: Vec<usize>,
        trend_window: Option<usize>,
    ) -> Result<Self> {
        if season_lengths.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "mstl requires at least one season length".to_string(),
            ));
        }
        season_lengths.sort_unstable();
        season_lengths.dedup();
        for &season_length in &season_lengths {
            validate_season_length(season_length)?;
        }
        if let Some(window) = trend_window {
            validate_trend_window(window)?;
        }
        Ok(Self {
            season_lengths,
            trend_window,
        })
    }

    pub fn season_lengths(&self) -> &[usize] {
        &self.season_lengths
    }

    pub fn trend_window(&self) -> Option<usize> {
        self.trend_window
    }

    pub fn decompose(&self, values: &[f64]) -> Result<MSTLDecompositionResult> {
        validate_values(values)?;
        let trend_window = self
            .trend_window
            .unwrap_or_else(|| self.season_lengths.iter().max().copied().unwrap_or(2) * 2 + 1);
        let trend = moving_average_trend(values, normalize_window(trend_window, values.len()));
        let mut residual = values
            .iter()
            .zip(&trend)
            .map(|(value, trend)| value - trend)
            .collect::<Vec<_>>();
        let mut seasonal_components = Vec::with_capacity(self.season_lengths.len());
        for &season_length in &self.season_lengths {
            let pattern = seasonal_pattern(&residual, season_length);
            let seasonal = (0..values.len())
                .map(|idx| pattern[idx % season_length])
                .collect::<Vec<_>>();
            for (remaining, seasonal_value) in residual.iter_mut().zip(&seasonal) {
                *remaining -= seasonal_value;
            }
            seasonal_components.push(MSTLSeasonalComponent {
                season_length,
                values: seasonal,
            });
        }
        Ok(MSTLDecompositionResult {
            observed: values.to_vec(),
            trend,
            seasonal_components,
            remainder: residual,
        })
    }

    pub fn metadata(&self) -> Value {
        json!({
            "method": "mstl",
            "season_lengths": self.season_lengths,
            "trend_window": self.trend_window,
        })
    }
}

impl MSTLDecompositionResult {
    pub fn len(&self) -> usize {
        self.observed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observed.is_empty()
    }

    pub fn total_seasonal(&self) -> Vec<f64> {
        let mut total = vec![0.0; self.observed.len()];
        for component in &self.seasonal_components {
            for (sum, value) in total.iter_mut().zip(&component.values) {
                *sum += value;
            }
        }
        total
    }

    pub fn recompose(&self) -> Vec<f64> {
        let seasonal = self.total_seasonal();
        self.trend
            .iter()
            .zip(seasonal)
            .zip(&self.remainder)
            .map(|((trend, seasonal), remainder)| trend + seasonal + remainder)
            .collect()
    }

    pub fn max_abs_recomposition_error(&self) -> f64 {
        self.observed
            .iter()
            .zip(self.recompose())
            .map(|(observed, recomposed)| (observed - recomposed).abs())
            .fold(0.0, f64::max)
    }
}
