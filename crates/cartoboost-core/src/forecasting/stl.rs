use crate::{CartoBoostError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct STLDecomposition {
    season_length: usize,
    trend_window: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct STLDecompositionResult {
    pub observed: Vec<f64>,
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub remainder: Vec<f64>,
}

impl STLDecomposition {
    pub fn new(season_length: usize) -> Result<Self> {
        Self::with_trend_window(season_length, None)
    }

    pub fn with_trend_window(season_length: usize, trend_window: Option<usize>) -> Result<Self> {
        validate_season_length(season_length)?;
        if let Some(window) = trend_window {
            validate_trend_window(window)?;
        }
        Ok(Self {
            season_length,
            trend_window,
        })
    }

    pub fn season_length(&self) -> usize {
        self.season_length
    }

    pub fn trend_window(&self) -> Option<usize> {
        self.trend_window
    }

    pub fn decompose(&self, values: &[f64]) -> Result<STLDecompositionResult> {
        validate_values(values)?;
        let trend = moving_average_trend(values, self.effective_trend_window(values.len()));
        let detrended = values
            .iter()
            .zip(&trend)
            .map(|(value, trend)| value - trend)
            .collect::<Vec<_>>();
        let pattern = seasonal_pattern(&detrended, self.season_length);
        let seasonal = (0..values.len())
            .map(|idx| pattern[idx % self.season_length])
            .collect::<Vec<_>>();
        let remainder = values
            .iter()
            .zip(&trend)
            .zip(&seasonal)
            .map(|((value, trend), seasonal)| value - trend - seasonal)
            .collect::<Vec<_>>();
        Ok(STLDecompositionResult {
            observed: values.to_vec(),
            trend,
            seasonal,
            remainder,
        })
    }

    pub fn metadata(&self) -> Value {
        json!({
            "method": "stl",
            "season_length": self.season_length,
            "trend_window": self.trend_window,
        })
    }

    fn effective_trend_window(&self, n: usize) -> usize {
        let requested = self
            .trend_window
            .unwrap_or_else(|| self.season_length.saturating_mul(2).saturating_add(1));
        normalize_window(requested, n)
    }
}

impl STLDecompositionResult {
    pub fn len(&self) -> usize {
        self.observed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observed.is_empty()
    }

    pub fn recompose(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(&self.seasonal)
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

    pub fn seasonal_pattern(&self, season_length: usize) -> Vec<f64> {
        seasonal_pattern(&self.seasonal, season_length)
    }
}

pub(crate) fn validate_values(values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "decomposition requires at least one observation".to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "decomposition values must be finite".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_season_length(season_length: usize) -> Result<()> {
    if season_length <= 1 {
        return Err(CartoBoostError::InvalidInput(
            "season_length must be greater than 1".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_trend_window(window: usize) -> Result<()> {
    if window < 3 {
        return Err(CartoBoostError::InvalidInput(
            "trend_window must be at least 3".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn normalize_window(window: usize, n: usize) -> usize {
    if n <= 2 {
        return n.max(1);
    }
    let mut normalized = window.clamp(3, n);
    if normalized.is_multiple_of(2) {
        normalized = normalized.saturating_sub(1).max(3);
    }
    normalized
}

pub(crate) fn moving_average_trend(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let window = normalize_window(window, n);
    let radius = window / 2;
    (0..n)
        .map(|idx| {
            let start = idx.saturating_sub(radius);
            let end = (idx + radius + 1).min(n);
            let count = end - start;
            values[start..end].iter().sum::<f64>() / count as f64
        })
        .collect()
}

pub(crate) fn seasonal_pattern(values: &[f64], season_length: usize) -> Vec<f64> {
    let mut sums = vec![0.0; season_length];
    let mut counts = vec![0usize; season_length];
    for (idx, value) in values.iter().enumerate() {
        let phase = idx % season_length;
        sums[phase] += value;
        counts[phase] += 1;
    }
    let mut pattern = sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| if count == 0 { 0.0 } else { sum / count as f64 })
        .collect::<Vec<_>>();
    let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
    for value in &mut pattern {
        *value -= mean;
    }
    pattern
}
