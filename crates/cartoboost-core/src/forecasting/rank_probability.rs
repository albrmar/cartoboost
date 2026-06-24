use crate::{CartoBoostError, Result};

use super::quantiles::validate_finite_values;

#[derive(Debug, Clone, PartialEq)]
pub struct RankProbabilityForecast {
    pub probabilities: Vec<f64>,
}

impl RankProbabilityForecast {
    pub fn new(probabilities: Vec<f64>) -> Result<Self> {
        validate_probability_vector(&probabilities)?;
        Ok(Self { probabilities })
    }

    pub fn score(&self, observed_rank: usize) -> Result<f64> {
        rank_probability_score(&self.probabilities, observed_rank)
    }
}

pub fn rank_probability_score(probabilities: &[f64], observed_rank: usize) -> Result<f64> {
    validate_probability_vector(probabilities)?;
    if observed_rank >= probabilities.len() {
        return Err(CartoBoostError::InvalidInput(
            "observed_rank must be a zero-based index inside probabilities".to_string(),
        ));
    }
    if probabilities.len() == 1 {
        return Ok(0.0);
    }

    let mut predicted_cdf = 0.0;
    let mut score = 0.0;
    for (rank, probability) in probabilities
        .iter()
        .enumerate()
        .take(probabilities.len() - 1)
    {
        predicted_cdf += probability;
        let observed_cdf = if observed_rank <= rank { 1.0 } else { 0.0 };
        let diff = predicted_cdf - observed_cdf;
        score += diff * diff;
    }
    Ok(score / (probabilities.len() - 1) as f64)
}

pub(crate) fn validate_probability_vector(probabilities: &[f64]) -> Result<()> {
    validate_finite_values(probabilities, "probabilities")?;
    if probabilities.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "probabilities must contain at least one rank".to_string(),
        ));
    }
    if probabilities.iter().any(|&probability| probability < 0.0) {
        return Err(CartoBoostError::InvalidInput(
            "probabilities must be non-negative".to_string(),
        ));
    }
    let sum = probabilities.iter().sum::<f64>();
    if (sum - 1.0).abs() > 1e-9 {
        return Err(CartoBoostError::InvalidInput(
            "probabilities must sum to 1".to_string(),
        ));
    }
    Ok(())
}
