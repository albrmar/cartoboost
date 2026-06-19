use crate::Result;

use super::pinball::pinball_loss;
use super::rps::rank_probability_score;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct M6MetricSummary {
    pub pinball_loss: f64,
    pub rank_probability_score: f64,
    pub combined_score: f64,
}

pub fn m6_combined_score(pinball_loss: f64, rank_probability_score: f64) -> Result<f64> {
    if !pinball_loss.is_finite() || pinball_loss < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "pinball_loss must be finite and non-negative".to_string(),
        ));
    }
    if !rank_probability_score.is_finite() || rank_probability_score < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "rank_probability_score must be finite and non-negative".to_string(),
        ));
    }
    Ok(0.5 * pinball_loss + 0.5 * rank_probability_score)
}

pub fn evaluate_m6_metrics(
    actual_returns: &[f64],
    quantile_predictions: &[f64],
    quantile: f64,
    rank_probabilities: &[f64],
    observed_rank: usize,
) -> Result<M6MetricSummary> {
    let pinball = pinball_loss(actual_returns, quantile_predictions, quantile)?;
    let rps = rank_probability_score(rank_probabilities, observed_rank)?;
    Ok(M6MetricSummary {
        pinball_loss: pinball,
        rank_probability_score: rps,
        combined_score: m6_combined_score(pinball, rps)?,
    })
}
