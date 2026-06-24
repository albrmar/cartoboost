use crate::{CartoBoostError, Result};

pub fn rank_probability_score(probabilities: &[f64], observed_rank: usize) -> Result<f64> {
    validate_probabilities(probabilities)?;
    if observed_rank >= probabilities.len() {
        return Err(CartoBoostError::InvalidInput(
            "observed_rank must be inside probabilities".to_string(),
        ));
    }
    if probabilities.len() == 1 {
        return Ok(0.0);
    }

    let mut cumulative = 0.0;
    let mut score = 0.0;
    for (rank, probability) in probabilities
        .iter()
        .enumerate()
        .take(probabilities.len() - 1)
    {
        cumulative += probability;
        let observed = if observed_rank <= rank { 1.0 } else { 0.0 };
        score += (cumulative - observed).powi(2);
    }
    Ok(score / (probabilities.len() - 1) as f64)
}

fn validate_probabilities(probabilities: &[f64]) -> Result<()> {
    if probabilities.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "probabilities must not be empty".to_string(),
        ));
    }
    if probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(CartoBoostError::InvalidInput(
            "probabilities must be finite and non-negative".to_string(),
        ));
    }
    if (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
        return Err(CartoBoostError::InvalidInput(
            "probabilities must sum to 1".to_string(),
        ));
    }
    Ok(())
}
