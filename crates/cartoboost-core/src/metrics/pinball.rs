use crate::{CartoBoostError, Result};

pub fn pinball_loss(actual: &[f64], prediction: &[f64], quantile: f64) -> Result<f64> {
    if !quantile.is_finite() || quantile <= 0.0 || quantile >= 1.0 {
        return Err(CartoBoostError::InvalidInput(
            "quantile must be finite and in (0, 1)".to_string(),
        ));
    }
    if actual.len() != prediction.len() {
        return Err(CartoBoostError::InvalidInput(
            "actual and prediction must have the same length".to_string(),
        ));
    }
    if actual.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "actual and prediction must contain at least one value".to_string(),
        ));
    }
    if actual
        .iter()
        .chain(prediction)
        .any(|value| !value.is_finite())
    {
        return Err(CartoBoostError::InvalidInput(
            "actual and prediction must contain only finite values".to_string(),
        ));
    }
    Ok(actual
        .iter()
        .zip(prediction)
        .map(|(&y, &q)| {
            let residual = y - q;
            (quantile * residual).max((quantile - 1.0) * residual)
        })
        .sum::<f64>()
        / actual.len() as f64)
}
