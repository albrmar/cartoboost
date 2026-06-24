use crate::{CartoBoostError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct QuantileForecast {
    pub quantiles: Vec<f64>,
    pub values: Vec<f64>,
}

impl QuantileForecast {
    pub fn new(quantiles: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        validate_quantile_grid(&quantiles)?;
        validate_finite_values(&values, "values")?;
        if quantiles.len() != values.len() {
            return Err(CartoBoostError::InvalidInput(
                "quantiles and values must have the same length".to_string(),
            ));
        }
        Ok(Self { quantiles, values })
    }

    pub fn repaired(quantiles: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        validate_quantile_grid(&quantiles)?;
        let values = repair_non_crossing_quantiles(&values)?;
        Self::new(quantiles, values)
    }
}

pub fn pinball_loss(actual: &[f64], prediction: &[f64], quantile: f64) -> Result<f64> {
    validate_quantile(quantile)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
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

pub fn repair_non_crossing_quantiles(values: &[f64]) -> Result<Vec<f64>> {
    validate_finite_values(values, "values")?;
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "values must contain at least one quantile prediction".to_string(),
        ));
    }
    let mut repaired = values.to_vec();
    for idx in 1..repaired.len() {
        if repaired[idx] < repaired[idx - 1] {
            repaired[idx] = repaired[idx - 1];
        }
    }
    Ok(repaired)
}

pub(crate) fn validate_quantile_grid(quantiles: &[f64]) -> Result<()> {
    if quantiles.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "quantiles must contain at least one level".to_string(),
        ));
    }
    let mut previous = f64::NEG_INFINITY;
    for &q in quantiles {
        validate_quantile(q)?;
        if q <= previous {
            return Err(CartoBoostError::InvalidInput(
                "quantiles must be strictly increasing".to_string(),
            ));
        }
        previous = q;
    }
    Ok(())
}

pub(crate) fn validate_quantile(quantile: f64) -> Result<()> {
    if !quantile.is_finite() || quantile <= 0.0 || quantile >= 1.0 {
        return Err(CartoBoostError::InvalidInput(
            "quantile must be finite and in (0, 1)".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_finite_values(values: &[f64], name: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

pub(crate) fn validate_same_non_empty(
    left: &[f64],
    right: &[f64],
    left_name: &str,
    right_name: &str,
) -> Result<()> {
    validate_finite_values(left, left_name)?;
    validate_finite_values(right, right_name)?;
    if left.len() != right.len() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{left_name} and {right_name} must have the same length"
        )));
    }
    if left.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{left_name} and {right_name} must contain at least one value"
        )));
    }
    Ok(())
}
