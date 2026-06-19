use crate::{CartoBoostError, Result};

pub fn croston_forecast(values: &[f64], horizon: usize, alpha: f64) -> Result<Vec<f64>> {
    validate_inputs(values, horizon, alpha, None)?;
    Ok(vec![croston_level(values, alpha)?; horizon])
}

pub fn sba_forecast(values: &[f64], horizon: usize, alpha: f64) -> Result<Vec<f64>> {
    validate_inputs(values, horizon, alpha, None)?;
    Ok(vec![
        croston_level(values, alpha)? * (1.0 - alpha / 2.0);
        horizon
    ])
}

pub fn tsb_forecast(values: &[f64], horizon: usize, alpha: f64, beta: f64) -> Result<Vec<f64>> {
    validate_inputs(values, horizon, alpha, Some(beta))?;
    Ok(vec![tsb_level(values, alpha, beta)?; horizon])
}

pub fn adida_forecast(
    values: &[f64],
    horizon: usize,
    bucket_size: usize,
    alpha: f64,
) -> Result<Vec<f64>> {
    validate_inputs(values, horizon, alpha, None)?;
    if bucket_size == 0 {
        return Err(CartoBoostError::InvalidInput(
            "ADIDA bucket_size must be positive".to_string(),
        ));
    }
    let mut buckets = Vec::new();
    for chunk in values.chunks(bucket_size) {
        buckets.push(chunk.iter().sum::<f64>());
    }
    let aggregate = croston_level(&buckets, alpha)? / bucket_size as f64;
    Ok(vec![aggregate; horizon])
}

fn validate_inputs(values: &[f64], horizon: usize, alpha: f64, beta: Option<f64>) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "intermittent forecast horizon must be positive".to_string(),
        ));
    }
    validate_unit_interval("alpha", alpha)?;
    if let Some(beta) = beta {
        validate_unit_interval("beta", beta)?;
    }
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent forecast requires at least one observation".to_string(),
        ));
    }
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent observation at index {idx} must be finite"
            )));
        }
        if *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent observation at index {idx} must be non-negative"
            )));
        }
    }
    Ok(())
}

fn validate_unit_interval(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be in (0, 1]"
        )));
    }
    Ok(())
}

fn croston_level(values: &[f64], alpha: f64) -> Result<f64> {
    let first_nonzero = values
        .iter()
        .position(|value| *value > 0.0)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "intermittent forecast requires at least one non-zero observation".to_string(),
            )
        })?;
    let mut demand = values[first_nonzero];
    let mut interval = (first_nonzero + 1) as f64;
    let mut elapsed = 0usize;
    for value in values.iter().skip(first_nonzero + 1) {
        elapsed += 1;
        if *value > 0.0 {
            demand += alpha * (*value - demand);
            interval += alpha * (elapsed as f64 - interval);
            elapsed = 0;
        }
    }
    if interval <= 0.0 || !interval.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent interval estimate is invalid".to_string(),
        ));
    }
    Ok(demand / interval)
}

fn tsb_level(values: &[f64], alpha: f64, beta: f64) -> Result<f64> {
    let first_nonzero = values.iter().find(|value| **value > 0.0).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "TSB forecast requires at least one non-zero observation".to_string(),
        )
    })?;
    let mut demand = *first_nonzero;
    let mut probability = 0.0;
    for value in values {
        let occurrence = if *value > 0.0 { 1.0 } else { 0.0 };
        probability += beta * (occurrence - probability);
        if *value > 0.0 {
            demand += alpha * (*value - demand);
        }
    }
    Ok(probability * demand)
}
