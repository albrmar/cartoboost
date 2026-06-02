use crate::{GeoBoostError, Result};

pub fn validate_weights(weights: Option<&[f64]>, n: usize) -> Result<Vec<f64>> {
    match weights {
        Some(w) => {
            if w.len() != n {
                return Err(GeoBoostError::InvalidInput(
                    "sample_weight length must match y".to_string(),
                ));
            }
            if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(GeoBoostError::InvalidInput(
                    "sample weights must be finite and non-negative".to_string(),
                ));
            }
            Ok(w.to_vec())
        }
        None => Ok(vec![1.0; n]),
    }
}
