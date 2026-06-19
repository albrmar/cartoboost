use super::{validate_non_negative_target, CountObjective, ObjectiveDerivatives};
use crate::{CartoBoostError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HurdleDerivatives {
    pub value: f64,
    pub zero_gradient: f64,
    pub zero_hessian: f64,
    pub positive: ObjectiveDerivatives,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HurdleObjective<T> {
    pub positive_objective: T,
}

impl<T> HurdleObjective<T> {
    pub fn new(positive_objective: T) -> Self {
        Self { positive_objective }
    }
}

impl<T: CountObjective> HurdleObjective<T> {
    pub fn value_gradient_hessian(
        &self,
        target: f64,
        zero_raw_prediction: f64,
        positive_raw_prediction: f64,
    ) -> Result<HurdleDerivatives> {
        validate_non_negative_target(target)?;
        if !zero_raw_prediction.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "hurdle zero raw prediction must be finite".to_string(),
            ));
        }
        let probability = sigmoid(zero_raw_prediction);
        let zero_hessian = probability * (1.0 - probability);
        let positive = self
            .positive_objective
            .value_gradient_hessian(target, positive_raw_prediction)?;
        let (value, zero_gradient) = if target > 0.0 {
            (
                -(probability.max(1e-15)).ln() + positive.value,
                probability - 1.0,
            )
        } else {
            (-(1.0 - probability).max(1e-15).ln(), probability)
        };
        if !value.is_finite() || !zero_gradient.is_finite() || !zero_hessian.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "hurdle objective derivatives must be finite".to_string(),
            ));
        }
        Ok(HurdleDerivatives {
            value,
            zero_gradient,
            zero_hessian,
            positive,
        })
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}
