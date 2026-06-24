use super::{
    finite_exp, validate_derivatives, validate_non_negative_target, CountObjective,
    ObjectiveDerivatives,
};
use crate::{CartoBoostError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialObjective {
    pub dispersion: f64,
}

impl NegativeBinomialObjective {
    pub fn new(dispersion: f64) -> Result<Self> {
        if !dispersion.is_finite() || dispersion <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "negative binomial dispersion must be positive and finite".to_string(),
            ));
        }
        Ok(Self { dispersion })
    }

    pub fn value_gradient_hessian(
        &self,
        target: f64,
        raw_prediction: f64,
    ) -> Result<ObjectiveDerivatives> {
        <Self as CountObjective>::value_gradient_hessian(self, target, raw_prediction)
    }
}

impl CountObjective for NegativeBinomialObjective {
    fn value_gradient_hessian(
        &self,
        target: f64,
        raw_prediction: f64,
    ) -> Result<ObjectiveDerivatives> {
        validate_non_negative_target(target)?;
        let alpha = self.dispersion;
        let mean = finite_exp(raw_prediction)?;
        let denom = 1.0 + alpha * mean;
        let value = (target + 1.0 / alpha) * denom.ln() - target * raw_prediction;
        let gradient = (mean - target) / denom;
        let hessian = mean * (1.0 + alpha * target) / (denom * denom);
        validate_derivatives(value, gradient, hessian)
    }
}
