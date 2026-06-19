use super::{
    finite_exp, validate_derivatives, validate_non_negative_target, CountObjective,
    ObjectiveDerivatives,
};
use crate::{CartoBoostError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieObjective {
    pub power: f64,
}

impl TweedieObjective {
    pub fn new(power: f64) -> Result<Self> {
        if !power.is_finite() || power <= 1.0 || power >= 2.0 {
            return Err(CartoBoostError::InvalidInput(
                "tweedie power must be finite and in (1, 2)".to_string(),
            ));
        }
        Ok(Self { power })
    }

    pub fn value_gradient_hessian(
        &self,
        target: f64,
        raw_prediction: f64,
    ) -> Result<ObjectiveDerivatives> {
        <Self as CountObjective>::value_gradient_hessian(self, target, raw_prediction)
    }
}

impl CountObjective for TweedieObjective {
    fn value_gradient_hessian(
        &self,
        target: f64,
        raw_prediction: f64,
    ) -> Result<ObjectiveDerivatives> {
        validate_non_negative_target(target)?;
        let p = self.power;
        let mean = finite_exp(raw_prediction)?;
        let mean_2_minus_p = mean.powf(2.0 - p);
        let mean_1_minus_p = mean.powf(1.0 - p);
        let value = mean_2_minus_p / (2.0 - p) - target * mean_1_minus_p / (1.0 - p);
        let gradient = mean_2_minus_p - target * mean_1_minus_p;
        let hessian = (2.0 - p) * mean_2_minus_p + (p - 1.0) * target * mean_1_minus_p;
        validate_derivatives(value, gradient, hessian)
    }
}
