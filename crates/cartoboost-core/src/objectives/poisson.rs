use super::{
    finite_exp, validate_derivatives, validate_non_negative_target, CountObjective,
    ObjectiveDerivatives,
};
use crate::Result;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PoissonObjective;

impl PoissonObjective {
    pub fn value_gradient_hessian(
        &self,
        target: f64,
        raw_prediction: f64,
    ) -> Result<ObjectiveDerivatives> {
        <Self as CountObjective>::value_gradient_hessian(self, target, raw_prediction)
    }
}

impl CountObjective for PoissonObjective {
    fn value_gradient_hessian(
        &self,
        target: f64,
        raw_prediction: f64,
    ) -> Result<ObjectiveDerivatives> {
        validate_non_negative_target(target)?;
        let mean = finite_exp(raw_prediction)?;
        let value = mean - target * raw_prediction;
        let gradient = mean - target;
        validate_derivatives(value, gradient, mean)
    }
}
