mod hurdle;
mod negative_binomial;
mod poisson;
mod tweedie;

pub use hurdle::{HurdleDerivatives, HurdleObjective};
pub use negative_binomial::NegativeBinomialObjective;
pub use poisson::PoissonObjective;
pub use tweedie::TweedieObjective;

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveDerivatives {
    pub value: f64,
    pub gradient: f64,
    pub hessian: f64,
}

pub trait CountObjective {
    fn value_gradient_hessian(
        &self,
        target: f64,
        raw_prediction: f64,
    ) -> Result<ObjectiveDerivatives>;
}

pub(crate) fn validate_non_negative_target(target: f64) -> Result<()> {
    if !target.is_finite() || target < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "count objective target must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn finite_exp(raw_prediction: f64) -> Result<f64> {
    if !raw_prediction.is_finite() {
        return Err(crate::CartoBoostError::InvalidInput(
            "count objective raw prediction must be finite".to_string(),
        ));
    }
    let clipped = raw_prediction.clamp(-35.0, 35.0);
    Ok(clipped.exp())
}

pub(crate) fn validate_derivatives(
    value: f64,
    gradient: f64,
    hessian: f64,
) -> Result<ObjectiveDerivatives> {
    if !value.is_finite() || !gradient.is_finite() || !hessian.is_finite() || hessian < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "count objective derivatives must be finite with non-negative hessian".to_string(),
        ));
    }
    Ok(ObjectiveDerivatives {
        value,
        gradient,
        hessian,
    })
}
