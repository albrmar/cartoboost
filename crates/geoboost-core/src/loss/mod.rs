mod l2;

pub use l2::L2Loss;

pub trait Loss {
    fn initial_prediction(&self, y: &[f64], w: Option<&[f64]>) -> f64;
    fn gradient(&self, y: f64, pred: f64) -> f64;
    fn hessian(&self, _y: f64, _pred: f64) -> f64 {
        1.0
    }
    fn leaf_value(&self, grad_sum: f64, hess_sum: f64, lambda_l2: f64) -> f64 {
        -grad_sum / (hess_sum + lambda_l2)
    }
}
