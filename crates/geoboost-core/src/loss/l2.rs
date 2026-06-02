use super::Loss;

#[derive(Debug, Clone, Copy, Default)]
pub struct L2Loss;

impl L2Loss {
    pub fn value(&self, y: &[f64], pred: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        y.iter()
            .zip(pred)
            .map(|(target, prediction)| (target - prediction).powi(2))
            .sum::<f64>()
            / y.len() as f64
    }

    pub fn residuals(&self, y: &[f64], pred: &[f64]) -> Vec<f64> {
        y.iter()
            .zip(pred)
            .map(|(target, prediction)| target - prediction)
            .collect()
    }
}

impl Loss for L2Loss {
    fn initial_prediction(&self, y: &[f64], w: Option<&[f64]>) -> f64 {
        match w {
            Some(weights) => {
                let denom: f64 = weights.iter().sum();
                if denom == 0.0 {
                    0.0
                } else {
                    y.iter().zip(weights).map(|(yi, wi)| yi * wi).sum::<f64>() / denom
                }
            }
            None => {
                if y.is_empty() {
                    0.0
                } else {
                    y.iter().sum::<f64>() / y.len() as f64
                }
            }
        }
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        pred - y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_is_weighted_mean() {
        let loss = L2Loss;
        assert_eq!(loss.initial_prediction(&[0.0, 10.0], None), 5.0);
        assert_eq!(
            loss.initial_prediction(&[0.0, 10.0], Some(&[3.0, 1.0])),
            2.5
        );
    }

    #[test]
    fn gradient_is_pred_minus_target() {
        assert_eq!(L2Loss.gradient(3.0, 10.0), 7.0);
    }

    #[test]
    fn value_and_residuals_are_mean_squared_error_helpers() {
        let loss = L2Loss;

        assert_eq!(loss.value(&[1.0, 2.0, 4.0], &[0.0, 2.5, 3.0]), 0.75);
        assert_eq!(
            loss.residuals(&[1.0, 2.0, 4.0], &[0.0, 2.5, 3.0]),
            vec![1.0, -0.5, 1.0]
        );
    }
}
