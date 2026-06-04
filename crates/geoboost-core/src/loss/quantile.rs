use super::Loss;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantileLossConfig {
    pub alpha: f64,
}

impl QuantileLossConfig {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantileLoss {
    pub alpha: f64,
}

impl QuantileLoss {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Loss for QuantileLoss {
    fn initial_prediction(&self, y: &[f64], w: Option<&[f64]>) -> f64 {
        let unit_weights;
        let weights = match w {
            Some(weights) => weights,
            None => {
                unit_weights = vec![1.0; y.len()];
                &unit_weights
            }
        };
        weighted_quantile(y, weights, self.alpha)
    }

    fn gradient(&self, y: f64, pred: f64) -> f64 {
        if y > pred {
            -self.alpha
        } else {
            1.0 - self.alpha
        }
    }
}

pub fn pinball_loss(value: f64, prediction: f64, alpha: f64) -> f64 {
    let residual = value - prediction;
    if residual >= 0.0 {
        alpha * residual
    } else {
        (alpha - 1.0) * residual
    }
}

pub fn weighted_pinball_loss(
    values: &[f64],
    weights: &[f64],
    indices: &[usize],
    alpha: f64,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let selected_values = indices.iter().map(|&idx| values[idx]).collect::<Vec<_>>();
    let selected_weights = indices.iter().map(|&idx| weights[idx]).collect::<Vec<_>>();
    let prediction = weighted_quantile(&selected_values, &selected_weights, alpha);
    indices
        .iter()
        .map(|&idx| weights[idx] * pinball_loss(values[idx], prediction, alpha))
        .sum()
}

pub fn weighted_quantile(values: &[f64], weights: &[f64], alpha: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut pairs = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter(|(value, weight)| value.is_finite() && weight.is_finite() && *weight > 0.0)
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return 0.0;
    }
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total_weight = pairs.iter().map(|(_, weight)| *weight).sum::<f64>();
    let threshold = alpha.clamp(0.0, 1.0) * total_weight;
    let mut cumulative = 0.0;
    for (value, weight) in pairs {
        cumulative += weight;
        if cumulative >= threshold {
            return value;
        }
    }
    values[values.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_is_weighted_quantile() {
        let loss = QuantileLoss::new(0.75);

        assert_eq!(loss.initial_prediction(&[0.0, 10.0, 20.0], None), 20.0);
        assert_eq!(
            loss.initial_prediction(&[0.0, 10.0, 20.0], Some(&[10.0, 1.0, 1.0])),
            0.0
        );
    }

    #[test]
    fn pinball_loss_is_asymmetric() {
        assert_eq!(pinball_loss(10.0, 8.0, 0.8), 1.6);
        assert_eq!(pinball_loss(8.0, 10.0, 0.8), 0.3999999999999999);
    }
}
