use rayon::prelude::*;

pub mod m6;
pub mod pinball;
pub mod rps;
pub mod wrmsse;
pub use m6::{evaluate_m6_metrics, m6_combined_score, M6MetricSummary};
pub use pinball::pinball_loss;
pub use rps::rank_probability_score;
pub use wrmsse::{rmsse_scale, wrmsse, WrmsseScore, WrmsseSeries, WrmsseSeriesScore};

pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .par_iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / y_true.len().max(1) as f64
}

pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    (y_true
        .par_iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / y_true.len().max(1) as f64)
        .sqrt()
}

pub fn volatility(pred: &[f64]) -> f64 {
    if pred.len() < 2 {
        return 0.0;
    }
    pred.par_windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum::<f64>()
        / (pred.len() - 1) as f64
}
