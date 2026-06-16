pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / y_true.len().max(1) as f64
}

pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    (y_true
        .iter()
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
    pred.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (pred.len() - 1) as f64
}
