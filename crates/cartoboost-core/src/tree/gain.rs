pub fn sse(values: &[f64], weights: &[f64], indices: &[usize]) -> f64 {
    let weight_sum: f64 = indices.iter().map(|&idx| weights[idx]).sum();
    if weight_sum <= 0.0 {
        return 0.0;
    }
    let mean = indices
        .iter()
        .map(|&idx| values[idx] * weights[idx])
        .sum::<f64>()
        / weight_sum;
    indices
        .iter()
        .map(|&idx| weights[idx] * (values[idx] - mean).powi(2))
        .sum()
}
