use crate::{CartoBoostError, Result};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct LocalLinearKalmanConfig {
    pub level_process_variance: f64,
    pub trend_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLinearKalmanState {
    pub level: f64,
    pub trend: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLinearKalmanEstimate {
    pub step: usize,
    pub observed: f64,
    pub prior_level: f64,
    pub prior_trend: f64,
    pub level: f64,
    pub trend: f64,
    pub innovation: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalLinearKalmanResult {
    pub final_state: LocalLinearKalmanState,
    pub estimates: Vec<LocalLinearKalmanEstimate>,
}

#[derive(Debug, Clone, Copy)]
pub struct LocalLevelKalmanConfig {
    pub level_process_variance: f64,
    pub observation_variance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLevelKalmanEstimate {
    pub step: usize,
    pub observed: f64,
    pub prior_level: f64,
    pub level: f64,
    pub innovation: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalLevelKalmanResult {
    pub final_level: f64,
    pub estimates: Vec<LocalLevelKalmanEstimate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntermittentDemandMethod {
    Croston,
    Sba,
    Tsb,
}

#[derive(Debug, Clone, Copy)]
pub struct OrdinaryKrigingConfig {
    pub range: f64,
    pub nugget: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KrigingObservation {
    pub x: f64,
    pub y: f64,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KrigingPrediction {
    pub x: f64,
    pub y: f64,
    pub mean: f64,
    pub weights: Vec<f64>,
}

impl LocalLinearKalmanConfig {
    pub fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        validate_positive_finite(level_process_variance, "level_process_variance")?;
        validate_positive_finite(trend_process_variance, "trend_process_variance")?;
        validate_positive_finite(observation_variance, "observation_variance")?;
        Ok(Self {
            level_process_variance,
            trend_process_variance,
            observation_variance,
        })
    }
}

impl LocalLevelKalmanConfig {
    pub fn new(level_process_variance: f64, observation_variance: f64) -> Result<Self> {
        validate_positive_finite(level_process_variance, "level_process_variance")?;
        validate_positive_finite(observation_variance, "observation_variance")?;
        Ok(Self {
            level_process_variance,
            observation_variance,
        })
    }
}

impl OrdinaryKrigingConfig {
    pub fn new(range: f64, nugget: f64) -> Result<Self> {
        validate_positive_finite(range, "range")?;
        if !nugget.is_finite() || nugget < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        Ok(Self { range, nugget })
    }
}

pub fn fit_local_level_kalman(
    values: &[f64],
    config: LocalLevelKalmanConfig,
) -> Result<LocalLevelKalmanResult> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman filter requires at least one observation".to_string(),
        ));
    }
    validate_numeric_series(values, "kalman observation")?;
    let mut level = values[0];
    let mut variance = config.observation_variance;
    let mut estimates = Vec::with_capacity(values.len().saturating_sub(1));
    for (idx, observed) in values.iter().enumerate().skip(1) {
        let prior_level = level;
        let prior_variance = variance + config.level_process_variance;
        let innovation = observed - prior_level;
        let innovation_variance = prior_variance + config.observation_variance;
        if innovation_variance <= 0.0 || !innovation_variance.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "local level kalman innovation variance is not positive".to_string(),
            ));
        }
        let gain = prior_variance / innovation_variance;
        level = prior_level + gain * innovation;
        variance = (1.0 - gain) * prior_variance;
        estimates.push(LocalLevelKalmanEstimate {
            step: idx,
            observed: *observed,
            prior_level,
            level,
            innovation,
        });
    }
    Ok(LocalLevelKalmanResult {
        final_level: level,
        estimates,
    })
}

pub fn local_level_kalman_forecast(final_level: f64, horizon: usize) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman forecast horizon must be positive".to_string(),
        ));
    }
    if !final_level.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman final level must be finite".to_string(),
        ));
    }
    Ok(vec![final_level; horizon])
}

pub fn fit_local_linear_kalman(
    values: &[f64],
    config: LocalLinearKalmanConfig,
) -> Result<LocalLinearKalmanResult> {
    if values.len() < 2 {
        return Err(CartoBoostError::InvalidInput(
            "local linear kalman filter requires at least two observations".to_string(),
        ));
    }
    validate_numeric_series(values, "kalman observation")?;
    let mut level = values[0];
    let mut trend = values[1] - values[0];
    let mut p00 = config.observation_variance;
    let mut p01 = 0.0;
    let mut p10 = 0.0;
    let mut p11 = config.observation_variance;
    let mut estimates = Vec::with_capacity(values.len() - 1);
    for (idx, observed) in values.iter().enumerate().skip(1) {
        let prior_level = level + trend;
        let prior_trend = trend;
        let pp00 = p00 + p01 + p10 + p11 + config.level_process_variance;
        let pp01 = p01 + p11;
        let pp10 = p10 + p11;
        let pp11 = p11 + config.trend_process_variance;

        let innovation = observed - prior_level;
        let innovation_variance = pp00 + config.observation_variance;
        if innovation_variance <= 0.0 || !innovation_variance.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "kalman innovation variance is not positive".to_string(),
            ));
        }
        let k0 = pp00 / innovation_variance;
        let k1 = pp10 / innovation_variance;
        level = prior_level + k0 * innovation;
        trend = prior_trend + k1 * innovation;
        p00 = (1.0 - k0) * pp00;
        p01 = (1.0 - k0) * pp01;
        p10 = pp10 - k1 * pp00;
        p11 = pp11 - k1 * pp01;
        estimates.push(LocalLinearKalmanEstimate {
            step: idx,
            observed: *observed,
            prior_level,
            prior_trend,
            level,
            trend,
            innovation,
        });
    }
    Ok(LocalLinearKalmanResult {
        final_state: LocalLinearKalmanState { level, trend },
        estimates,
    })
}

pub fn intermittent_demand_forecast(
    values: &[f64],
    horizon: usize,
    alpha: f64,
    beta: f64,
    method: IntermittentDemandMethod,
) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand forecast horizon must be positive".to_string(),
        ));
    }
    validate_unit_interval("alpha", alpha)?;
    validate_unit_interval("beta", beta)?;
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand forecast requires at least one observation".to_string(),
        ));
    }
    validate_numeric_series(values, "intermittent demand observation")?;
    for (idx, value) in values.iter().enumerate() {
        if *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent demand observation at index {idx} must be non-negative"
            )));
        }
    }
    match method {
        IntermittentDemandMethod::Croston | IntermittentDemandMethod::Sba => {
            let estimate = croston_level(values, alpha)?;
            let adjusted = if method == IntermittentDemandMethod::Sba {
                estimate * (1.0 - alpha / 2.0)
            } else {
                estimate
            };
            Ok(vec![adjusted; horizon])
        }
        IntermittentDemandMethod::Tsb => {
            let estimate = tsb_level(values, alpha, beta)?;
            Ok(vec![estimate; horizon])
        }
    }
}

pub fn local_linear_kalman_forecast(
    state: LocalLinearKalmanState,
    horizon: usize,
) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast horizon must be positive".to_string(),
        ));
    }
    Ok((1..=horizon)
        .map(|step| state.level + step as f64 * state.trend)
        .collect())
}

pub fn ordinary_kriging_predict_many(
    observations: &[KrigingObservation],
    targets: &[(f64, f64)],
    config: OrdinaryKrigingConfig,
) -> Result<Vec<KrigingPrediction>> {
    validate_kriging_observations(observations)?;
    if targets.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging targets must not be empty".to_string(),
        ));
    }
    targets
        .par_iter()
        .map(|target| ordinary_kriging_predict_unchecked(observations, *target, config))
        .collect()
}

pub fn ordinary_kriging_predict(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Result<KrigingPrediction> {
    validate_kriging_observations(observations)?;
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging target coordinates must be finite".to_string(),
        ));
    }
    ordinary_kriging_predict_unchecked(observations, target, config)
}

fn ordinary_kriging_predict_unchecked(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Result<KrigingPrediction> {
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging target coordinates must be finite".to_string(),
        ));
    }
    let n = observations.len();
    let mut matrix = vec![vec![0.0; n + 1]; n + 1];
    let mut rhs = vec![0.0; n + 1];
    for (i, left) in observations.iter().enumerate() {
        for (j, right) in observations.iter().enumerate() {
            matrix[i][j] =
                exponential_covariance((left.x, left.y), (right.x, right.y), config.range);
            if i == j {
                matrix[i][j] += config.nugget;
            }
        }
        matrix[i][n] = 1.0;
        matrix[n][i] = 1.0;
        rhs[i] = exponential_covariance((left.x, left.y), target, config.range);
    }
    rhs[n] = 1.0;
    let weights = solve_linear_system(matrix, rhs).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "kriging system is singular; adjust coordinates or nugget".to_string(),
        )
    })?;
    let mean = observations
        .iter()
        .enumerate()
        .map(|(idx, observation)| weights[idx] * observation.value)
        .sum::<f64>();
    if !mean.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging estimate is not finite".to_string(),
        ));
    }
    Ok(KrigingPrediction {
        x: target.0,
        y: target.1,
        mean,
        weights: weights.into_iter().take(n).collect(),
    })
}

pub fn validate_positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be positive and finite"
        )));
    }
    Ok(())
}

fn validate_numeric_series(values: &[f64], label: &str) -> Result<()> {
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "{label} at index {idx} must be finite"
            )));
        }
    }
    Ok(())
}

fn validate_unit_interval(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be in (0, 1]"
        )));
    }
    Ok(())
}

fn croston_level(values: &[f64], alpha: f64) -> Result<f64> {
    let first_nonzero = values
        .iter()
        .position(|value| *value > 0.0)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "intermittent demand forecast requires at least one non-zero observation"
                    .to_string(),
            )
        })?;
    let mut demand = values[first_nonzero];
    let mut interval = (first_nonzero + 1) as f64;
    let mut elapsed = 0usize;
    for value in values.iter().skip(first_nonzero + 1) {
        elapsed += 1;
        if *value > 0.0 {
            demand += alpha * (*value - demand);
            interval += alpha * (elapsed as f64 - interval);
            elapsed = 0;
        }
    }
    if interval <= 0.0 || !interval.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand interval estimate is invalid".to_string(),
        ));
    }
    Ok(demand / interval)
}

fn tsb_level(values: &[f64], alpha: f64, beta: f64) -> Result<f64> {
    let first_nonzero = values.iter().find(|value| **value > 0.0).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "TSB forecast requires at least one non-zero observation".to_string(),
        )
    })?;
    let mut demand = *first_nonzero;
    let mut probability = 0.0;
    for value in values {
        let occurrence = if *value > 0.0 { 1.0 } else { 0.0 };
        probability += beta * (occurrence - probability);
        if *value > 0.0 {
            demand += alpha * (*value - demand);
        }
    }
    Ok(probability * demand)
}

fn validate_kriging_observations(observations: &[KrigingObservation]) -> Result<()> {
    if observations.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging observations must not be empty".to_string(),
        ));
    }
    for (idx, observation) in observations.iter().enumerate() {
        if !observation.x.is_finite() || !observation.y.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging observation {idx} coordinates must be finite"
            )));
        }
        if !observation.value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging observation {idx} value must be finite"
            )));
        }
    }
    Ok(())
}

fn exponential_covariance(left: (f64, f64), right: (f64, f64), range: f64) -> f64 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    let distance = (dx * dx + dy * dy).sqrt();
    (-distance / range).exp()
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in (pivot + 1)..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if matrix[best][pivot].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= divisor;
        }
        rhs[pivot] /= divisor;
        let pivot_row = matrix[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for (value, pivot_value) in matrix[row].iter_mut().zip(pivot_row.iter()).skip(pivot) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman_filter_tracks_linear_trend_and_forecasts() {
        let config = LocalLinearKalmanConfig::new(0.01, 0.001, 0.1).expect("config");
        let result = fit_local_linear_kalman(&[12.0, 14.0, 16.0, 18.0], config).expect("filter");
        let forecast = local_linear_kalman_forecast(result.final_state, 2).expect("forecast");

        assert_eq!(result.estimates.len(), 3);
        assert!(result.final_state.trend > 0.0);
        assert!(forecast[1] > forecast[0]);
    }

    #[test]
    fn kalman_filter_rejects_bad_inputs() {
        let config = LocalLinearKalmanConfig::new(0.01, 0.001, 0.1).expect("config");

        assert!(fit_local_linear_kalman(&[1.0], config).is_err());
        assert!(fit_local_linear_kalman(&[1.0, f64::NAN], config).is_err());
        assert!(LocalLinearKalmanConfig::new(0.0, 0.001, 0.1).is_err());
        assert!(local_linear_kalman_forecast(
            LocalLinearKalmanState {
                level: 1.0,
                trend: 0.0
            },
            0
        )
        .is_err());
    }

    #[test]
    fn local_level_kalman_forecasts_flat_level() {
        let config = LocalLevelKalmanConfig::new(0.01, 0.1).expect("config");
        let result = fit_local_level_kalman(&[12.0, 13.0, 13.5], config).expect("filter");
        let forecast = local_level_kalman_forecast(result.final_level, 3).expect("forecast");

        assert_eq!(result.estimates.len(), 2);
        assert_eq!(forecast, vec![result.final_level; 3]);
    }

    #[test]
    fn intermittent_demand_methods_are_positive_and_bias_adjusted() {
        let values = [0.0, 0.0, 5.0, 0.0, 0.0, 7.0, 0.0];
        let croston =
            intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Croston)
                .expect("croston");
        let sba = intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Sba)
            .expect("sba");
        let tsb = intermittent_demand_forecast(&values, 2, 0.2, 0.2, IntermittentDemandMethod::Tsb)
            .expect("tsb");

        assert_eq!(croston.len(), 2);
        assert!(croston[0] > 0.0);
        assert!(sba[0] < croston[0]);
        assert!(tsb[0] > 0.0);
    }

    #[test]
    fn intermittent_demand_rejects_invalid_inputs() {
        assert!(intermittent_demand_forecast(
            &[0.0, -1.0],
            1,
            0.1,
            0.1,
            IntermittentDemandMethod::Croston,
        )
        .is_err());
        assert!(intermittent_demand_forecast(
            &[0.0, 0.0],
            1,
            0.1,
            0.1,
            IntermittentDemandMethod::Tsb,
        )
        .is_err());
        assert!(intermittent_demand_forecast(
            &[1.0],
            0,
            0.1,
            0.1,
            IntermittentDemandMethod::Croston,
        )
        .is_err());
    }

    #[test]
    fn ordinary_kriging_returns_exact_known_coordinate_with_tiny_nugget() {
        let observations = vec![
            KrigingObservation {
                x: 0.0,
                y: 0.0,
                value: 12.0,
            },
            KrigingObservation {
                x: 10.0,
                y: 0.0,
                value: 42.0,
            },
        ];
        let config = OrdinaryKrigingConfig::new(1.0, 1.0e-9).expect("config");

        let prediction =
            ordinary_kriging_predict(&observations, (0.0, 0.0), config).expect("kriging");

        assert!((prediction.mean - 12.0).abs() < 1.0e-4);
        assert_eq!(prediction.weights.len(), 2);
    }

    #[test]
    fn ordinary_kriging_rejects_bad_inputs() {
        let config = OrdinaryKrigingConfig::new(1.0, 1.0e-6).expect("config");

        assert!(ordinary_kriging_predict(&[], (0.0, 0.0), config).is_err());
        assert!(OrdinaryKrigingConfig::new(0.0, 1.0e-6).is_err());
        assert!(OrdinaryKrigingConfig::new(1.0, -1.0).is_err());
        assert!(ordinary_kriging_predict(
            &[KrigingObservation {
                x: 0.0,
                y: f64::NAN,
                value: 1.0,
            }],
            (0.0, 0.0),
            config,
        )
        .is_err());
    }
}
