use crate::{GeoBoostError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default)]
pub struct LinearLeafPredictor;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLeafModel {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub features: Vec<usize>,
}

impl LinearLeafPredictor {
    pub fn fit_ridge(
        x: &[Vec<f64>],
        y: &[f64],
        weights: &[f64],
        features: Vec<usize>,
        lambda_l2: f64,
    ) -> Result<LinearLeafModel> {
        if x.len() != y.len() || y.len() != weights.len() {
            return Err(GeoBoostError::InvalidInput(
                "X, y, and weights must have the same row count".to_string(),
            ));
        }
        if x.is_empty() {
            return Err(GeoBoostError::InvalidInput(
                "linear leaf fit requires at least one row".to_string(),
            ));
        }
        if !lambda_l2.is_finite() || lambda_l2 < 0.0 {
            return Err(GeoBoostError::InvalidInput(
                "lambda_l2 must be finite and non-negative".to_string(),
            ));
        }

        let p = features.len() + 1;
        let mut xtwx = vec![vec![0.0; p]; p];
        let mut xtwy = vec![0.0; p];
        for ((row, target), weight) in x.iter().zip(y).zip(weights) {
            if !target.is_finite() || !weight.is_finite() || *weight < 0.0 {
                return Err(GeoBoostError::InvalidInput(
                    "targets and weights must be finite, with non-negative weights".to_string(),
                ));
            }
            let design = design_row(row, &features)?;
            for i in 0..p {
                xtwy[i] += weight * design[i] * target;
                for j in 0..p {
                    xtwx[i][j] += weight * design[i] * design[j];
                }
            }
        }
        for (idx, row) in xtwx.iter_mut().enumerate().skip(1) {
            row[idx] += lambda_l2;
        }

        let beta = solve_linear_system(xtwx, xtwy)?;
        Ok(LinearLeafModel {
            intercept: beta[0],
            coefficients: beta[1..].to_vec(),
            features,
        })
    }
}

impl LinearLeafModel {
    pub fn predict(&self, row: &[f64]) -> Result<f64> {
        let mut value = self.intercept;
        for (feature, coef) in self.features.iter().zip(&self.coefficients) {
            let Some(x) = row.get(*feature) else {
                return Err(GeoBoostError::InvalidInput(format!(
                    "feature index {feature} is out of bounds"
                )));
            };
            value += coef * x;
        }
        Ok(value)
    }
}

fn design_row(row: &[f64], features: &[usize]) -> Result<Vec<f64>> {
    let mut design = Vec::with_capacity(features.len() + 1);
    design.push(1.0);
    for feature in features {
        let Some(value) = row.get(*feature) else {
            return Err(GeoBoostError::InvalidInput(format!(
                "feature index {feature} is out of bounds"
            )));
        };
        if !value.is_finite() {
            return Err(GeoBoostError::InvalidInput(
                "linear leaf features must be finite".to_string(),
            ));
        }
        design.push(*value);
    }
    Ok(design)
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&lhs, &rhs| a[lhs][col].abs().total_cmp(&a[rhs][col].abs()))
            .expect("non-empty pivot range");
        if a[pivot][col].abs() < 1e-12 {
            a[col][col] += 1e-9;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let denom = a[col][col];
        if denom.abs() < 1e-15 {
            return Err(GeoBoostError::InvalidInput(
                "linear system is singular".to_string(),
            ));
        }
        let pivot_tail = a[col][col..].to_vec();
        for row in (col + 1)..n {
            let factor = a[row][col] / denom;
            for (cell, pivot) in a[row][col..].iter_mut().zip(&pivot_tail) {
                *cell -= factor * pivot;
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let rhs = b[row] - ((row + 1)..n).map(|col| a[row][col] * x[col]).sum::<f64>();
        x[row] = rhs / a[row][row];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-8,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn fits_noiseless_line() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let w = vec![1.0; 4];

        let model = LinearLeafPredictor::fit_ridge(&x, &y, &w, vec![0], 0.0).unwrap();

        assert_close(model.intercept, 3.0);
        assert_close(model.coefficients[0], 2.0);
        assert_close(model.predict(&[4.0]).unwrap(), 11.0);
    }

    #[test]
    fn ridge_shrinks_slope() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let w = vec![1.0; 4];

        let unregularized = LinearLeafPredictor::fit_ridge(&x, &y, &w, vec![0], 0.0).unwrap();
        let regularized = LinearLeafPredictor::fit_ridge(&x, &y, &w, vec![0], 100.0).unwrap();

        assert!(regularized.coefficients[0].abs() < unregularized.coefficients[0].abs());
    }
}
