use crate::{GeoBoostError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

#[derive(Debug, Copy, Clone)]
pub struct SampleRef<'a> {
    dataset: &'a Dataset,
    row: usize,
}

impl Dataset {
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self> {
        let rows_len = rows.len();
        let cols = rows.first().map_or(0, Vec::len);
        if rows_len == 0 {
            return Err(GeoBoostError::InvalidInput(
                "dataset must contain at least one row".to_string(),
            ));
        }
        if cols == 0 {
            return Err(GeoBoostError::InvalidInput(
                "dataset must contain at least one column".to_string(),
            ));
        }
        if rows.iter().any(|row| row.len() != cols) {
            return Err(GeoBoostError::InvalidInput(
                "all rows must have the same number of columns".to_string(),
            ));
        }
        let mut values = Vec::with_capacity(rows_len * cols);
        for row in rows {
            for value in row {
                values.push(value);
            }
        }
        Self::from_flat(rows_len, cols, values)
    }

    pub fn from_flat(rows: usize, cols: usize, values: Vec<f64>) -> Result<Self> {
        if rows == 0 {
            return Err(GeoBoostError::InvalidInput(
                "dataset must contain at least one row".to_string(),
            ));
        }
        if cols == 0 {
            return Err(GeoBoostError::InvalidInput(
                "dataset must contain at least one column".to_string(),
            ));
        }
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(GeoBoostError::InvalidInput(format!(
                "matrix shape {rows}x{cols} does not match {} values",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(GeoBoostError::InvalidInput(
                "dataset values must be finite".to_string(),
            ));
        }
        Ok(Self { rows, cols, values })
    }

    pub fn n_rows(&self) -> usize {
        self.rows
    }

    pub fn n_cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }

    pub fn row(&self, row: usize) -> SampleRef<'_> {
        SampleRef { dataset: self, row }
    }

    pub fn rows(&self) -> impl Iterator<Item = SampleRef<'_>> {
        (0..self.rows).map(|row| self.row(row))
    }
}

impl SampleRef<'_> {
    pub fn get(&self, col: usize) -> f64 {
        self.dataset.get(self.row, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_ragged_or_non_finite_rows() {
        assert!(Dataset::from_rows(vec![vec![1.0], vec![2.0, 3.0]]).is_err());
        assert!(Dataset::from_rows(vec![vec![f64::NAN]]).is_err());
    }
}
