use super::{FeatureKind, FeatureSchema, SparseSetColumn};
use crate::{GeoBoostError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
    #[serde(default)]
    sparse_sets: Vec<SparseSetColumn>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    schema: Option<FeatureSchema>,
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
        Ok(Self {
            rows,
            cols,
            values,
            sparse_sets: Vec::new(),
            schema: None,
        })
    }

    pub fn with_sparse_sets(mut self, sparse_sets: Vec<SparseSetColumn>) -> Result<Self> {
        for (idx, column) in sparse_sets.iter().enumerate() {
            if column.len() != self.rows {
                return Err(GeoBoostError::InvalidInput(format!(
                    "sparse set column {idx} has {} rows but dense matrix has {} rows",
                    column.len(),
                    self.rows
                )));
            }
        }
        self.sparse_sets = sparse_sets;
        Ok(self)
    }

    pub fn with_schema(mut self, schema: FeatureSchema) -> Result<Self> {
        schema.validate()?;
        let expected = self.cols + self.sparse_sets.len();
        if !schema.is_empty() && schema.len() != expected {
            return Err(GeoBoostError::InvalidInput(format!(
                "feature schema length {} does not match dataset feature count {expected}",
                schema.len()
            )));
        }
        self.schema = Some(schema);
        Ok(self)
    }

    pub fn mixed(
        dense_rows: Vec<Vec<f64>>,
        sparse_sets: Vec<SparseSetColumn>,
        schema: Option<FeatureSchema>,
    ) -> Result<Self> {
        let dataset = Self::from_rows(dense_rows)?.with_sparse_sets(sparse_sets)?;
        match schema {
            Some(schema) => dataset.with_schema(schema),
            None => Ok(dataset),
        }
    }

    pub fn n_rows(&self) -> usize {
        self.rows
    }

    pub fn n_cols(&self) -> usize {
        self.cols
    }

    pub fn n_sparse_sets(&self) -> usize {
        self.sparse_sets.len()
    }

    pub fn feature_schema(&self) -> Option<&FeatureSchema> {
        self.schema.as_ref()
    }

    pub fn feature_schema_or_default(&self) -> FeatureSchema {
        self.schema.clone().unwrap_or_else(|| {
            let mut names = (0..self.cols)
                .map(|idx| format!("feature_{idx}"))
                .collect::<Vec<_>>();
            let mut kinds = vec![FeatureKind::Numeric; self.cols];
            names.extend((0..self.sparse_sets.len()).map(|idx| format!("sparse_set_{idx}")));
            kinds.extend(vec![FeatureKind::SparseSet; self.sparse_sets.len()]);
            FeatureSchema { names, kinds }
        })
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }

    pub fn sparse_set_row(&self, row: usize, sparse_col: usize) -> Option<&[u64]> {
        self.sparse_sets
            .get(sparse_col)
            .and_then(|column| column.row(row))
    }

    pub fn sparse_set_contains_any(&self, row: usize, sparse_col: usize, ids: &[u64]) -> bool {
        self.sparse_sets
            .get(sparse_col)
            .is_some_and(|column| column.contains_any(row, ids.iter().copied()))
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
