use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparseSetColumn {
    #[serde(default)]
    pub values: Vec<Vec<u64>>,
}

impl SparseSetColumn {
    pub fn new(values: Vec<Vec<u64>>) -> Self {
        let mut column = Self { values };
        column.normalize();
        column
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn row(&self, row: usize) -> Option<&[u64]> {
        self.values.get(row).map(Vec::as_slice)
    }

    pub fn contains(&self, row: usize, id: u64) -> bool {
        self.row(row).is_some_and(|values| values.contains(&id))
    }

    pub fn contains_any<I>(&self, row: usize, ids: I) -> bool
    where
        I: IntoIterator<Item = u64>,
    {
        let Some(values) = self.row(row) else {
            return false;
        };
        ids.into_iter().any(|id| values.contains(&id))
    }

    pub fn normalize(&mut self) {
        for values in &mut self.values {
            values.sort_unstable();
            values.dedup();
        }
    }
}
