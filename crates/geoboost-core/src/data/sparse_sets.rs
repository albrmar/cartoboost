use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparseSetColumn {
    pub values: Vec<Vec<u64>>,
}
