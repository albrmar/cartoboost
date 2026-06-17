use crate::artifact::EmbeddingTable;
use crate::error::Result;
use crate::features::NeuralFeatureBlock;

pub trait NeuralEncoder: Send + Sync {
    fn name(&self) -> &str;
    fn output_dim(&self) -> usize;
    fn encode_ids(&self, ids: &[u64]) -> Result<NeuralFeatureBlock>;
}

#[derive(Clone)]
pub struct EmbeddingTableEncoder {
    name: String,
    table: EmbeddingTable,
}

impl EmbeddingTableEncoder {
    pub fn new(name: impl Into<String>, table: EmbeddingTable) -> Self {
        Self {
            name: name.into(),
            table,
        }
    }

    pub fn table(&self) -> &EmbeddingTable {
        &self.table
    }

    pub fn table_mut(&mut self) -> &mut EmbeddingTable {
        &mut self.table
    }
}

impl NeuralEncoder for EmbeddingTableEncoder {
    fn name(&self) -> &str {
        &self.name
    }

    fn output_dim(&self) -> usize {
        self.table.dim()
    }

    fn encode_ids(&self, ids: &[u64]) -> Result<NeuralFeatureBlock> {
        self.table.encode_ids(ids, self.name.clone())
    }
}
