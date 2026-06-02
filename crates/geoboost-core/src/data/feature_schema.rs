use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureKind {
    Numeric,
    Periodic { period: u32 },
    SparseSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSchema {
    pub names: Vec<String>,
    pub kinds: Vec<FeatureKind>,
}

impl FeatureSchema {
    pub fn numeric(names: Vec<String>) -> Self {
        let kinds = vec![FeatureKind::Numeric; names.len()];
        Self { names, kinds }
    }

    pub fn empty() -> Self {
        Self {
            names: Vec::new(),
            kinds: Vec::new(),
        }
    }
}
