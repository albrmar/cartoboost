use super::{LeafPredictorKind, SplitterKind};
use crate::data::Dataset;
use crate::data::FeatureSchema;
use crate::predictors::LinearLeafModel;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MODEL_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub artifact_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ModelMetadata>,
    pub init_prediction: f64,
    pub learning_rate: f64,
    pub feature_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_schema: Option<FeatureSchema>,
    #[serde(default)]
    pub target_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_config: Option<TrainingConfigMetadata>,
    pub trees: Vec<Tree>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub library_name: String,
    pub library_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfigMetadata {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    pub splitters: Vec<SplitterKind>,
    pub leaf_predictor: LeafPredictorKind,
    pub linear_leaf_features: Vec<usize>,
    pub linear_lambda_l2: f64,
    pub fuzzy: bool,
    pub fuzzy_bandwidth: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub root: Node,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    Leaf {
        value: f64,
        sample_weight_sum: f64,
        training_loss: f64,
    },
    LinearLeaf {
        model: LinearLeafModel,
        sample_weight_sum: f64,
        training_loss: f64,
    },
    Branch {
        split: Split,
        left: Box<Node>,
        right: Box<Node>,
        gain: f64,
        sample_weight_sum: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Split {
    Axis {
        feature: usize,
        threshold: f64,
        missing_goes_left: bool,
    },
    Diagonal2D {
        x_feature: usize,
        y_feature: usize,
        normal_x: f64,
        normal_y: f64,
        threshold: f64,
        missing_goes_left: bool,
    },
    Gaussian2D {
        x_feature: usize,
        y_feature: usize,
        center_x: f64,
        center_y: f64,
        radius: f64,
        missing_goes_left: bool,
    },
    PeriodicInterval {
        feature: usize,
        period: f64,
        start: f64,
        end: f64,
        missing_goes_left: bool,
    },
    SparseSetContainsAny {
        feature: usize,
        ids: Vec<u64>,
        missing_goes_left: bool,
    },
    Fuzzy {
        base: Box<Split>,
        bandwidth: f64,
        kernel: FuzzyKernel,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BranchWeights {
    pub left: f64,
    pub right: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuzzyKernel {
    Linear,
}

impl Model {
    pub fn default_metadata() -> ModelMetadata {
        ModelMetadata {
            library_name: env!("CARGO_PKG_NAME").to_string(),
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            description: None,
        }
    }

    pub fn feature_schema_or_default(&self) -> FeatureSchema {
        self.feature_schema
            .clone()
            .unwrap_or_else(|| FeatureSchema::unnamed_numeric(self.feature_count))
    }

    pub fn predict_one(&self, row: &[f64]) -> f64 {
        self.init_prediction
            + self
                .trees
                .iter()
                .map(|tree| self.learning_rate * tree.predict_one(row))
                .sum::<f64>()
    }

    pub fn predict(&self, x: &Dataset) -> Vec<f64> {
        (0..x.n_rows())
            .map(|row| {
                let values = (0..x.n_cols())
                    .map(|col| x.get(row, col))
                    .collect::<Vec<_>>();
                self.predict_one(&values)
            })
            .collect()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialize::save_json(self, path)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model = crate::serialize::load_json(path)?;
        if model.artifact_version != MODEL_ARTIFACT_VERSION {
            return Err(crate::GeoBoostError::InvalidInput(format!(
                "unsupported model artifact version {}",
                model.artifact_version
            )));
        }
        Ok(model)
    }
}

impl Tree {
    pub fn predict_one(&self, row: &[f64]) -> f64 {
        self.root.predict_one(row)
    }
}

impl Node {
    pub fn predict_one(&self, row: &[f64]) -> f64 {
        match self {
            Node::Leaf { value, .. } => *value,
            Node::LinearLeaf { model, .. } => model.predict(row).unwrap_or(model.intercept),
            Node::Branch {
                split, left, right, ..
            } => {
                let weights = split.branch_weights(row);
                weights.left * left.predict_one(row) + weights.right * right.predict_one(row)
            }
        }
    }
}

impl Split {
    pub fn goes_left(&self, row: &[f64]) -> bool {
        let weights = self.branch_weights(row);
        weights.left >= weights.right
    }

    pub fn branch_weights(&self, row: &[f64]) -> BranchWeights {
        match self {
            Split::Fuzzy {
                base, bandwidth, ..
            } => base
                .signed_distance(row)
                .map(|distance| fuzzy_weights(distance, *bandwidth))
                .unwrap_or_else(|| base.hard_branch_weights(row)),
            _ => self.hard_branch_weights(row),
        }
    }

    pub fn hard_branch_weights(&self, row: &[f64]) -> BranchWeights {
        if self.hard_goes_left(row) {
            BranchWeights {
                left: 1.0,
                right: 0.0,
            }
        } else {
            BranchWeights {
                left: 0.0,
                right: 1.0,
            }
        }
    }

    pub fn signed_distance(&self, row: &[f64]) -> Option<f64> {
        match self {
            Split::Axis {
                feature, threshold, ..
            } => row
                .get(*feature)
                .filter(|value| value.is_finite())
                .map(|value| value - threshold),
            Split::Diagonal2D {
                x_feature,
                y_feature,
                normal_x,
                normal_y,
                threshold,
                ..
            } => {
                let x = *row.get(*x_feature)?;
                let y = *row.get(*y_feature)?;
                (x.is_finite() && y.is_finite()).then_some(normal_x * x + normal_y * y - threshold)
            }
            Split::Gaussian2D {
                x_feature,
                y_feature,
                center_x,
                center_y,
                radius,
                ..
            } => {
                let x = *row.get(*x_feature)?;
                let y = *row.get(*y_feature)?;
                (x.is_finite() && y.is_finite())
                    .then_some(((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt() - radius)
            }
            Split::PeriodicInterval {
                feature,
                period,
                start,
                end,
                ..
            } => {
                let value = *row.get(*feature)?;
                value
                    .is_finite()
                    .then_some(periodic_signed_distance(value, *period, *start, *end))
            }
            Split::SparseSetContainsAny { .. } | Split::Fuzzy { .. } => None,
        }
    }

    fn hard_goes_left(&self, row: &[f64]) -> bool {
        match self {
            Split::Axis {
                missing_goes_left, ..
            }
            | Split::Diagonal2D {
                missing_goes_left, ..
            }
            | Split::Gaussian2D {
                missing_goes_left, ..
            } => self
                .signed_distance(row)
                .map(|distance| distance <= 0.0)
                .unwrap_or(*missing_goes_left),
            Split::PeriodicInterval {
                feature,
                period,
                start,
                end,
                missing_goes_left,
            } => row
                .get(*feature)
                .filter(|value| value.is_finite())
                .map(|value| periodic_contains(*value, *period, *start, *end))
                .unwrap_or(*missing_goes_left),
            Split::SparseSetContainsAny {
                feature,
                ids,
                missing_goes_left,
            } => row
                .get(*feature)
                .filter(|value| value.is_finite())
                .map(|value| sparse_set_value_contains_any(*value, ids))
                .unwrap_or(*missing_goes_left),
            Split::Fuzzy { base, .. } => base.hard_goes_left(row),
        }
    }
}

pub fn sparse_set_value_contains_any(value: f64, ids: &[u64]) -> bool {
    if value < 0.0 || !value.is_finite() {
        return false;
    }
    let id = value as u64;
    value == id as f64 && ids.contains(&id)
}

pub fn periodic_contains(value: f64, period: f64, start: f64, end: f64) -> bool {
    if period <= 0.0 || !period.is_finite() {
        return false;
    }
    let value = normalize_periodic(value, period);
    let start = normalize_periodic(start, period);
    let end = normalize_periodic(end, period);
    if start <= end {
        value >= start && value <= end
    } else {
        value >= start || value <= end
    }
}

pub fn periodic_signed_distance(value: f64, period: f64, start: f64, end: f64) -> f64 {
    if period <= 0.0 || !period.is_finite() {
        return f64::NAN;
    }
    let value = normalize_periodic(value, period);
    let start = normalize_periodic(start, period);
    let end = normalize_periodic(end, period);
    let distance_to_start = circular_distance(value, start, period);
    let distance_to_end = circular_distance(value, end, period);
    let nearest_boundary = distance_to_start.min(distance_to_end);
    if periodic_contains(value, period, start, end) {
        -nearest_boundary
    } else {
        nearest_boundary
    }
}

fn circular_distance(a: f64, b: f64, period: f64) -> f64 {
    let direct = (normalize_periodic(a, period) - normalize_periodic(b, period)).abs();
    direct.min(period - direct)
}

pub fn normalize_periodic(value: f64, period: f64) -> f64 {
    value.rem_euclid(period)
}

pub fn fuzzy_weights(distance: f64, bandwidth: f64) -> BranchWeights {
    if bandwidth <= 0.0 || !bandwidth.is_finite() || distance.abs() >= bandwidth {
        return if distance <= 0.0 {
            BranchWeights {
                left: 1.0,
                right: 0.0,
            }
        } else {
            BranchWeights {
                left: 0.0,
                right: 1.0,
            }
        };
    }
    let left = (0.5 - distance / (2.0 * bandwidth)).clamp(0.0, 1.0);
    BranchWeights {
        left,
        right: 1.0 - left,
    }
}
