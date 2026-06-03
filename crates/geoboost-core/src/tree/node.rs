use super::{LeafPredictorKind, SplitterKind};
use crate::data::Dataset;
use crate::data::FeatureSchema;
use crate::predictors::LinearLeafModel;
use crate::{GeoBoostError, Result};
use rayon::prelude::*;
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
    SparseListContainsAny {
        sparse_feature: usize,
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

    pub fn try_predict_one_dense(&self, row: &[f64]) -> Result<f64> {
        if self.requires_sparse_sets() {
            return Err(GeoBoostError::InvalidInput(
                "dense prediction is not available for models with list-valued sparse splits"
                    .to_string(),
            ));
        }
        Ok(self.predict_one(row))
    }

    pub fn predict_dataset_row(&self, x: &Dataset, row: usize) -> f64 {
        self.init_prediction
            + self
                .trees
                .iter()
                .map(|tree| self.learning_rate * tree.predict_dataset_row(x, row))
                .sum::<f64>()
    }

    pub fn try_predict_dataset_row(&self, x: &Dataset, row: usize) -> Result<f64> {
        if self.requires_sparse_sets() && x.n_sparse_sets() == 0 {
            return Err(GeoBoostError::InvalidInput(
                "prediction requires sparse_sets for a model with list-valued sparse splits"
                    .to_string(),
            ));
        }
        Ok(self.predict_dataset_row(x, row))
    }

    pub fn predict(&self, x: &Dataset) -> Vec<f64> {
        (0..x.n_rows())
            .map(|row| self.predict_dataset_row(x, row))
            .collect()
    }

    pub fn try_predict(&self, x: &Dataset) -> Result<Vec<f64>> {
        (0..x.n_rows())
            .map(|row| self.try_predict_dataset_row(x, row))
            .collect()
    }

    pub fn try_predict_flat(
        &self,
        rows: usize,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
    ) -> Result<Vec<f64>> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(GeoBoostError::InvalidInput(format!(
                "matrix shape {rows}x{cols} does not match {} values",
                values.len()
            )));
        }
        if cols != self.feature_count {
            return Err(GeoBoostError::InvalidInput(format!(
                "X has {cols} features, but model expects {}",
                self.feature_count
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(GeoBoostError::InvalidInput(
                "dataset values must be finite".to_string(),
            ));
        }
        if sparse_offsets.len() != sparse_ids.len() {
            return Err(GeoBoostError::InvalidInput(
                "sparse_offsets and sparse_ids must contain the same number of columns".to_string(),
            ));
        }
        if self.requires_sparse_sets() && sparse_offsets.is_empty() {
            return Err(GeoBoostError::InvalidInput(
                "prediction requires sparse_sets for a model with list-valued sparse splits"
                    .to_string(),
            ));
        }
        for (column, (offsets, ids)) in sparse_offsets.iter().zip(sparse_ids).enumerate() {
            if offsets.len() != rows + 1 {
                return Err(GeoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must have rows + 1 entries"
                )));
            }
            if offsets.first().copied() != Some(0) || offsets.last().copied() != Some(ids.len()) {
                return Err(GeoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must span sparse_ids exactly"
                )));
            }
            if offsets.windows(2).any(|window| window[0] > window[1]) {
                return Err(GeoBoostError::InvalidInput(format!(
                    "sparse_offsets column {column} must be non-decreasing"
                )));
            }
        }
        Ok((0..rows)
            .into_par_iter()
            .map(|row| self.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row))
            .collect())
    }

    fn predict_flat_row(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> f64 {
        self.init_prediction
            + self
                .trees
                .iter()
                .map(|tree| {
                    self.learning_rate
                        * tree.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row)
                })
                .sum::<f64>()
    }

    pub fn requires_sparse_sets(&self) -> bool {
        self.trees.iter().any(Tree::contains_sparse_list_split)
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

    pub fn predict_dataset_row(&self, x: &Dataset, row: usize) -> f64 {
        self.root.predict_dataset_row(x, row)
    }

    pub fn contains_sparse_list_split(&self) -> bool {
        self.root.contains_sparse_list_split()
    }

    fn predict_flat_row(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> f64 {
        self.root
            .predict_flat_row(cols, values, sparse_offsets, sparse_ids, row)
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
                weighted_child_prediction(
                    weights,
                    || left.predict_one(row),
                    || right.predict_one(row),
                )
            }
        }
    }

    pub fn predict_dataset_row(&self, x: &Dataset, row: usize) -> f64 {
        match self {
            Node::Leaf { value, .. } => *value,
            Node::LinearLeaf { model, .. } => {
                let dense = dense_row(x, row);
                model.predict(&dense).unwrap_or(model.intercept)
            }
            Node::Branch {
                split, left, right, ..
            } => {
                let weights = split.branch_weights_dataset_row(x, row);
                weighted_child_prediction(
                    weights,
                    || left.predict_dataset_row(x, row),
                    || right.predict_dataset_row(x, row),
                )
            }
        }
    }

    fn predict_flat_row(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> f64 {
        match self {
            Node::Leaf { value, .. } => *value,
            Node::LinearLeaf { model, .. } => {
                let dense = flat_dense_row(cols, values, row);
                model.predict(&dense).unwrap_or(model.intercept)
            }
            Node::Branch {
                split, left, right, ..
            } => {
                let weights =
                    split.branch_weights_flat(cols, values, sparse_offsets, sparse_ids, row);
                weighted_child_prediction(
                    weights,
                    || left.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row),
                    || right.predict_flat_row(cols, values, sparse_offsets, sparse_ids, row),
                )
            }
        }
    }

    pub fn contains_sparse_list_split(&self) -> bool {
        match self {
            Node::Leaf { .. } | Node::LinearLeaf { .. } => false,
            Node::Branch {
                split, left, right, ..
            } => {
                split.contains_sparse_list_split()
                    || left.contains_sparse_list_split()
                    || right.contains_sparse_list_split()
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

    pub fn branch_weights_dataset_row(&self, x: &Dataset, row: usize) -> BranchWeights {
        let dense = dense_row(x, row);
        match self {
            Split::Fuzzy {
                base, bandwidth, ..
            } => base
                .signed_distance_dataset_row(x, row, &dense)
                .map(|distance| fuzzy_weights(distance, *bandwidth))
                .unwrap_or_else(|| base.hard_branch_weights_dataset_row(x, row, &dense)),
            _ => self.hard_branch_weights_dataset_row(x, row, &dense),
        }
    }

    fn branch_weights_flat(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> BranchWeights {
        match self {
            Split::Fuzzy {
                base, bandwidth, ..
            } => base
                .signed_distance_flat(cols, values, row)
                .map(|distance| fuzzy_weights(distance, *bandwidth))
                .unwrap_or_else(|| {
                    base.hard_branch_weights_flat(cols, values, sparse_offsets, sparse_ids, row)
                }),
            _ => self.hard_branch_weights_flat(cols, values, sparse_offsets, sparse_ids, row),
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
            Split::SparseSetContainsAny { .. }
            | Split::SparseListContainsAny { .. }
            | Split::Fuzzy { .. } => None,
        }
    }

    fn signed_distance_dataset_row(
        &self,
        x: &Dataset,
        row_index: usize,
        dense_row: &[f64],
    ) -> Option<f64> {
        match self {
            Split::SparseListContainsAny { .. } => None,
            _ => {
                let _ = (x, row_index);
                self.signed_distance(dense_row)
            }
        }
    }

    fn hard_branch_weights_dataset_row(
        &self,
        x: &Dataset,
        row_index: usize,
        dense_row: &[f64],
    ) -> BranchWeights {
        let goes_left = match self {
            Split::SparseListContainsAny {
                sparse_feature,
                ids,
                missing_goes_left,
            } => {
                if ids.is_empty() {
                    *missing_goes_left
                } else {
                    x.sparse_set_contains_any(row_index, *sparse_feature, ids)
                }
            }
            _ => self.hard_goes_left(dense_row),
        };
        if goes_left {
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

    fn hard_branch_weights_flat(
        &self,
        cols: usize,
        values: &[f64],
        sparse_offsets: &[Vec<usize>],
        sparse_ids: &[Vec<u64>],
        row: usize,
    ) -> BranchWeights {
        let goes_left = match self {
            Split::SparseListContainsAny {
                sparse_feature,
                ids,
                missing_goes_left,
            } => {
                if ids.is_empty() {
                    *missing_goes_left
                } else {
                    encoded_sparse_contains_any(
                        sparse_offsets,
                        sparse_ids,
                        row,
                        *sparse_feature,
                        ids,
                    )
                }
            }
            _ => self.hard_goes_left_flat(cols, values, row),
        };
        if goes_left {
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
            Split::SparseListContainsAny {
                missing_goes_left, ..
            } => *missing_goes_left,
            Split::Fuzzy { base, .. } => base.hard_goes_left(row),
        }
    }

    fn hard_goes_left_flat(&self, cols: usize, values: &[f64], row: usize) -> bool {
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
                .signed_distance_flat(cols, values, row)
                .map(|distance| distance <= 0.0)
                .unwrap_or(*missing_goes_left),
            Split::PeriodicInterval {
                feature,
                period,
                start,
                end,
                missing_goes_left,
            } => flat_get(cols, values, row, *feature)
                .filter(|value| value.is_finite())
                .map(|value| periodic_contains(value, *period, *start, *end))
                .unwrap_or(*missing_goes_left),
            Split::SparseSetContainsAny {
                feature,
                ids,
                missing_goes_left,
            } => flat_get(cols, values, row, *feature)
                .filter(|value| value.is_finite())
                .map(|value| sparse_set_value_contains_any(value, ids))
                .unwrap_or(*missing_goes_left),
            Split::SparseListContainsAny {
                missing_goes_left, ..
            } => *missing_goes_left,
            Split::Fuzzy { base, .. } => base.hard_goes_left_flat(cols, values, row),
        }
    }

    fn signed_distance_flat(&self, cols: usize, values: &[f64], row: usize) -> Option<f64> {
        match self {
            Split::Axis {
                feature, threshold, ..
            } => flat_get(cols, values, row, *feature)
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
                let x = flat_get(cols, values, row, *x_feature)?;
                let y = flat_get(cols, values, row, *y_feature)?;
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
                let x = flat_get(cols, values, row, *x_feature)?;
                let y = flat_get(cols, values, row, *y_feature)?;
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
                let value = flat_get(cols, values, row, *feature)?;
                value
                    .is_finite()
                    .then_some(periodic_signed_distance(value, *period, *start, *end))
            }
            Split::SparseSetContainsAny { .. }
            | Split::SparseListContainsAny { .. }
            | Split::Fuzzy { .. } => None,
        }
    }

    pub fn contains_sparse_list_split(&self) -> bool {
        match self {
            Split::SparseListContainsAny { .. } => true,
            Split::Fuzzy { base, .. } => base.contains_sparse_list_split(),
            _ => false,
        }
    }
}

fn dense_row(x: &Dataset, row: usize) -> Vec<f64> {
    (0..x.n_cols()).map(|col| x.get(row, col)).collect()
}

fn weighted_child_prediction(
    weights: BranchWeights,
    left: impl FnOnce() -> f64,
    right: impl FnOnce() -> f64,
) -> f64 {
    if weights.left == 0.0 {
        weights.right * right()
    } else if weights.right == 0.0 {
        weights.left * left()
    } else {
        weights.left * left() + weights.right * right()
    }
}

fn flat_get(cols: usize, values: &[f64], row: usize, col: usize) -> Option<f64> {
    (col < cols)
        .then(|| values.get(row * cols + col).copied())
        .flatten()
}

fn flat_dense_row(cols: usize, values: &[f64], row: usize) -> Vec<f64> {
    let start = row * cols;
    values[start..start + cols].to_vec()
}

fn encoded_sparse_contains_any(
    sparse_offsets: &[Vec<usize>],
    sparse_ids: &[Vec<u64>],
    row: usize,
    sparse_col: usize,
    ids: &[u64],
) -> bool {
    let (Some(offsets), Some(values)) =
        (sparse_offsets.get(sparse_col), sparse_ids.get(sparse_col))
    else {
        return false;
    };
    let Some(window) = offsets.get(row..row + 2) else {
        return false;
    };
    values[window[0]..window[1]]
        .iter()
        .any(|value| ids.contains(value))
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
