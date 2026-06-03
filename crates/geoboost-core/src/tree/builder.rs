use super::{sse, Node, Split, Tree};
use crate::data::Dataset;
use crate::predictors::LinearLeafPredictor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct TreeBuilder {
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

#[derive(Debug, Clone)]
struct BestSplit {
    split: Split,
    gain: f64,
    left: Vec<usize>,
    right: Vec<usize>,
    left_weights: Vec<f64>,
    right_weights: Vec<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum SplitterKind {
    #[default]
    Axis,
    Diagonal2D,
    Gaussian2D,
    Periodic {
        period: f64,
    },
    SparseSet,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum LeafPredictorKind {
    #[default]
    Constant,
    Linear,
}

impl TreeBuilder {
    pub fn fit(&self, x: &Dataset, target: &[f64], weights: &[f64]) -> Tree {
        let indices = (0..x.n_rows()).collect::<Vec<_>>();
        Tree {
            root: self.build_node(x, target, weights, &indices, 0),
        }
    }

    fn build_node(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> Node {
        let leaf = || {
            let weight_sum: f64 = indices.iter().map(|&idx| weights[idx]).sum();
            let value = if weight_sum <= 0.0 {
                0.0
            } else {
                indices
                    .iter()
                    .map(|&idx| target[idx] * weights[idx])
                    .sum::<f64>()
                    / weight_sum
            };
            let training_loss = sse(target, weights, indices);
            match self.leaf_predictor {
                LeafPredictorKind::Constant => Node::Leaf {
                    value,
                    sample_weight_sum: weight_sum,
                    training_loss,
                },
                LeafPredictorKind::Linear => {
                    let features = if self.linear_leaf_features.is_empty() {
                        (0..x.n_cols()).collect()
                    } else {
                        self.linear_leaf_features.clone()
                    };
                    let rows = indices
                        .iter()
                        .map(|&idx| (0..x.n_cols()).map(|col| x.get(idx, col)).collect())
                        .collect::<Vec<Vec<f64>>>();
                    let leaf_targets = indices.iter().map(|&idx| target[idx]).collect::<Vec<_>>();
                    let leaf_weights = indices.iter().map(|&idx| weights[idx]).collect::<Vec<_>>();
                    LinearLeafPredictor::fit_ridge(
                        &rows,
                        &leaf_targets,
                        &leaf_weights,
                        features,
                        self.linear_lambda_l2,
                    )
                    .map(|model| Node::LinearLeaf {
                        model,
                        sample_weight_sum: weight_sum,
                        training_loss,
                    })
                    .unwrap_or(Node::Leaf {
                        value,
                        sample_weight_sum: weight_sum,
                        training_loss,
                    })
                }
            }
        };

        if depth >= self.max_depth || indices.len() < self.min_samples_leaf * 2 {
            return leaf();
        }

        let Some(best) = self.best_split(x, target, weights, indices) else {
            return leaf();
        };
        if best.gain < self.min_gain {
            return leaf();
        }

        let sample_weight_sum = indices.iter().map(|&idx| weights[idx]).sum();
        Node::Branch {
            split: best.split,
            left: Box::new(self.build_node(x, target, &best.left_weights, &best.left, depth + 1)),
            right: Box::new(self.build_node(
                x,
                target,
                &best.right_weights,
                &best.right,
                depth + 1,
            )),
            gain: best.gain,
            sample_weight_sum,
        }
    }

    fn best_split(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
    ) -> Option<BestSplit> {
        let parent_sse = sse(target, weights, indices);
        let mut best: Option<BestSplit> = None;

        let splitters = if self.splitters.is_empty() {
            vec![SplitterKind::Axis]
        } else {
            self.splitters.clone()
        };
        for splitter in splitters {
            match splitter {
                SplitterKind::Axis => {
                    self.axis_candidates(x, target, weights, indices, parent_sse, &mut best)
                }
                SplitterKind::Diagonal2D => {
                    self.diagonal_candidates(x, target, weights, indices, parent_sse, &mut best)
                }
                SplitterKind::Gaussian2D => {
                    self.gaussian_candidates(x, target, weights, indices, parent_sse, &mut best)
                }
                SplitterKind::Periodic { period } => self.periodic_candidates(
                    x, target, weights, indices, parent_sse, period, &mut best,
                ),
                SplitterKind::SparseSet => {
                    self.sparse_set_candidates(x, target, weights, indices, parent_sse, &mut best)
                }
            }
        }

        best
    }

    fn axis_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        best: &mut Option<BestSplit>,
    ) {
        for feature in 0..x.n_cols() {
            let mut pairs = indices
                .iter()
                .filter_map(|&idx| {
                    let value = x.get(idx, feature);
                    value.is_finite().then_some((value, idx))
                })
                .collect::<Vec<_>>();
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

            for window in pairs.windows(2) {
                let (a, _) = window[0];
                let (b, _) = window[1];
                if a == b {
                    continue;
                }
                let threshold = (a + b) / 2.0;
                let split = Split::Axis {
                    feature,
                    threshold,
                    missing_goes_left: true,
                };
                self.consider_split(split, x, target, weights, indices, parent_sse, best);
            }
        }
    }

    fn diagonal_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        best: &mut Option<BestSplit>,
    ) {
        if x.n_cols() < 2 {
            return;
        }
        let normals = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)];
        for x_feature in 0..x.n_cols() {
            for y_feature in (x_feature + 1)..x.n_cols() {
                for (normal_x, normal_y) in normals {
                    let mut pairs = indices
                        .iter()
                        .map(|&idx| {
                            (
                                normal_x * x.get(idx, x_feature) + normal_y * x.get(idx, y_feature),
                                idx,
                            )
                        })
                        .collect::<Vec<_>>();
                    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    for window in pairs.windows(2) {
                        let threshold = (window[0].0 + window[1].0) / 2.0;
                        if window[0].0 == window[1].0 {
                            continue;
                        }
                        let split = Split::Diagonal2D {
                            x_feature,
                            y_feature,
                            normal_x,
                            normal_y,
                            threshold,
                            missing_goes_left: true,
                        };
                        self.consider_split(split, x, target, weights, indices, parent_sse, best);
                    }
                }
            }
        }
    }

    fn gaussian_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        best: &mut Option<BestSplit>,
    ) {
        if x.n_cols() < 2 || indices.is_empty() {
            return;
        }
        for x_feature in 0..x.n_cols() {
            for y_feature in (x_feature + 1)..x.n_cols() {
                for (center_x, center_y) in
                    self.gaussian_centers(x, target, weights, indices, x_feature, y_feature)
                {
                    let mut distances = indices
                        .iter()
                        .map(|&idx| {
                            (
                                ((x.get(idx, x_feature) - center_x).powi(2)
                                    + (x.get(idx, y_feature) - center_y).powi(2))
                                .sqrt(),
                                idx,
                            )
                        })
                        .filter(|(distance, _)| distance.is_finite())
                        .collect::<Vec<_>>();
                    distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    for window in distances.windows(2) {
                        if window[0].0 == window[1].0 {
                            continue;
                        }
                        let radius = (window[0].0 + window[1].0) / 2.0;
                        let split = Split::Gaussian2D {
                            x_feature,
                            y_feature,
                            center_x,
                            center_y,
                            radius,
                            missing_goes_left: true,
                        };
                        self.consider_split(split, x, target, weights, indices, parent_sse, best);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gaussian_centers(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        x_feature: usize,
        y_feature: usize,
    ) -> Vec<(f64, f64)> {
        let mut centers = Vec::new();
        let mut push_center = |center_x: f64, center_y: f64| {
            if !center_x.is_finite() || !center_y.is_finite() {
                return;
            }
            if centers.iter().any(|&(old_x, old_y): &(f64, f64)| {
                (old_x - center_x).abs() < 1e-12 && (old_y - center_y).abs() < 1e-12
            }) {
                return;
            }
            centers.push((center_x, center_y));
        };

        let weighted_centroid = |selected: &[usize]| -> Option<(f64, f64)> {
            let weight_sum = selected.iter().map(|&idx| weights[idx]).sum::<f64>();
            if weight_sum <= 0.0 {
                return None;
            }
            let center_x = selected
                .iter()
                .map(|&idx| x.get(idx, x_feature) * weights[idx])
                .sum::<f64>()
                / weight_sum;
            let center_y = selected
                .iter()
                .map(|&idx| x.get(idx, y_feature) * weights[idx])
                .sum::<f64>()
                / weight_sum;
            Some((center_x, center_y))
        };

        if let Some((center_x, center_y)) = weighted_centroid(indices) {
            push_center(center_x, center_y);
        }

        let weight_sum = indices.iter().map(|&idx| weights[idx]).sum::<f64>();
        let target_mean = if weight_sum > 0.0 {
            indices
                .iter()
                .map(|&idx| target[idx] * weights[idx])
                .sum::<f64>()
                / weight_sum
        } else {
            0.0
        };
        let above_mean = indices
            .iter()
            .copied()
            .filter(|&idx| target[idx] >= target_mean)
            .collect::<Vec<_>>();
        let below_mean = indices
            .iter()
            .copied()
            .filter(|&idx| target[idx] < target_mean)
            .collect::<Vec<_>>();
        for selected in [&above_mean, &below_mean] {
            if selected.len() >= self.min_samples_leaf {
                if let Some((center_x, center_y)) = weighted_centroid(selected) {
                    push_center(center_x, center_y);
                }
            }
        }

        centers
    }

    #[allow(clippy::too_many_arguments)]
    fn periodic_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        period: f64,
        best: &mut Option<BestSplit>,
    ) {
        if period <= 0.0 || !period.is_finite() {
            return;
        }
        for feature in 0..x.n_cols() {
            if !looks_like_periodic_feature(x, indices, feature, period) {
                continue;
            }
            let mut values = indices
                .iter()
                .filter_map(|&idx| {
                    let value = x.get(idx, feature);
                    value
                        .is_finite()
                        .then_some(super::normalize_periodic(value, period))
                })
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            if values.len() < 2 {
                continue;
            }

            let mut boundaries = values.clone();
            for idx in 0..values.len() {
                let current = values[idx];
                let next = values[(idx + 1) % values.len()];
                let gap = (next - current).rem_euclid(period);
                boundaries.push(super::normalize_periodic(current + gap / 2.0, period));
            }

            for start_idx in 0..boundaries.len() {
                for end_idx in 0..boundaries.len() {
                    if start_idx == end_idx {
                        continue;
                    }
                    let split = Split::PeriodicInterval {
                        feature,
                        period,
                        start: boundaries[start_idx],
                        end: boundaries[end_idx],
                        missing_goes_left: true,
                    };
                    self.consider_split(split, x, target, weights, indices, parent_sse, best);
                }
            }
        }
    }

    fn sparse_set_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        best: &mut Option<BestSplit>,
    ) {
        for feature in 0..x.n_cols() {
            let mut ids = indices
                .iter()
                .filter_map(|&idx| {
                    let value = x.get(idx, feature);
                    let id = value as u64;
                    (value.is_finite() && value >= 0.0 && value == id as f64).then_some(id)
                })
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ids.dedup();
            for id in ids {
                let split = Split::SparseSetContainsAny {
                    feature,
                    ids: vec![id],
                    missing_goes_left: false,
                };
                self.consider_split(split, x, target, weights, indices, parent_sse, best);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn consider_split(
        &self,
        split: Split,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        best: &mut Option<BestSplit>,
    ) {
        let scoring_split = if self.fuzzy && self.fuzzy_bandwidth > 0.0 {
            Split::Fuzzy {
                base: Box::new(split),
                bandwidth: self.fuzzy_bandwidth,
                kernel: super::FuzzyKernel::Linear,
            }
        } else {
            split
        };
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut left_weights = vec![0.0; weights.len()];
        let mut right_weights = vec![0.0; weights.len()];
        for &idx in indices {
            let row = (0..x.n_cols())
                .map(|col| x.get(idx, col))
                .collect::<Vec<_>>();
            let branch_weights = scoring_split.branch_weights(&row);
            if branch_weights.left > 0.0 {
                left.push(idx);
                left_weights[idx] = weights[idx] * branch_weights.left;
            }
            if branch_weights.right > 0.0 {
                right.push(idx);
                right_weights[idx] = weights[idx] * branch_weights.right;
            }
        }
        if left.len() < self.min_samples_leaf || right.len() < self.min_samples_leaf {
            return;
        }
        let gain =
            parent_sse - sse(target, &left_weights, &left) - sse(target, &right_weights, &right);
        if best
            .as_ref()
            .is_none_or(|old| is_better_split(gain, &scoring_split, old))
        {
            *best = Some(BestSplit {
                split: scoring_split,
                gain,
                left_weights,
                right_weights,
                left,
                right,
            });
        }
    }
}

fn looks_like_periodic_feature(
    x: &Dataset,
    indices: &[usize],
    feature: usize,
    period: f64,
) -> bool {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut count = 0usize;
    for &idx in indices {
        let value = x.get(idx, feature);
        if !value.is_finite() {
            continue;
        }
        if value < 0.0 || value > period {
            return false;
        }
        min = min.min(value);
        max = max.max(value);
        count += 1;
    }
    count >= 2 && min <= period * 0.25 && max >= period * 0.75
}

fn is_better_split(gain: f64, split: &Split, old: &BestSplit) -> bool {
    if gain > old.gain + 1e-12 {
        return true;
    }
    if (gain - old.gain).abs() > 1e-12 {
        return false;
    }
    match (periodic_width(split), periodic_width(&old.split)) {
        (Some(width), Some(old_width)) => width < old_width - 1e-12,
        _ => false,
    }
}

fn periodic_width(split: &Split) -> Option<f64> {
    match split {
        Split::PeriodicInterval {
            period, start, end, ..
        } => Some((end - start).rem_euclid(*period)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn one_stump_finds_golden_axis_split_with_constant_leaves() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
        };

        let tree = builder.fit(&x, &y, &weights);

        match tree.root {
            Node::Branch {
                split, left, right, ..
            } => {
                match split {
                    Split::Axis {
                        feature, threshold, ..
                    } => {
                        assert_eq!(feature, 0);
                        assert_close(threshold, 1.5);
                    }
                    other => panic!("expected axis split, got {other:?}"),
                }
                match (*left, *right) {
                    (Node::Leaf { value: left, .. }, Node::Leaf { value: right, .. }) => {
                        assert_close(left, 0.0);
                        assert_close(right, 1.0);
                    }
                    other => panic!("expected constant leaves, got {other:?}"),
                }
            }
            other => panic!("expected branch root, got {other:?}"),
        }
    }

    #[test]
    fn fuzzy_training_uses_fractional_child_weights() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            fuzzy: true,
            fuzzy_bandwidth: 2.0,
        };

        let tree = builder.fit(&x, &y, &weights);

        match tree.root {
            Node::Branch {
                split, left, right, ..
            } => {
                assert!(matches!(split, Split::Fuzzy { .. }));
                match (*left, *right) {
                    (Node::Leaf { value: left, .. }, Node::Leaf { value: right, .. }) => {
                        assert!(left > 0.0 && left < 5.0, "left leaf was {left}");
                        assert!(right > 5.0 && right < 10.0, "right leaf was {right}");
                    }
                    other => panic!("expected constant leaves, got {other:?}"),
                }
            }
            other => panic!("expected branch root, got {other:?}"),
        }
    }
}
