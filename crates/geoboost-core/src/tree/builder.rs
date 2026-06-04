use super::{sse, Node, Split, Tree};
use crate::data::{Dataset, FeatureKind};
use crate::predictors::LinearLeafPredictor;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    left_weights: Option<Vec<f64>>,
    right_weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct PeriodicValueGroup {
    value: f64,
    count: usize,
    weight_sum: f64,
    weighted_target_sum: f64,
    weighted_target_square_sum: f64,
}

#[derive(Debug, Clone, Default)]
struct CandidateStats {
    count: usize,
    weight_sum: f64,
    weighted_target_sum: f64,
    weighted_target_square_sum: f64,
}

#[derive(Debug, Clone)]
struct FitContext {
    sorted_dense_rows: Vec<Option<Vec<usize>>>,
    histogram_bins: Option<usize>,
    histogram_features: Vec<Option<HistogramFeature>>,
}

#[derive(Debug, Clone)]
struct HistogramFeature {
    bin_count: usize,
    min_value: f64,
    max_value: f64,
    scale: f64,
    bins: Vec<Option<usize>>,
}

impl FitContext {
    fn new(x: &Dataset, splitters: &[SplitterKind]) -> Self {
        let needs_exact_axis_order = splitters
            .iter()
            .any(|splitter| matches!(splitter, SplitterKind::Axis));
        let histogram_bins = splitters.iter().find_map(|splitter| match splitter {
            SplitterKind::AxisHistogram { bins } => Some((*bins).clamp(2, 1024)),
            _ => None,
        });
        let sorted_dense_rows = if needs_exact_axis_order {
            (0..x.n_cols())
                .map(|feature| {
                    let mut rows = (0..x.n_rows())
                        .filter(|&row| x.get(row, feature).is_finite())
                        .collect::<Vec<_>>();
                    rows.sort_by(|&left, &right| {
                        x.get(left, feature)
                            .total_cmp(&x.get(right, feature))
                            .then(left.cmp(&right))
                    });
                    (!rows.is_empty()).then_some(rows)
                })
                .collect()
        } else {
            Vec::new()
        };
        let histogram_features = histogram_bins
            .map(|bins| {
                (0..x.n_cols())
                    .map(|feature| prebinned_histogram_feature(x, feature, bins))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            sorted_dense_rows,
            histogram_bins,
            histogram_features,
        }
    }

    fn sorted_rows(&self, feature: usize) -> Option<&[usize]> {
        self.sorted_dense_rows
            .get(feature)
            .and_then(Option::as_deref)
    }

    fn histogram_feature(&self, feature: usize, bins: usize) -> Option<&HistogramFeature> {
        (self.histogram_bins == Some(bins))
            .then(|| self.histogram_features.get(feature))
            .flatten()
            .and_then(Option::as_ref)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum SplitterKind {
    #[default]
    Axis,
    AxisHistogram {
        bins: usize,
    },
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
        let context = FitContext::new(x, &self.splitters);
        Tree {
            root: self.build_node_inner(x, target, weights, &indices, 0, &context, None),
        }
    }

    pub fn fit_with_leaf_updates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
    ) -> (Tree, Vec<f64>) {
        let indices = (0..x.n_rows()).collect::<Vec<_>>();
        let context = FitContext::new(x, &self.splitters);
        let mut updates = vec![0.0; x.n_rows()];
        let root = self.build_node_inner(
            x,
            target,
            weights,
            &indices,
            0,
            &context,
            Some(&mut updates),
        );
        (Tree { root }, updates)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_node_inner(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        depth: usize,
        context: &FitContext,
        updates: Option<&mut [f64]>,
    ) -> Node {
        let leaf = |updates: Option<&mut [f64]>| {
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
            if let Some(updates) = updates {
                for &idx in indices {
                    updates[idx] = value;
                }
            }
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
            return leaf(updates);
        }

        let Some(best) = self.best_split(x, target, weights, indices, context) else {
            return leaf(updates);
        };
        if best.gain < self.min_gain {
            return leaf(updates);
        }

        let BestSplit {
            split,
            gain,
            left,
            right,
            left_weights,
            right_weights,
        } = best;
        let sample_weight_sum = indices.iter().map(|&idx| weights[idx]).sum();
        let left_weight_values = left_weights.as_deref().unwrap_or(weights);
        let right_weight_values = right_weights.as_deref().unwrap_or(weights);
        let left_node;
        let right_node;
        if let Some(updates) = updates {
            left_node = self.build_node_inner(
                x,
                target,
                left_weight_values,
                &left,
                depth + 1,
                context,
                Some(updates),
            );
            right_node = self.build_node_inner(
                x,
                target,
                right_weight_values,
                &right,
                depth + 1,
                context,
                Some(updates),
            );
        } else {
            left_node = self.build_node_inner(
                x,
                target,
                left_weight_values,
                &left,
                depth + 1,
                context,
                None,
            );
            right_node = self.build_node_inner(
                x,
                target,
                right_weight_values,
                &right,
                depth + 1,
                context,
                None,
            );
        }
        Node::Branch {
            split,
            left: Box::new(left_node),
            right: Box::new(right_node),
            gain,
            sample_weight_sum,
        }
    }

    fn best_split(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        context: &FitContext,
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
                SplitterKind::Axis => self
                    .axis_candidates(x, target, weights, indices, parent_sse, context, &mut best),
                SplitterKind::AxisHistogram { bins } => self.axis_histogram_candidates(
                    x, target, weights, indices, parent_sse, bins, context, &mut best,
                ),
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

    #[allow(clippy::too_many_arguments)]
    fn axis_histogram_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        bins: usize,
        context: &FitContext,
        best: &mut Option<BestSplit>,
    ) {
        let bins = bins.clamp(2, 1024);
        for feature in 0..x.n_cols() {
            if !dense_feature_allows_axis(x, feature) {
                continue;
            }
            if let Some(histogram_feature) = context.histogram_feature(feature, bins) {
                self.axis_histogram_prebinned_candidates(
                    feature,
                    histogram_feature,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    best,
                );
                continue;
            }
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;
            for &idx in indices {
                let value = x.get(idx, feature);
                if value.is_finite() {
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
            }
            if !min_value.is_finite() || min_value >= max_value {
                continue;
            }

            let scale = bins as f64 / (max_value - min_value);
            let mut stats = vec![CandidateStats::default(); bins];
            for &idx in indices {
                let value = x.get(idx, feature);
                if !value.is_finite() {
                    continue;
                }
                let bin = (((value - min_value) * scale) as usize).min(bins - 1);
                stats[bin].add_row(idx, target, weights);
            }

            let total = stats
                .iter()
                .fold(CandidateStats::default(), |mut total, item| {
                    total.count += item.count;
                    total.weight_sum += item.weight_sum;
                    total.weighted_target_sum += item.weighted_target_sum;
                    total.weighted_target_square_sum += item.weighted_target_square_sum;
                    total
                });
            if total.count < self.min_samples_leaf * 2 {
                continue;
            }

            let mut left_stats = CandidateStats::default();
            for (split_bin, bin_stats) in stats.iter().enumerate().take(bins - 1) {
                left_stats.count += bin_stats.count;
                left_stats.weight_sum += bin_stats.weight_sum;
                left_stats.weighted_target_sum += bin_stats.weighted_target_sum;
                left_stats.weighted_target_square_sum += bin_stats.weighted_target_square_sum;
                let right_stats = total.minus(&left_stats);
                if left_stats.count < self.min_samples_leaf
                    || right_stats.count < self.min_samples_leaf
                {
                    continue;
                }
                let threshold = min_value + ((split_bin + 1) as f64 / scale);
                if threshold >= max_value {
                    continue;
                }
                let gain = parent_sse - left_stats.sse() - right_stats.sse();
                let split = Split::Axis {
                    feature,
                    threshold,
                    missing_goes_left: true,
                };
                if best
                    .as_ref()
                    .is_some_and(|old| !is_better_split(gain, &split, old))
                {
                    continue;
                }
                materialize_axis_split(feature, threshold, x, indices, best);
                if let Some(best) = best.as_mut() {
                    best.split = split;
                    best.gain = gain;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn axis_histogram_prebinned_candidates(
        &self,
        feature: usize,
        histogram_feature: &HistogramFeature,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        best: &mut Option<BestSplit>,
    ) {
        let bins = histogram_feature.bin_count;
        let mut stats = vec![CandidateStats::default(); bins];
        for &idx in indices {
            if let Some(bin) = histogram_feature.bins[idx] {
                stats[bin].add_row(idx, target, weights);
            }
        }

        let total = stats
            .iter()
            .fold(CandidateStats::default(), |mut total, item| {
                total.count += item.count;
                total.weight_sum += item.weight_sum;
                total.weighted_target_sum += item.weighted_target_sum;
                total.weighted_target_square_sum += item.weighted_target_square_sum;
                total
            });
        if total.count < self.min_samples_leaf * 2 {
            return;
        }

        let mut left_stats = CandidateStats::default();
        for (split_bin, bin_stats) in stats.iter().enumerate().take(bins - 1) {
            left_stats.count += bin_stats.count;
            left_stats.weight_sum += bin_stats.weight_sum;
            left_stats.weighted_target_sum += bin_stats.weighted_target_sum;
            left_stats.weighted_target_square_sum += bin_stats.weighted_target_square_sum;
            let right_stats = total.minus(&left_stats);
            if left_stats.count < self.min_samples_leaf || right_stats.count < self.min_samples_leaf
            {
                continue;
            }
            let threshold =
                histogram_feature.min_value + ((split_bin + 1) as f64 / histogram_feature.scale);
            if threshold >= histogram_feature.max_value {
                continue;
            }
            let gain = parent_sse - left_stats.sse() - right_stats.sse();
            let split = Split::Axis {
                feature,
                threshold,
                missing_goes_left: true,
            };
            if best
                .as_ref()
                .is_some_and(|old| !is_better_split(gain, &split, old))
            {
                continue;
            }

            let mut left = Vec::with_capacity(left_stats.count);
            let mut right = Vec::with_capacity(right_stats.count);
            for &idx in indices {
                if histogram_feature.bins[idx].is_some_and(|bin| bin <= split_bin) {
                    left.push(idx);
                } else {
                    right.push(idx);
                }
            }

            *best = Some(BestSplit {
                split,
                gain,
                left,
                right,
                left_weights: None,
                right_weights: None,
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn axis_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        context: &FitContext,
        best: &mut Option<BestSplit>,
    ) {
        if !self.fuzzy || self.fuzzy_bandwidth <= 0.0 {
            self.axis_candidates_prefix(x, target, weights, indices, parent_sse, context, best);
            return;
        }

        let active = active_row_mask(x.n_rows(), indices);
        for feature in 0..x.n_cols() {
            if !dense_feature_allows_axis(x, feature) {
                continue;
            }
            let Some(sorted_rows) = context.sorted_rows(feature) else {
                continue;
            };
            let pairs = sorted_rows
                .iter()
                .copied()
                .filter(|&idx| active[idx])
                .filter_map(|idx| {
                    let value = x.get(idx, feature);
                    value.is_finite().then_some((value, idx))
                })
                .collect::<Vec<_>>();

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

    #[allow(clippy::too_many_arguments)]
    fn axis_candidates_prefix(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        context: &FitContext,
        best: &mut Option<BestSplit>,
    ) {
        let active = active_row_mask(x.n_rows(), indices);
        for feature in 0..x.n_cols() {
            if !dense_feature_allows_axis(x, feature) {
                continue;
            }
            let Some(sorted_rows) = context.sorted_rows(feature) else {
                continue;
            };
            let pairs = sorted_rows
                .iter()
                .copied()
                .filter(|&idx| active[idx])
                .filter_map(|idx| {
                    let value = x.get(idx, feature);
                    value.is_finite().then_some((value, idx))
                })
                .collect::<Vec<_>>();
            if pairs.len() < self.min_samples_leaf * 2 {
                continue;
            }

            let mut total_weight = 0.0;
            let mut total_weighted_target = 0.0;
            let mut total_weighted_target_sq = 0.0;
            for &(_, idx) in &pairs {
                let weight = weights[idx];
                let value = target[idx];
                total_weight += weight;
                total_weighted_target += weight * value;
                total_weighted_target_sq += weight * value * value;
            }

            let mut left_weight = 0.0;
            let mut left_weighted_target = 0.0;
            let mut left_weighted_target_sq = 0.0;
            for split_idx in 0..pairs.len() - 1 {
                let (_, idx) = pairs[split_idx];
                let weight = weights[idx];
                let value = target[idx];
                left_weight += weight;
                left_weighted_target += weight * value;
                left_weighted_target_sq += weight * value * value;

                let (current, _) = pairs[split_idx];
                let (next, _) = pairs[split_idx + 1];
                if current == next {
                    continue;
                }

                let left_count = split_idx + 1;
                let right_count = pairs.len() - left_count;
                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    continue;
                }

                let right_weight = total_weight - left_weight;
                let right_weighted_target = total_weighted_target - left_weighted_target;
                let right_weighted_target_sq = total_weighted_target_sq - left_weighted_target_sq;
                let gain = parent_sse
                    - weighted_sse_from_sums(
                        left_weight,
                        left_weighted_target,
                        left_weighted_target_sq,
                    )
                    - weighted_sse_from_sums(
                        right_weight,
                        right_weighted_target,
                        right_weighted_target_sq,
                    );
                let split = Split::Axis {
                    feature,
                    threshold: (current + next) / 2.0,
                    missing_goes_left: true,
                };
                if best
                    .as_ref()
                    .is_some_and(|old| !is_better_split(gain, &split, old))
                {
                    continue;
                }

                let mut left = Vec::with_capacity(left_count);
                let mut right = Vec::with_capacity(right_count);
                for (position, &(_, row_idx)) in pairs.iter().enumerate() {
                    if position <= split_idx {
                        left.push(row_idx);
                    } else {
                        right.push(row_idx);
                    }
                }

                *best = Some(BestSplit {
                    split,
                    gain,
                    left,
                    right,
                    left_weights: None,
                    right_weights: None,
                });
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
            if !dense_feature_allows_spatial(x, x_feature) {
                continue;
            }
            for y_feature in (x_feature + 1)..x.n_cols() {
                if !dense_feature_allows_spatial(x, y_feature) {
                    continue;
                }
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
            if !dense_feature_allows_spatial(x, x_feature) {
                continue;
            }
            for y_feature in (x_feature + 1)..x.n_cols() {
                if !dense_feature_allows_spatial(x, y_feature) {
                    continue;
                }
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
        if !self.fuzzy || self.fuzzy_bandwidth <= 0.0 {
            self.periodic_candidates_grouped(x, target, weights, indices, parent_sse, period, best);
            return;
        }

        if period <= 0.0 || !period.is_finite() {
            return;
        }
        for feature in 0..x.n_cols() {
            let Some(feature_period) = periodic_period_for_feature(x, indices, feature, period)
            else {
                continue;
            };
            let mut values = indices
                .iter()
                .filter_map(|&idx| {
                    let value = x.get(idx, feature);
                    value
                        .is_finite()
                        .then_some(super::normalize_periodic(value, feature_period))
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
                let gap = (next - current).rem_euclid(feature_period);
                boundaries.push(super::normalize_periodic(
                    current + gap / 2.0,
                    feature_period,
                ));
            }

            for start_idx in 0..boundaries.len() {
                for end_idx in 0..boundaries.len() {
                    if start_idx == end_idx {
                        continue;
                    }
                    let split = Split::PeriodicInterval {
                        feature,
                        period: feature_period,
                        start: boundaries[start_idx],
                        end: boundaries[end_idx],
                        missing_goes_left: true,
                    };
                    self.consider_split(split, x, target, weights, indices, parent_sse, best);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn periodic_candidates_grouped(
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
            let Some(feature_period) = periodic_period_for_feature(x, indices, feature, period)
            else {
                continue;
            };

            let mut values = indices
                .iter()
                .filter_map(|&idx| {
                    let value = x.get(idx, feature);
                    value
                        .is_finite()
                        .then_some((super::normalize_periodic(value, feature_period), idx))
                })
                .collect::<Vec<_>>();
            values.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            if values.len() < self.min_samples_leaf * 2 {
                continue;
            }

            let mut groups: Vec<PeriodicValueGroup> = Vec::new();
            for (value, idx) in values {
                let weight = weights[idx];
                let target_value = target[idx];
                if let Some(group) = groups
                    .last_mut()
                    .filter(|group| (group.value - value).abs() < 1e-12)
                {
                    group.count += 1;
                    group.weight_sum += weight;
                    group.weighted_target_sum += weight * target_value;
                    group.weighted_target_square_sum += weight * target_value * target_value;
                } else {
                    groups.push(PeriodicValueGroup {
                        value,
                        count: 1,
                        weight_sum: weight,
                        weighted_target_sum: weight * target_value,
                        weighted_target_square_sum: weight * target_value * target_value,
                    });
                }
            }
            if groups.len() < 2 {
                continue;
            }

            let mut boundaries = groups.iter().map(|group| group.value).collect::<Vec<_>>();
            for idx in 0..groups.len() {
                let current = groups[idx].value;
                let next = groups[(idx + 1) % groups.len()].value;
                let gap = (next - current).rem_euclid(feature_period);
                boundaries.push(super::normalize_periodic(
                    current + gap / 2.0,
                    feature_period,
                ));
            }

            let total_count = groups.iter().map(|group| group.count).sum::<usize>();
            let total_weight = groups.iter().map(|group| group.weight_sum).sum::<f64>();
            let total_weighted_target = groups
                .iter()
                .map(|group| group.weighted_target_sum)
                .sum::<f64>();
            let total_weighted_target_sq = groups
                .iter()
                .map(|group| group.weighted_target_square_sum)
                .sum::<f64>();

            for &start in &boundaries {
                for &end in &boundaries {
                    if (start - end).abs() < 1e-12 {
                        continue;
                    }

                    let mut left_count = 0usize;
                    let mut left_weight = 0.0;
                    let mut left_weighted_target = 0.0;
                    let mut left_weighted_target_sq = 0.0;
                    for group in &groups {
                        if super::periodic_contains(group.value, feature_period, start, end) {
                            left_count += group.count;
                            left_weight += group.weight_sum;
                            left_weighted_target += group.weighted_target_sum;
                            left_weighted_target_sq += group.weighted_target_square_sum;
                        }
                    }

                    let right_count = total_count - left_count;
                    if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                        continue;
                    }

                    let right_weight = total_weight - left_weight;
                    let right_weighted_target = total_weighted_target - left_weighted_target;
                    let right_weighted_target_sq =
                        total_weighted_target_sq - left_weighted_target_sq;
                    let gain = parent_sse
                        - weighted_sse_from_sums(
                            left_weight,
                            left_weighted_target,
                            left_weighted_target_sq,
                        )
                        - weighted_sse_from_sums(
                            right_weight,
                            right_weighted_target,
                            right_weighted_target_sq,
                        );
                    let split = Split::PeriodicInterval {
                        feature,
                        period: feature_period,
                        start,
                        end,
                        missing_goes_left: true,
                    };
                    if best
                        .as_ref()
                        .is_some_and(|old| !is_better_split(gain, &split, old))
                    {
                        continue;
                    }

                    let mut left = Vec::with_capacity(left_count);
                    let mut right = Vec::with_capacity(right_count);
                    for &idx in indices {
                        if super::periodic_contains(x.get(idx, feature), feature_period, start, end)
                        {
                            left.push(idx);
                        } else {
                            right.push(idx);
                        }
                    }

                    *best = Some(BestSplit {
                        split,
                        gain,
                        left,
                        right,
                        left_weights: None,
                        right_weights: None,
                    });
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
        if !self.fuzzy || self.fuzzy_bandwidth <= 0.0 {
            self.sparse_set_candidates_grouped(x, target, weights, indices, best);
            return;
        }

        for sparse_feature in 0..x.n_sparse_sets() {
            if !sparse_feature_allows_sparse_set(x, sparse_feature) {
                continue;
            }
            let mut ids = Vec::new();
            for &idx in indices {
                if let Some(row_ids) = x.sparse_set_row(idx, sparse_feature) {
                    ids.extend_from_slice(row_ids);
                }
            }
            ids.sort_unstable();
            ids.dedup();
            for id in ids {
                let split = Split::SparseListContainsAny {
                    sparse_feature,
                    ids: vec![id],
                    missing_goes_left: false,
                };
                self.consider_split(split, x, target, weights, indices, parent_sse, best);
            }
        }

        for feature in 0..x.n_cols() {
            if !dense_feature_allows_sparse_set(x, feature) {
                continue;
            }
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

    fn sparse_set_candidates_grouped(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        let total = candidate_stats(indices.iter().copied(), target, weights);

        for sparse_feature in 0..x.n_sparse_sets() {
            if !sparse_feature_allows_sparse_set(x, sparse_feature) {
                continue;
            }
            let mut by_id = BTreeMap::<u64, CandidateStats>::new();
            for &idx in indices {
                if let Some(row_ids) = x.sparse_set_row(idx, sparse_feature) {
                    for &id in row_ids {
                        by_id.entry(id).or_default().add_row(idx, target, weights);
                    }
                }
            }
            for (id, stats) in by_id {
                let split = Split::SparseListContainsAny {
                    sparse_feature,
                    ids: vec![id],
                    missing_goes_left: false,
                };
                self.consider_grouped_binary_split(split, stats, &total, best);
                if best.as_ref().is_some_and(|current| {
                    matches!(
                        current.split,
                        Split::SparseListContainsAny {
                            sparse_feature: current_feature,
                            ref ids,
                            ..
                        } if current_feature == sparse_feature && ids == &[id]
                    )
                }) {
                    materialize_sparse_list_split(sparse_feature, id, x, indices, best);
                }
            }
        }

        for feature in 0..x.n_cols() {
            if !dense_feature_allows_sparse_set(x, feature) {
                continue;
            }
            let mut by_id = BTreeMap::<u64, CandidateStats>::new();
            for &idx in indices {
                let value = x.get(idx, feature);
                let id = value as u64;
                if value.is_finite() && value >= 0.0 && value == id as f64 {
                    by_id.entry(id).or_default().add_row(idx, target, weights);
                }
            }
            for (id, stats) in by_id {
                let split = Split::SparseSetContainsAny {
                    feature,
                    ids: vec![id],
                    missing_goes_left: false,
                };
                self.consider_grouped_binary_split(split, stats, &total, best);
                if best.as_ref().is_some_and(|current| {
                    matches!(
                        current.split,
                        Split::SparseSetContainsAny {
                            feature: current_feature,
                            ref ids,
                            ..
                        } if current_feature == feature && ids == &[id]
                    )
                }) {
                    materialize_dense_sparse_split(feature, id, x, indices, best);
                }
            }
        }
    }

    fn consider_grouped_binary_split(
        &self,
        split: Split,
        left_stats: CandidateStats,
        total_stats: &CandidateStats,
        best: &mut Option<BestSplit>,
    ) {
        let right_stats = total_stats.minus(&left_stats);
        if left_stats.count < self.min_samples_leaf || right_stats.count < self.min_samples_leaf {
            return;
        }
        let gain = total_stats.sse() - left_stats.sse() - right_stats.sse();
        if best
            .as_ref()
            .is_some_and(|old| !is_better_split(gain, &split, old))
        {
            return;
        }
        *best = Some(BestSplit {
            split,
            gain,
            left: Vec::with_capacity(left_stats.count),
            right: Vec::with_capacity(right_stats.count),
            left_weights: None,
            right_weights: None,
        });
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
            let branch_weights = scoring_split.branch_weights_dataset_row(x, idx);
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
                left_weights: Some(left_weights),
                right_weights: Some(right_weights),
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

fn active_row_mask(rows: usize, indices: &[usize]) -> Vec<bool> {
    let mut active = vec![false; rows];
    for &idx in indices {
        active[idx] = true;
    }
    active
}

fn prebinned_histogram_feature(
    x: &Dataset,
    feature: usize,
    bin_count: usize,
) -> Option<HistogramFeature> {
    if !dense_feature_allows_axis(x, feature) {
        return None;
    }

    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for row in 0..x.n_rows() {
        let value = x.get(row, feature);
        if value.is_finite() {
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }
    }
    if !min_value.is_finite() || min_value >= max_value {
        return None;
    }

    let scale = bin_count as f64 / (max_value - min_value);
    let bins = (0..x.n_rows())
        .map(|row| {
            let value = x.get(row, feature);
            value
                .is_finite()
                .then(|| (((value - min_value) * scale) as usize).min(bin_count - 1))
        })
        .collect();
    Some(HistogramFeature {
        bin_count,
        min_value,
        max_value,
        scale,
        bins,
    })
}

fn dense_feature_kind(x: &Dataset, feature: usize) -> Option<&FeatureKind> {
    x.feature_schema()
        .and_then(|schema| schema.kinds.get(feature))
}

fn sparse_feature_kind(x: &Dataset, sparse_feature: usize) -> Option<&FeatureKind> {
    x.feature_schema()
        .and_then(|schema| schema.kinds.get(x.n_cols() + sparse_feature))
}

fn dense_feature_allows_axis(x: &Dataset, feature: usize) -> bool {
    !matches!(dense_feature_kind(x, feature), Some(FeatureKind::SparseSet))
}

fn dense_feature_allows_spatial(x: &Dataset, feature: usize) -> bool {
    matches!(
        dense_feature_kind(x, feature),
        None | Some(FeatureKind::Numeric)
    )
}

fn dense_feature_allows_sparse_set(x: &Dataset, feature: usize) -> bool {
    match x.feature_schema() {
        Some(_) => matches!(dense_feature_kind(x, feature), Some(FeatureKind::SparseSet)),
        None => true,
    }
}

impl CandidateStats {
    fn add_row(&mut self, idx: usize, target: &[f64], weights: &[f64]) {
        let weight = weights[idx];
        let value = target[idx];
        self.count += 1;
        self.weight_sum += weight;
        self.weighted_target_sum += weight * value;
        self.weighted_target_square_sum += weight * value * value;
    }

    fn minus(&self, other: &Self) -> Self {
        Self {
            count: self.count - other.count,
            weight_sum: self.weight_sum - other.weight_sum,
            weighted_target_sum: self.weighted_target_sum - other.weighted_target_sum,
            weighted_target_square_sum: self.weighted_target_square_sum
                - other.weighted_target_square_sum,
        }
    }

    fn sse(&self) -> f64 {
        weighted_sse_from_sums(
            self.weight_sum,
            self.weighted_target_sum,
            self.weighted_target_square_sum,
        )
    }
}

fn candidate_stats(
    indices: impl Iterator<Item = usize>,
    target: &[f64],
    weights: &[f64],
) -> CandidateStats {
    let mut stats = CandidateStats::default();
    for idx in indices {
        stats.add_row(idx, target, weights);
    }
    stats
}

fn materialize_sparse_list_split(
    sparse_feature: usize,
    id: u64,
    x: &Dataset,
    indices: &[usize],
    best: &mut Option<BestSplit>,
) {
    let Some(best) = best.as_mut() else {
        return;
    };
    best.left.clear();
    best.right.clear();
    best.left_weights = None;
    best.right_weights = None;
    for &idx in indices {
        if x.sparse_set_contains_any(idx, sparse_feature, &[id]) {
            best.left.push(idx);
        } else {
            best.right.push(idx);
        }
    }
}

fn materialize_dense_sparse_split(
    feature: usize,
    id: u64,
    x: &Dataset,
    indices: &[usize],
    best: &mut Option<BestSplit>,
) {
    let Some(best) = best.as_mut() else {
        return;
    };
    best.left.clear();
    best.right.clear();
    best.left_weights = None;
    best.right_weights = None;
    for &idx in indices {
        if super::sparse_set_value_contains_any(x.get(idx, feature), &[id]) {
            best.left.push(idx);
        } else {
            best.right.push(idx);
        }
    }
}

fn materialize_axis_split(
    feature: usize,
    threshold: f64,
    x: &Dataset,
    indices: &[usize],
    best: &mut Option<BestSplit>,
) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &idx in indices {
        if x.get(idx, feature) <= threshold {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }
    *best = Some(BestSplit {
        split: Split::Axis {
            feature,
            threshold,
            missing_goes_left: true,
        },
        gain: 0.0,
        left,
        right,
        left_weights: None,
        right_weights: None,
    });
}

fn sparse_feature_allows_sparse_set(x: &Dataset, sparse_feature: usize) -> bool {
    match x.feature_schema() {
        Some(_) => matches!(
            sparse_feature_kind(x, sparse_feature),
            Some(FeatureKind::SparseSet)
        ),
        None => true,
    }
}

fn periodic_period_for_feature(
    x: &Dataset,
    indices: &[usize],
    feature: usize,
    requested_period: f64,
) -> Option<f64> {
    match x.feature_schema() {
        Some(_) => match dense_feature_kind(x, feature) {
            Some(FeatureKind::Periodic { period }) => Some(*period as f64),
            _ => None,
        },
        None => looks_like_periodic_feature(x, indices, feature, requested_period)
            .then_some(requested_period),
    }
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

fn weighted_sse_from_sums(weight_sum: f64, weighted_sum: f64, weighted_square_sum: f64) -> f64 {
    if weight_sum <= 0.0 {
        0.0
    } else {
        weighted_square_sum - (weighted_sum * weighted_sum / weight_sum)
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
    fn histogram_axis_splitter_fits_monotonic_stump() {
        let x = Dataset::from_rows(vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::AxisHistogram { bins: 4 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::Axis { .. },
                ..
            }
        ));
        assert!(tree.predict_dataset_row(&x, 0) < tree.predict_dataset_row(&x, 5));
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

    #[test]
    fn sparse_set_splitter_trains_on_list_valued_rows() {
        let dense = Dataset::from_rows(vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]]).unwrap();
        let x = dense
            .with_sparse_sets(vec![crate::data::SparseSetColumn::new(vec![
                vec![10, 20],
                vec![20, 30],
                vec![40],
                vec![],
            ])])
            .unwrap();
        let y = vec![7.0, 7.0, -2.0, -2.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert_eq!(
            (0..x.n_rows())
                .map(|row| tree.predict_dataset_row(&x, row))
                .collect::<Vec<_>>(),
            y
        );
        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::SparseListContainsAny { .. },
                ..
            }
        ));
    }

    #[test]
    fn schema_declared_periodic_feature_does_not_need_full_observed_cycle() {
        let x = Dataset::from_rows(vec![vec![7.0], vec![8.0], vec![9.0], vec![10.0]])
            .unwrap()
            .with_schema(crate::data::FeatureSchema {
                names: vec!["hour".to_string()],
                kinds: vec![FeatureKind::Periodic { period: 24 }],
            })
            .unwrap();
        let y = vec![3.0, 3.0, -1.0, -1.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert_eq!(
            (0..x.n_rows())
                .map(|row| tree.predict_dataset_row(&x, row))
                .collect::<Vec<_>>(),
            y
        );
        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::PeriodicInterval { .. },
                ..
            }
        ));
    }

    #[test]
    fn schema_present_periodic_splitter_ignores_non_periodic_columns() {
        let x = Dataset::from_rows(vec![
            vec![0.0, 7.0],
            vec![1.0, 7.0],
            vec![23.0, 7.0],
            vec![24.0, 7.0],
        ])
        .unwrap()
        .with_schema(crate::data::FeatureSchema {
            names: vec!["numeric_covering_period".to_string(), "hour".to_string()],
            kinds: vec![FeatureKind::Numeric, FeatureKind::Periodic { period: 24 }],
        })
        .unwrap();
        let y = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(tree.root, Node::Leaf { .. }));
    }

    #[test]
    fn schema_present_sparse_splitter_ignores_dense_scalar_id_columns() {
        let x = Dataset::mixed(
            vec![vec![7.0], vec![7.0], vec![2.0], vec![2.0]],
            vec![crate::data::SparseSetColumn::new(vec![
                vec![100],
                vec![101],
                vec![102],
                vec![103],
            ])],
            Some(crate::data::FeatureSchema {
                names: vec!["dense_id_like".to_string(), "route_cells".to_string()],
                kinds: vec![FeatureKind::Numeric, FeatureKind::SparseSet],
            }),
        )
        .unwrap();
        let y = vec![9.0, 9.0, -4.0, -4.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(tree.root, Node::Leaf { .. }));
    }
}
