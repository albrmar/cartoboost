use super::{sse, Node, Split, Tree};
use crate::data::Dataset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct TreeBuilder {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    pub splitters: Vec<SplitterKind>,
}

#[derive(Debug, Clone)]
struct BestSplit {
    split: Split,
    gain: f64,
    left: Vec<usize>,
    right: Vec<usize>,
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
            Node::Leaf {
                value,
                sample_weight_sum: weight_sum,
                training_loss: sse(target, weights, indices),
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
            left: Box::new(self.build_node(x, target, weights, &best.left, depth + 1)),
            right: Box::new(self.build_node(x, target, weights, &best.right, depth + 1)),
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
                let mut left = Vec::new();
                let mut right = Vec::new();
                for &idx in indices {
                    let value = x.get(idx, feature);
                    if value.is_nan() || value <= threshold {
                        left.push(idx);
                    } else {
                        right.push(idx);
                    }
                }
                if left.len() < self.min_samples_leaf || right.len() < self.min_samples_leaf {
                    continue;
                }
                let gain = parent_sse - sse(target, weights, &left) - sse(target, weights, &right);
                let replace = best.as_ref().is_none_or(|old| gain > old.gain);
                if replace {
                    *best = Some(BestSplit {
                        split: Split::Axis {
                            feature,
                            threshold,
                            missing_goes_left: true,
                        },
                        gain,
                        left,
                        right,
                    });
                }
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
                let center_x = indices
                    .iter()
                    .map(|&idx| x.get(idx, x_feature))
                    .sum::<f64>()
                    / indices.len() as f64;
                let center_y = indices
                    .iter()
                    .map(|&idx| x.get(idx, y_feature))
                    .sum::<f64>()
                    / indices.len() as f64;
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
        let starts = [0.0, 6.0, 7.0, 8.0, 16.0, 22.0]
            .into_iter()
            .filter(|value| *value < period)
            .collect::<Vec<_>>();
        let widths = [2.0, 3.0, 4.0, 8.0]
            .into_iter()
            .filter(|value| *value < period)
            .collect::<Vec<_>>();
        for feature in 0..x.n_cols() {
            for start in &starts {
                for width in &widths {
                    let split = Split::PeriodicInterval {
                        feature,
                        period,
                        start: *start,
                        end: (start + width).rem_euclid(period),
                        missing_goes_left: true,
                    };
                    self.consider_split(split, x, target, weights, indices, parent_sse, best);
                }
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
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &idx in indices {
            let row = (0..x.n_cols())
                .map(|col| x.get(idx, col))
                .collect::<Vec<_>>();
            if split.goes_left(&row) {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }
        if left.len() < self.min_samples_leaf || right.len() < self.min_samples_leaf {
            return;
        }
        let gain = parent_sse - sse(target, weights, &left) - sse(target, weights, &right);
        if best.as_ref().is_none_or(|old| gain > old.gain) {
            *best = Some(BestSplit {
                split,
                gain,
                left,
                right,
            });
        }
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
}
