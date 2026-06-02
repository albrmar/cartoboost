use crate::data::{validate_weights, Dataset};
use crate::loss::{L2Loss, Loss};
use crate::tree::{LeafPredictorKind, Model, SplitterKind, TreeBuilder, MODEL_ARTIFACT_VERSION};
use crate::{GeoBoostError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoosterConfig {
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

#[derive(Debug, Clone)]
pub struct Booster {
    pub config: BoosterConfig,
}

impl Default for BoosterConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.05,
            max_depth: 4,
            min_samples_leaf: 20,
            min_gain: 1e-8,
            splitters: vec![SplitterKind::Axis],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
        }
    }
}

impl Booster {
    pub fn new(config: BoosterConfig) -> Self {
        Self { config }
    }

    pub fn fit(&self, x: &Dataset, y: &[f64], sample_weight: Option<&[f64]>) -> Result<Model> {
        if !self.config.learning_rate.is_finite() {
            return Err(GeoBoostError::InvalidInput(
                "learning_rate must be finite".to_string(),
            ));
        }
        if x.n_rows() != y.len() {
            return Err(GeoBoostError::InvalidInput(
                "X row count must match y length".to_string(),
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(GeoBoostError::InvalidInput(
                "targets must be finite".to_string(),
            ));
        }
        let weights = validate_weights(sample_weight, y.len())?;
        let loss = L2Loss;
        let init_prediction = loss.initial_prediction(y, Some(&weights));
        let mut pred = vec![init_prediction; y.len()];
        let mut trees = Vec::with_capacity(self.config.n_estimators);
        let builder = TreeBuilder {
            max_depth: self.config.max_depth,
            min_samples_leaf: self.config.min_samples_leaf,
            min_gain: self.config.min_gain,
            splitters: self.config.splitters.clone(),
            leaf_predictor: self.config.leaf_predictor.clone(),
            linear_leaf_features: self.config.linear_leaf_features.clone(),
            linear_lambda_l2: self.config.linear_lambda_l2,
            fuzzy: self.config.fuzzy,
            fuzzy_bandwidth: self.config.fuzzy_bandwidth,
        };

        for _ in 0..self.config.n_estimators {
            // For L2, the negative gradient is the residual. Each tree fits this
            // target, then shrinkage applies learning_rate to the fitted update.
            let residuals = y
                .iter()
                .zip(&pred)
                .map(|(target, prediction)| target - prediction)
                .collect::<Vec<_>>();
            let tree = builder.fit(x, &residuals, &weights);
            for (row, prediction) in pred.iter_mut().enumerate() {
                let values = (0..x.n_cols())
                    .map(|col| x.get(row, col))
                    .collect::<Vec<_>>();
                *prediction += self.config.learning_rate * tree.predict_one(&values);
            }
            trees.push(tree);
        }

        Ok(Model {
            artifact_version: MODEL_ARTIFACT_VERSION,
            init_prediction,
            learning_rate: self.config.learning_rate,
            feature_count: x.n_cols(),
            target_name: None,
            trees,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Split, MODEL_ARTIFACT_VERSION};

    #[test]
    fn one_tree_booster_predicts_training_stump_with_learning_rate_one() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(model.artifact_version, MODEL_ARTIFACT_VERSION);
        assert_eq!(model.predict(&x), y);
    }

    #[test]
    fn booster_reduces_l2_loss_and_json_round_trips_predictions() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 3,
            learning_rate: 0.5,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let loss = L2Loss;
        let initial = vec![model.init_prediction; y.len()];
        let predictions = model.predict(&x);

        assert!(loss.value(&y, &predictions) < loss.value(&y, &initial));

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("model.json");
        model.save(&path).unwrap();
        let loaded = Model::load(&path).unwrap();

        assert_eq!(loaded.artifact_version, MODEL_ARTIFACT_VERSION);
        assert_eq!(loaded.predict(&x), predictions);
    }

    #[test]
    fn diagonal_splitter_solves_diagonal_boundary_stump() {
        let x = Dataset::from_rows(vec![
            vec![-2.0, -1.0],
            vec![-1.0, -1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
        ])
        .unwrap();
        let y = vec![-10.0, -10.0, 10.0, 10.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Diagonal2D],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(model.predict(&x), y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Diagonal2D { .. },
                ..
            }
        ));
    }

    #[test]
    fn gaussian_splitter_solves_radial_stump() {
        let x = Dataset::from_rows(vec![
            vec![0.0, 0.0],
            vec![0.25, 0.25],
            vec![3.0, 0.0],
            vec![0.0, 3.0],
        ])
        .unwrap();
        let y = vec![10.0, 10.0, -10.0, -10.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Gaussian2D],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(model.predict(&x), y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Gaussian2D { .. },
                ..
            }
        ));
    }

    #[test]
    fn periodic_splitter_handles_late_night_wraparound_stump() {
        let x = Dataset::from_rows(vec![vec![23.0], vec![1.0], vec![12.0], vec![15.0]]).unwrap();
        let y = vec![5.0, 5.0, -5.0, -5.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(model.predict(&x), y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::PeriodicInterval { .. },
                ..
            }
        ));
    }

    #[test]
    fn sparse_set_splitter_finds_integer_id_membership() {
        let x = Dataset::from_rows(vec![vec![7.0], vec![7.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![25.0, 25.0, -5.0, -5.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(model.predict(&x), y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::SparseSetContainsAny { .. },
                ..
            }
        ));
    }

    #[test]
    fn linear_leaf_fits_gradient_residuals_with_learning_rate_shrinkage() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 0.5,
            max_depth: 0,
            min_samples_leaf: 1,
            min_gain: 0.0,
            leaf_predictor: LeafPredictorKind::Linear,
            linear_leaf_features: vec![0],
            linear_lambda_l2: 0.0,
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let pred = model.predict(&x);

        assert_eq!(model.init_prediction, 6.0);
        for (actual, expected) in pred.iter().zip([4.5, 5.5, 6.5, 7.5]) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            );
        }
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::LinearLeaf { .. }
        ));
    }

    #[test]
    fn fuzzy_training_wraps_learned_split_and_preserves_shrinkage() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 10.0, 10.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            fuzzy: true,
            fuzzy_bandwidth: 1.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Fuzzy { .. },
                ..
            }
        ));
        assert_eq!(model.predict_one(&[0.0]), 0.0);
        assert_eq!(model.predict_one(&[3.0]), 10.0);
        assert_eq!(model.predict_one(&[1.5]), 5.0);
    }
}
