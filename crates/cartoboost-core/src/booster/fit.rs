use crate::data::{validate_weights, Dataset};
use crate::loss::LossConfig;
use crate::objectives::{initial_margin_for_loss, pseudo_residual_for_loss};
use crate::profile;
use crate::tree::{
    FuzzyKernel, LeafPredictorKind, Model, PredictionTransform, SplitterKind,
    TrainingConfigMetadata, TreeBuilder, MODEL_ARTIFACT_VERSION,
};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
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
    pub constant_lambda_l2: f64,
    pub fuzzy: bool,
    pub fuzzy_bandwidth: f64,
    pub fuzzy_kernel: FuzzyKernel,
    pub loss: LossConfig,
    pub monotonic_constraints: Vec<i8>,
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
            splitters: vec![SplitterKind::Auto],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: LossConfig::L2,
            monotonic_constraints: Vec::new(),
        }
    }
}

impl Booster {
    pub fn new(config: BoosterConfig) -> Self {
        Self { config }
    }

    pub fn fit(&self, x: &Dataset, y: &[f64], sample_weight: Option<&[f64]>) -> Result<Model> {
        let profile_enabled = profile::enabled();
        if profile_enabled {
            profile::reset();
        }
        let profile_started = profile::ProfileTimer::start();
        if !self.config.learning_rate.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "learning_rate must be finite".to_string(),
            ));
        }
        if x.n_rows() != y.len() {
            return Err(CartoBoostError::InvalidInput(
                "X row count must match y length".to_string(),
            ));
        }
        if y.par_iter().any(|v| !v.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "targets must be finite".to_string(),
            ));
        }
        if let LossConfig::LogL2(config) = self.config.loss {
            if y.par_iter().any(|value| *value + config.offset <= 0.0) {
                return Err(CartoBoostError::InvalidInput(
                    "log_l2 targets must be greater than -offset".to_string(),
                ));
            }
        }
        let weights = validate_weights(sample_weight, y.len())?;
        validate_training_config(&self.config, x.n_cols())?;
        let resolved_splitters = resolve_splitters(&self.config, x);
        let transformed_y;
        let training_targets = match self.config.loss {
            LossConfig::LogL2(config) => {
                transformed_y = y
                    .par_iter()
                    .map(|value| (*value + config.offset).ln())
                    .collect::<Vec<_>>();
                &transformed_y
            }
            _ => y,
        };
        let init_prediction =
            initial_margin_for_loss(&self.config.loss, training_targets, Some(&weights))?;
        let prediction_transform = match self.config.loss {
            LossConfig::LogL2(_) => PredictionTransform::Expm1,
            _ => PredictionTransform::Identity,
        };
        let mut pred = vec![init_prediction; training_targets.len()];
        let mut residuals = vec![0.0; training_targets.len()];
        let mut trees = Vec::with_capacity(self.config.n_estimators);
        let builder = TreeBuilder {
            max_depth: self.config.max_depth,
            min_samples_leaf: self.config.min_samples_leaf,
            min_gain: self.config.min_gain,
            splitters: resolved_splitters.clone(),
            leaf_predictor: self.config.leaf_predictor.clone(),
            linear_leaf_features: self.config.linear_leaf_features.clone(),
            linear_lambda_l2: self.config.linear_lambda_l2,
            constant_lambda_l2: self.config.constant_lambda_l2,
            fuzzy: self.config.fuzzy,
            fuzzy_bandwidth: self.config.fuzzy_bandwidth,
            fuzzy_kernel: self.config.fuzzy_kernel,
            loss: self.config.loss.clone(),
            monotonic_constraints: self.config.monotonic_constraints.clone(),
        };
        let fit_context = profile::timed(profile::CONTEXT, || builder.fit_context(x));

        let tree_count = if self.config.max_depth == 0
            && matches!(self.config.leaf_predictor, LeafPredictorKind::Constant)
        {
            0
        } else {
            self.config.n_estimators
        };
        for _ in 0..tree_count {
            profile::timed(profile::RESIDUAL, || {
                residuals
                    .par_iter_mut()
                    .zip(training_targets.par_iter())
                    .zip(pred.par_iter())
                    .for_each(|((residual, target), prediction)| {
                        *residual =
                            pseudo_residual_for_loss(&self.config.loss, *target, *prediction);
                    });
            });
            let use_leaf_updates = !self.config.fuzzy
                && matches!(self.config.leaf_predictor, LeafPredictorKind::Constant);
            let use_leaf_updates = use_leaf_updates
                && matches!(self.config.loss, LossConfig::L2 | LossConfig::LogL2(_))
                && self.config.monotonic_constraints.is_empty();
            let tree = if use_leaf_updates {
                let (tree, updates) = profile::timed(profile::TREE_FIT, || {
                    builder.fit_with_leaf_updates_in_context(x, &residuals, &weights, &fit_context)
                });
                profile::timed(profile::PRED_UPDATE, || {
                    pred.par_iter_mut()
                        .zip(updates.par_iter())
                        .for_each(|(prediction, update)| {
                            *prediction += self.config.learning_rate * update;
                        });
                });
                tree
            } else {
                let tree = profile::timed(profile::TREE_FIT, || {
                    builder.fit_in_context(x, &residuals, &weights, &fit_context)
                });
                profile::timed(profile::PRED_UPDATE, || {
                    pred.par_iter_mut()
                        .enumerate()
                        .for_each(|(row, prediction)| {
                            *prediction +=
                                self.config.learning_rate * tree.predict_dataset_row(x, row);
                        });
                });
                tree
            };
            trees.push(tree);
        }

        let model = Model {
            artifact_version: MODEL_ARTIFACT_VERSION,
            metadata: Some(Model::default_metadata()),
            init_prediction,
            learning_rate: self.config.learning_rate,
            feature_count: x.n_cols(),
            feature_schema: Some(x.feature_schema_or_default()),
            target_name: None,
            training_config: Some(TrainingConfigMetadata {
                n_estimators: self.config.n_estimators,
                learning_rate: self.config.learning_rate,
                max_depth: self.config.max_depth,
                min_samples_leaf: self.config.min_samples_leaf,
                min_gain: self.config.min_gain,
                splitters: resolved_splitters,
                leaf_predictor: self.config.leaf_predictor.clone(),
                linear_leaf_features: self.config.linear_leaf_features.clone(),
                linear_lambda_l2: self.config.linear_lambda_l2,
                constant_lambda_l2: self.config.constant_lambda_l2,
                fuzzy: self.config.fuzzy,
                fuzzy_bandwidth: self.config.fuzzy_bandwidth,
                fuzzy_kernel: self.config.fuzzy_kernel,
                loss: self.config.loss.clone(),
                monotonic_constraints: self.config.monotonic_constraints.clone(),
            }),
            prediction_transform,
            trees,
        };
        profile::report("booster_fit", profile_started.elapsed());
        Ok(model)
    }
}

fn validate_training_config(config: &BoosterConfig, feature_count: usize) -> Result<()> {
    match config.loss {
        LossConfig::L2 => {}
        LossConfig::L1 => {
            if config.leaf_predictor != LeafPredictorKind::Constant {
                return Err(CartoBoostError::InvalidInput(
                    "l1 loss currently requires constant leaves".to_string(),
                ));
            }
        }
        LossConfig::Huber(loss) => {
            if !loss.delta.is_finite() || loss.delta <= 0.0 {
                return Err(CartoBoostError::InvalidInput(
                    "huber delta must be positive and finite".to_string(),
                ));
            }
            if config.leaf_predictor != LeafPredictorKind::Constant {
                return Err(CartoBoostError::InvalidInput(
                    "huber loss currently requires constant leaves".to_string(),
                ));
            }
        }
        LossConfig::LogL2(loss) => {
            if (loss.offset - 1.0).abs() > 1e-12 {
                return Err(CartoBoostError::InvalidInput(
                    "log_l2 currently supports offset=1.0".to_string(),
                ));
            }
            if config.leaf_predictor != LeafPredictorKind::Constant {
                return Err(CartoBoostError::InvalidInput(
                    "log_l2 loss currently requires constant leaves".to_string(),
                ));
            }
        }
        LossConfig::Quantile(loss) => {
            if !loss.alpha.is_finite() || loss.alpha <= 0.0 || loss.alpha >= 1.0 {
                return Err(CartoBoostError::InvalidInput(
                    "quantile alpha must be finite and in (0, 1)".to_string(),
                ));
            }
            if config.leaf_predictor != LeafPredictorKind::Constant {
                return Err(CartoBoostError::InvalidInput(
                    "quantile loss currently requires constant leaves".to_string(),
                ));
            }
        }
    }
    if !config.monotonic_constraints.is_empty() {
        if config.monotonic_constraints.len() != feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "monotonic_constraints has length {}, but X has {feature_count} features",
                config.monotonic_constraints.len()
            )));
        }
        if config
            .monotonic_constraints
            .iter()
            .any(|constraint| !matches!(*constraint, -1..=1))
        {
            return Err(CartoBoostError::InvalidInput(
                "monotonic_constraints values must be -1, 0, or 1".to_string(),
            ));
        }
        if config.leaf_predictor != LeafPredictorKind::Constant {
            return Err(CartoBoostError::InvalidInput(
                "monotonic constraints currently require constant leaves".to_string(),
            ));
        }
        if config.fuzzy {
            return Err(CartoBoostError::InvalidInput(
                "monotonic constraints currently require hard routing".to_string(),
            ));
        }
        if config.splitters.iter().any(|splitter| {
            !matches!(
                splitter,
                SplitterKind::Auto | SplitterKind::Axis | SplitterKind::AxisHistogram { .. }
            )
        }) {
            return Err(CartoBoostError::InvalidInput(
                "monotonic constraints currently support only axis splitters".to_string(),
            ));
        }
    }
    Ok(())
}

fn resolve_splitters(config: &BoosterConfig, x: &Dataset) -> Vec<SplitterKind> {
    let fallback = if should_use_histogram_auto(config, x) {
        SplitterKind::AxisHistogram { bins: 512 }
    } else {
        SplitterKind::Axis
    };
    let mut resolved = Vec::with_capacity(config.splitters.len().max(1));
    for splitter in &config.splitters {
        match splitter {
            SplitterKind::Auto => resolved.push(fallback.clone()),
            other => resolved.push(other.clone()),
        }
    }
    if resolved.is_empty() {
        resolved.push(fallback);
    }
    resolved
}

fn should_use_histogram_auto(config: &BoosterConfig, x: &Dataset) -> bool {
    x.n_rows() >= 2_048
        && !config.fuzzy
        && config.monotonic_constraints.is_empty()
        && matches!(config.loss, LossConfig::L2 | LossConfig::LogL2(_))
        && matches!(config.leaf_predictor, LeafPredictorKind::Constant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::{L2Loss, LossConfig, QuantileLossConfig};
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
        assert_predictions_close(&model.predict(&x), &y);
    }

    fn assert_predictions_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn auto_splitter_resolves_to_exact_axis_for_small_training_sets() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(
            model.training_config.as_ref().unwrap().splitters,
            vec![SplitterKind::Axis]
        );
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Axis { .. },
                ..
            }
        ));
    }

    #[test]
    fn auto_splitter_resolves_to_histogram_for_large_dense_l2_training_sets() {
        let rows = 2_048;
        let cols = 2;
        let values = (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                (row as f64 * 0.03 + col as f64).sin()
            })
            .collect::<Vec<_>>();
        let y = (0..rows)
            .map(|row| if row % 11 < 5 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let x = Dataset::from_flat(rows, cols, values).unwrap();
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(
            model.training_config.as_ref().unwrap().splitters,
            vec![SplitterKind::AxisHistogram { bins: 512 }]
        );
    }

    #[test]
    fn explicit_axis_splitter_preserves_exact_axis_metadata_on_large_training_sets() {
        let rows = 2_048;
        let x = Dataset::from_flat(rows, 1, (0..rows).map(|row| row as f64).collect::<Vec<_>>())
            .unwrap();
        let y = (0..rows)
            .map(|row| if row < rows / 2 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(
            model.training_config.as_ref().unwrap().splitters,
            vec![SplitterKind::Axis]
        );
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Axis { .. },
                ..
            }
        ));
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
    fn quantile_booster_uses_weighted_quantile_initial_prediction_and_metadata() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let weights = vec![1.0, 1.0, 10.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            loss: LossConfig::Quantile(QuantileLossConfig { alpha: 0.8 }),
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, Some(&weights)).unwrap();

        assert_eq!(model.init_prediction, 20.0);
        assert_eq!(
            model.training_config.as_ref().unwrap().loss,
            LossConfig::Quantile(QuantileLossConfig { alpha: 0.8 })
        );
        assert!(model
            .predict(&x)
            .iter()
            .all(|prediction| prediction.is_finite()));
    }

    #[test]
    fn l1_booster_uses_weighted_median_initial_prediction_and_predictions_are_finite() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let weights = vec![1.0, 1.0, 10.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 0,
            min_samples_leaf: 1,
            min_gain: 0.0,
            loss: LossConfig::L1,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, Some(&weights)).unwrap();

        assert_eq!(model.init_prediction, 20.0);
        assert_eq!(model.training_config.as_ref().unwrap().loss, LossConfig::L1);
        assert_eq!(model.predict(&x), vec![20.0; 4]);
    }

    #[test]
    fn quantile_zero_depth_json_round_trips_constant_prediction() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 0,
            min_samples_leaf: 1,
            min_gain: 0.0,
            loss: LossConfig::Quantile(QuantileLossConfig { alpha: 0.8 }),
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("quantile.json");
        model.save(&path).unwrap();
        let loaded = Model::load(&path).unwrap();

        assert_eq!(model.predict(&x), vec![30.0; 4]);
        assert_eq!(loaded.predict(&x), model.predict(&x));
    }

    #[test]
    fn monotonic_constraints_reject_decreasing_axis_split_for_increasing_feature() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![10.0, 10.0, 0.0, 0.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            monotonic_constraints: vec![1],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Leaf { .. }
        ));
        assert_eq!(
            model
                .training_config
                .as_ref()
                .unwrap()
                .monotonic_constraints,
            vec![1]
        );
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

        assert_predictions_close(&model.predict(&x), &y);
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

        assert_predictions_close(&model.predict(&x), &y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::PeriodicInterval { .. },
                ..
            }
        ));
    }

    #[test]
    fn periodic_splitter_learns_shifted_interval_from_observed_boundaries() {
        let x = Dataset::from_rows(vec![
            vec![0.0],
            vec![1.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
            vec![11.0],
            vec![14.0],
            vec![18.0],
            vec![21.0],
        ])
        .unwrap();
        let y = vec![-3.0, -3.0, 8.0, 8.0, 8.0, 8.0, -3.0, -3.0, -3.0, -3.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_predictions_close(&model.predict(&x), &y);
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
    fn sparse_list_boosting_updates_residuals_with_dataset_aware_prediction() {
        let dense = Dataset::from_rows(vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]]).unwrap();
        let x = dense
            .with_sparse_sets(vec![crate::data::SparseSetColumn::new(vec![
                vec![7, 11],
                vec![11],
                vec![3],
                vec![],
            ])])
            .unwrap();
        let y = vec![10.0, 10.0, 0.0, 0.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 2,
            learning_rate: 0.5,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let predictions = model.predict(&x);

        assert_predictions_close(&predictions, &[8.75, 8.75, 1.25, 1.25]);
        assert!(model.trees.iter().all(|tree| matches!(
            tree.root,
            crate::tree::Node::Branch {
                split: Split::SparseListContainsAny { .. },
                ..
            }
        )));
    }

    #[test]
    fn gaussian_splitter_learns_off_center_hotspot() {
        let x = Dataset::from_rows(vec![
            vec![4.8, -2.0],
            vec![5.0, -2.1],
            vec![5.2, -1.9],
            vec![5.1, -2.2],
            vec![-5.0, -5.0],
            vec![-4.0, 4.0],
            vec![0.0, 5.0],
            vec![5.0, 5.0],
            vec![-5.0, 0.0],
            vec![0.0, -5.0],
        ])
        .unwrap();
        let y = vec![12.0, 12.0, 12.0, 12.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Gaussian2D],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_predictions_close(&model.predict(&x), &y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Gaussian2D { .. },
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
    fn fuzzy_training_uses_fractional_prediction_and_preserves_mass() {
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
        assert_predictions_close(&model.predict(&x), &[1.25, 3.125, 6.875, 8.75]);
        assert_predictions_close(&[model.predict_one(&[1.5])], &[5.0]);
    }
}
