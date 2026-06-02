use crate::data::{validate_weights, Dataset};
use crate::loss::{L2Loss, Loss};
use crate::tree::{Model, SplitterKind, TreeBuilder, MODEL_ARTIFACT_VERSION};
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
        };

        for _ in 0..self.config.n_estimators {
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
}
