use cartoboost_core::data::{Dataset, FeatureKind, FeatureSchema, SparseSetColumn};
use cartoboost_core::loss::{HuberLossConfig, LogL2LossConfig, LossConfig, QuantileLossConfig};
use cartoboost_core::tree::{FuzzyKernel, LeafPredictorKind, SplitterKind};
use cartoboost_core::{Booster, BoosterConfig};
use tempfile::tempdir;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-6,
        "expected {actual} to be within 1e-6 of {expected}"
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_close(*actual, *expected);
    }
}

fn axis_stump_config() -> BoosterConfig {
    BoosterConfig {
        n_estimators: 1,
        learning_rate: 1.0,
        max_depth: 1,
        min_samples_leaf: 1,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        ..BoosterConfig::default()
    }
}

#[test]
fn l2_axis_booster_fits_exact_two_leaf_step_function() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).expect("x");
    let y = vec![5.0, 5.0, -3.0, -3.0];

    let model = Booster::new(axis_stump_config())
        .fit(&x, &y, None)
        .expect("fit");
    let predictions = model.predict(&x);

    assert_eq!(predictions, y);
    assert_eq!(model.trees.len(), 1);
    assert_eq!(
        model.training_config.as_ref().expect("config").splitters,
        vec![SplitterKind::Axis]
    );
}

#[test]
fn weighted_l1_and_quantile_initial_predictions_are_exact_for_zero_depth_models() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).expect("x");
    let y = vec![0.0, 10.0, 20.0, 30.0];
    let weights = vec![1.0, 1.0, 10.0, 1.0];

    let l1 = Booster::new(BoosterConfig {
        n_estimators: 1,
        learning_rate: 1.0,
        max_depth: 0,
        min_samples_leaf: 1,
        loss: LossConfig::L1,
        ..BoosterConfig::default()
    })
    .fit(&x, &y, Some(&weights))
    .expect("fit l1");
    assert_eq!(l1.predict(&x), vec![20.0, 20.0, 20.0, 20.0]);

    let quantile = Booster::new(BoosterConfig {
        n_estimators: 1,
        learning_rate: 1.0,
        max_depth: 0,
        min_samples_leaf: 1,
        loss: LossConfig::Quantile(QuantileLossConfig::new(0.8)),
        ..BoosterConfig::default()
    })
    .fit(&x, &y, None)
    .expect("fit quantile");
    assert_eq!(quantile.init_prediction, 30.0);
    assert_eq!(quantile.predict(&x), vec![29.8, 29.8, 29.8, 29.8]);
}

#[test]
fn log_l2_predictions_are_transformed_back_to_original_target_scale() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).expect("x");
    let y = vec![1.0, 1.0, 7.0, 7.0];
    let model = Booster::new(BoosterConfig {
        n_estimators: 1,
        learning_rate: 1.0,
        max_depth: 1,
        min_samples_leaf: 1,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        loss: LossConfig::LogL2(LogL2LossConfig::new(1.0)),
        ..BoosterConfig::default()
    })
    .fit(&x, &y, None)
    .expect("fit log l2");

    let predictions = model.predict(&x);

    for (actual, expected) in predictions.iter().zip(y) {
        assert_close(*actual, expected);
    }
}

#[test]
fn huber_booster_caps_outlier_influence_and_stays_finite() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).expect("x");
    let y = vec![0.0, 0.0, 1_000.0, 1_000.0];
    let model = Booster::new(BoosterConfig {
        n_estimators: 1,
        learning_rate: 1.0,
        max_depth: 1,
        min_samples_leaf: 1,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        loss: LossConfig::Huber(HuberLossConfig::new(1.0)),
        ..BoosterConfig::default()
    })
    .fit(&x, &y, None)
    .expect("fit huber");
    let predictions = model.predict(&x);

    assert!(predictions.iter().all(|value| value.is_finite()));
    assert!(predictions[2] > predictions[0]);
    assert!(predictions[2] < 1_000.0);
}

#[test]
fn specialized_splitters_solve_their_native_geometry_fixtures() {
    let diagonal_x = Dataset::from_rows(vec![
        vec![-2.0, -1.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        vec![2.0, 1.0],
    ])
    .expect("diagonal x");
    let diagonal_y = vec![-10.0, -10.0, 10.0, 10.0];
    let diagonal_model = Booster::new(BoosterConfig {
        splitters: vec![SplitterKind::Diagonal2D],
        ..axis_stump_config()
    })
    .fit(&diagonal_x, &diagonal_y, None)
    .expect("fit diagonal");
    assert_eq!(diagonal_model.predict(&diagonal_x), diagonal_y);

    let gaussian_x = Dataset::from_rows(vec![
        vec![0.0, 0.0],
        vec![0.5, 0.5],
        vec![3.0, 3.0],
        vec![-3.0, -3.0],
    ])
    .expect("gaussian x");
    let gaussian_y = vec![8.0, 8.0, -8.0, -8.0];
    let gaussian_model = Booster::new(BoosterConfig {
        splitters: vec![SplitterKind::Gaussian2D],
        ..axis_stump_config()
    })
    .fit(&gaussian_x, &gaussian_y, None)
    .expect("fit gaussian");
    assert_eq!(gaussian_model.predict(&gaussian_x), gaussian_y);

    let periodic_x = Dataset::from_rows(vec![vec![23.0], vec![0.5], vec![11.0], vec![12.0]])
        .expect("periodic x");
    let periodic_y = vec![6.0, 6.0, -6.0, -6.0];
    let periodic_model = Booster::new(BoosterConfig {
        splitters: vec![SplitterKind::Periodic { period: 24.0 }],
        ..axis_stump_config()
    })
    .fit(&periodic_x, &periodic_y, None)
    .expect("fit periodic");
    assert_eq!(periodic_model.predict(&periodic_x), periodic_y);
}

#[test]
fn sparse_set_splitter_routes_list_valued_features_and_requires_sparse_prediction_input() {
    let dense = vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]];
    let sparse = SparseSetColumn::new(vec![vec![10, 20], vec![20, 30], vec![40], vec![]]);
    let schema = FeatureSchema {
        names: vec!["bias".to_string(), "route_cells".to_string()],
        kinds: vec![FeatureKind::Numeric, FeatureKind::SparseSet],
    };
    let x = Dataset::mixed(dense.clone(), vec![sparse], Some(schema)).expect("mixed dataset");
    let y = vec![7.0, 7.0, -2.0, -2.0];
    let model = Booster::new(BoosterConfig {
        splitters: vec![SplitterKind::SparseSet],
        ..axis_stump_config()
    })
    .fit(&x, &y, None)
    .expect("fit sparse");

    assert_eq!(model.try_predict(&x).expect("predict sparse"), y);
    let dense_only = Dataset::from_rows(dense).expect("dense only");
    assert!(model.try_predict(&dense_only).is_err());
}

#[test]
fn linear_leaf_and_fuzzy_boundary_produce_interpolated_known_answers() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).expect("x");
    let linear_model = Booster::new(BoosterConfig {
        n_estimators: 1,
        learning_rate: 0.5,
        max_depth: 1,
        min_samples_leaf: 3,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        leaf_predictor: LeafPredictorKind::Linear,
        linear_leaf_features: vec![0],
        linear_lambda_l2: 0.0,
        ..BoosterConfig::default()
    })
    .fit(&x, &[3.0, 5.0, 7.0, 9.0], None)
    .expect("fit linear leaf");
    assert_vec_close(&linear_model.predict(&x), &[4.5, 5.5, 6.5, 7.5]);

    let fuzzy_model = Booster::new(BoosterConfig {
        fuzzy: true,
        fuzzy_bandwidth: 1.0,
        fuzzy_kernel: FuzzyKernel::Linear,
        ..axis_stump_config()
    })
    .fit(&x, &[0.0, 0.0, 10.0, 10.0], None)
    .expect("fit fuzzy");
    assert_close(fuzzy_model.predict_one(&[1.5]), 5.0);
}

#[test]
fn monotonic_constraint_blocks_decreasing_fit_and_allows_increasing_fit() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).expect("x");
    let decreasing = Booster::new(BoosterConfig {
        monotonic_constraints: vec![1],
        ..axis_stump_config()
    })
    .fit(&x, &[10.0, 10.0, 0.0, 0.0], None)
    .expect("fit decreasing");
    let decreasing_pred = decreasing.predict(&x);
    assert!(decreasing_pred
        .windows(2)
        .all(|window| window[1] >= window[0] - 1.0e-9));

    let increasing = Booster::new(BoosterConfig {
        monotonic_constraints: vec![1],
        ..axis_stump_config()
    })
    .fit(&x, &[0.0, 0.0, 10.0, 10.0], None)
    .expect("fit increasing");
    let increasing_pred = increasing.predict(&x);
    assert!(increasing_pred
        .windows(2)
        .all(|window| window[1] >= window[0] - 1.0e-9));
    assert!(increasing_pred[3] > increasing_pred[0]);
}

#[test]
fn model_additive_contributions_and_json_roundtrip_preserve_predictions() {
    let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).expect("x");
    let y = vec![0.0, 0.0, 10.0, 10.0];
    let model = Booster::new(BoosterConfig {
        n_estimators: 2,
        learning_rate: 0.5,
        max_depth: 1,
        min_samples_leaf: 1,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        ..BoosterConfig::default()
    })
    .fit(&x, &y, None)
    .expect("fit");
    let before = model.try_predict(&x).expect("predict before");
    let additive = model.try_predict_additive(&x).expect("additive");
    assert_eq!(additive.len(), x.n_rows());
    assert!(additive
        .iter()
        .all(|row| row.len() == model.trees.len() + 1));

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("model.json");
    model.save(&path).expect("save");
    let restored = cartoboost_core::tree::Model::load(&path).expect("load");

    assert_eq!(restored.try_predict(&x).expect("predict after"), before);
    assert_eq!(restored.feature_count, 1);
}
