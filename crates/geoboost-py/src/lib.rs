use geoboost_core::tree::{LeafPredictorKind, SplitterKind};
use geoboost_core::{Booster, BoosterConfig, Dataset, GeoBoostError, Model};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

#[pyclass(name = "GeoBoostRegressor")]
#[derive(Clone, Debug)]
struct NativeGeoBoostRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    model: Option<Model>,
}

#[pymethods]
impl NativeGeoBoostRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.05, max_depth=4, min_samples_leaf=20, min_gain=1e-8, splitters=None, leaf_predictor="constant", linear_leaf_features=None, l2_regularization=1.0, fuzzy=false, fuzzy_bandwidth=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
    ) -> PyResult<Self> {
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            fuzzy_bandwidth,
        )?;
        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            splitters: splitters.unwrap_or_else(|| vec!["axis".to_string()]),
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            model: None,
        })
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let dataset = dataset_from_rows(x)?;
        let config = BoosterConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            splitters: parse_splitters(&self.splitters),
            leaf_predictor: parse_leaf_predictor(&self.leaf_predictor),
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
        };
        self.model = Some(
            Booster::new(config)
                .fit(&dataset, &y, None)
                .map_err(to_py_value_error)?,
        );
        Ok(())
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GeoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_rows(x)?;
        Ok(model.predict(&dataset))
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GeoBoostRegressor is not fitted"))?;
        model.save(path).map_err(to_py_error)
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let model = Model::load(path).map_err(to_py_error)?;
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec!["axis".to_string()],
            leaf_predictor: "constant".to_string(),
            linear_leaf_features: Vec::new(),
            l2_regularization: 1.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            model: Some(model),
        })
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.n_estimators
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    #[getter]
    fn max_depth(&self) -> usize {
        self.max_depth
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    #[getter]
    fn min_gain(&self) -> f64 {
        self.min_gain
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    #[getter]
    fn feature_count(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| model.feature_count)
            .unwrap_or(0)
    }
}

fn parse_splitters(names: &[String]) -> Vec<SplitterKind> {
    let splitters = names
        .iter()
        .filter_map(|name| match name.as_str() {
            "axis" => Some(SplitterKind::Axis),
            "diagonal_2d" | "diagonal2d" => Some(SplitterKind::Diagonal2D),
            "gaussian_2d" | "gaussian2d" | "radial" => Some(SplitterKind::Gaussian2D),
            "periodic_time" | "periodic_24" => Some(SplitterKind::Periodic { period: 24.0 }),
            "sparse_set" | "sparse" => Some(SplitterKind::SparseSet),
            _ => None,
        })
        .collect::<Vec<_>>();
    if splitters.is_empty() {
        vec![SplitterKind::Axis]
    } else {
        splitters
    }
}

fn parse_leaf_predictor(name: &str) -> LeafPredictorKind {
    match name {
        "linear" => LeafPredictorKind::Linear,
        _ => LeafPredictorKind::Constant,
    }
}

fn validate_params(
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    l2_regularization: f64,
    fuzzy_bandwidth: f64,
) -> PyResult<()> {
    if n_estimators == 0 {
        return Err(PyValueError::new_err("n_estimators must be positive"));
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(PyValueError::new_err(
            "learning_rate must be positive and finite",
        ));
    }
    if max_depth == 0 {
        return Err(PyValueError::new_err("max_depth must be positive"));
    }
    if min_samples_leaf == 0 {
        return Err(PyValueError::new_err("min_samples_leaf must be positive"));
    }
    if !min_gain.is_finite() || min_gain < 0.0 {
        return Err(PyValueError::new_err(
            "min_gain must be finite and non-negative",
        ));
    }
    if !l2_regularization.is_finite() || l2_regularization < 0.0 {
        return Err(PyValueError::new_err(
            "l2_regularization must be finite and non-negative",
        ));
    }
    if !fuzzy_bandwidth.is_finite() || fuzzy_bandwidth < 0.0 {
        return Err(PyValueError::new_err(
            "fuzzy_bandwidth must be finite and non-negative",
        ));
    }
    Ok(())
}

fn dataset_from_rows(rows: Vec<Vec<f64>>) -> PyResult<Dataset> {
    if rows.is_empty() {
        return Err(PyValueError::new_err("X must not be empty"));
    }
    if rows[0].is_empty() {
        return Err(PyValueError::new_err(
            "X rows must contain at least one feature",
        ));
    }
    if rows
        .iter()
        .any(|row| row.iter().any(|value| !value.is_finite()))
    {
        return Err(PyValueError::new_err("X must contain only finite values"));
    }
    Dataset::from_rows(rows).map_err(to_py_value_error)
}

fn to_py_value_error(err: GeoBoostError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_error(err: GeoBoostError) -> PyErr {
    match err {
        GeoBoostError::Io(_) => PyIOError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeGeoBoostRegressor>()?;
    Ok(())
}
