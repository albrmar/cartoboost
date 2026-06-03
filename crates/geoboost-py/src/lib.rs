use geoboost_core::data::{FeatureSchema, SparseSetColumn};
use geoboost_core::tree::{LeafPredictorKind, SplitterKind};
use geoboost_core::{Booster, BoosterConfig, Dataset, GeoBoostError, Model};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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
        let splitters = splitters.unwrap_or_else(|| vec!["axis".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            model: None,
        })
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_sets=None, feature_schema_json=None))]
    fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_parts(x, sparse_sets, feature_schema_json)?;
        let splitters = parse_splitters(&self.splitters)?;
        let leaf_predictor = parse_leaf_predictor(&self.leaf_predictor)?;
        let config = BoosterConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            splitters,
            leaf_predictor,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
        };
        self.model = Some(
            Booster::new(config)
                .fit(&dataset, &y, sample_weight.as_deref())
                .map_err(to_py_value_error)?,
        );
        Ok(())
    }

    #[pyo3(signature = (x, y, sample_weight=None, sparse_offsets=None, sparse_ids=None, feature_schema_json=None))]
    fn fit_arrays(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        sample_weight: Option<PyReadonlyArray1<'_, f64>>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
    ) -> PyResult<()> {
        let dataset = dataset_from_arrays(x, sparse_offsets, sparse_ids, feature_schema_json)?;
        let targets = y.as_slice()?.to_vec();
        let weights = sample_weight
            .map(|array| array.as_slice().map(|slice| slice.to_vec()))
            .transpose()?;
        self.fit_dataset(dataset, targets, weights)
    }

    #[pyo3(signature = (x, y, sparse_sets, feature_schema_json=None, sample_weight=None))]
    fn fit_mixed(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        sparse_sets: Vec<Vec<Vec<u64>>>,
        feature_schema_json: Option<String>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.fit(x, y, sample_weight, Some(sparse_sets), feature_schema_json)
    }

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict(
        &self,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GeoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        model.try_predict(&dataset).map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_arrays<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GeoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        let predictions = model
            .try_predict_flat(rows, cols, values, &offsets, &ids)
            .map_err(to_py_value_error)?;
        Ok(predictions.into_pyarray(py))
    }

    #[pyo3(signature = (x, sparse_sets))]
    fn predict_mixed(
        &self,
        x: Vec<Vec<f64>>,
        sparse_sets: Vec<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<f64>> {
        self.predict(x, Some(sparse_sets))
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
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
            )
        } else {
            (
                1,
                1,
                0.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                false,
                0.0,
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
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
    fn splitters(&self) -> Vec<String> {
        self.splitters.clone()
    }

    #[getter]
    fn leaf_predictor(&self) -> String {
        self.leaf_predictor.clone()
    }

    #[getter]
    fn linear_leaf_features(&self) -> Vec<usize> {
        self.linear_leaf_features.clone()
    }

    #[getter]
    fn l2_regularization(&self) -> f64 {
        self.l2_regularization
    }

    #[getter]
    fn fuzzy(&self) -> bool {
        self.fuzzy
    }

    #[getter]
    fn fuzzy_bandwidth(&self) -> f64 {
        self.fuzzy_bandwidth
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

    #[getter]
    fn requires_sparse_sets(&self) -> bool {
        self.model
            .as_ref()
            .map(Model::requires_sparse_sets)
            .unwrap_or(false)
    }

    #[getter]
    fn feature_schema_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.feature_schema.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.metadata.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn training_config_json(&self) -> PyResult<Option<String>> {
        self.model
            .as_ref()
            .and_then(|model| model.training_config.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

impl NativeGeoBoostRegressor {
    fn fit_dataset(
        &mut self,
        dataset: Dataset,
        y: Vec<f64>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let splitters = parse_splitters(&self.splitters)?;
        let leaf_predictor = parse_leaf_predictor(&self.leaf_predictor)?;
        let config = BoosterConfig {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            min_gain: self.min_gain,
            splitters,
            leaf_predictor,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
        };
        self.model = Some(
            Booster::new(config)
                .fit(&dataset, &y, sample_weight.as_deref())
                .map_err(to_py_value_error)?,
        );
        Ok(())
    }
}

fn parse_splitters(names: &[String]) -> PyResult<Vec<SplitterKind>> {
    let mut splitters = Vec::with_capacity(names.len());
    for name in names {
        let splitter = match name.as_str() {
            "axis" => SplitterKind::Axis,
            "axis_histogram" | "axis_hist" | "histogram" => {
                SplitterKind::AxisHistogram { bins: 64 }
            }
            "diagonal_2d" | "diagonal2d" => SplitterKind::Diagonal2D,
            "gaussian_2d" | "gaussian2d" | "radial" => SplitterKind::Gaussian2D,
            "periodic_time" | "periodic_24" => SplitterKind::Periodic { period: 24.0 },
            "sparse_set" | "sparse" => SplitterKind::SparseSet,
            _ => {
                if let Some(bins) = name
                    .strip_prefix("axis_histogram:")
                    .or_else(|| name.strip_prefix("axis_hist:"))
                    .and_then(|bins| bins.parse::<usize>().ok())
                    .filter(|bins| *bins >= 2)
                {
                    SplitterKind::AxisHistogram { bins }
                } else if let Some(period) = name
                    .strip_prefix("periodic:")
                    .and_then(|period| period.parse::<f64>().ok())
                    .filter(|period| period.is_finite() && *period > 0.0)
                {
                    SplitterKind::Periodic { period }
                } else {
                    return Err(PyValueError::new_err(format!(
                        "unknown splitter {name:?}; expected one of 'axis', 'axis_histogram', \
                         'diagonal_2d', 'gaussian_2d', 'periodic_time', or 'sparse_set'"
                    )));
                }
            }
        };
        splitters.push(splitter);
    }
    if splitters.is_empty() {
        Ok(vec![SplitterKind::Axis])
    } else {
        Ok(splitters)
    }
}

fn parse_leaf_predictor(name: &str) -> PyResult<LeafPredictorKind> {
    match name {
        "constant" => Ok(LeafPredictorKind::Constant),
        "linear" => Ok(LeafPredictorKind::Linear),
        _ => Err(PyValueError::new_err(format!(
            "unknown leaf_predictor {name:?}; expected 'constant' or 'linear'"
        ))),
    }
}

fn splitter_names(splitters: &[SplitterKind]) -> Vec<String> {
    splitters
        .iter()
        .map(|splitter| match splitter {
            SplitterKind::Axis => "axis".to_string(),
            SplitterKind::AxisHistogram { bins } => format!("axis_histogram:{bins}"),
            SplitterKind::Diagonal2D => "diagonal_2d".to_string(),
            SplitterKind::Gaussian2D => "gaussian_2d".to_string(),
            SplitterKind::Periodic { period } if (*period - 24.0).abs() < 1e-12 => {
                "periodic_time".to_string()
            }
            SplitterKind::Periodic { period } => format!("periodic:{period}"),
            SplitterKind::SparseSet => "sparse_set".to_string(),
        })
        .collect()
}

fn leaf_predictor_name(leaf_predictor: &LeafPredictorKind) -> &'static str {
    match leaf_predictor {
        LeafPredictorKind::Constant => "constant",
        LeafPredictorKind::Linear => "linear",
    }
}

fn validate_params(
    n_estimators: usize,
    learning_rate: f64,
    _max_depth: usize,
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

fn dataset_from_parts(
    rows: Vec<Vec<f64>>,
    sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    feature_schema_json: Option<String>,
) -> PyResult<Dataset> {
    let dataset = dataset_from_rows(rows)?;
    let sparse_sets = sparse_sets
        .unwrap_or_default()
        .into_iter()
        .map(SparseSetColumn::new)
        .collect::<Vec<_>>();
    let schema = feature_schema_json
        .map(|payload| serde_json::from_str::<FeatureSchema>(&payload))
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema: {err}")))?;
    let dataset = dataset
        .with_sparse_sets(sparse_sets)
        .map_err(to_py_value_error)?;
    match schema {
        Some(schema) => dataset.with_schema(schema).map_err(to_py_value_error),
        None => Ok(dataset),
    }
}

fn dataset_from_arrays(
    x: PyReadonlyArray2<'_, f64>,
    sparse_offsets: Option<Vec<Vec<usize>>>,
    sparse_ids: Option<Vec<Vec<u64>>>,
    feature_schema_json: Option<String>,
) -> PyResult<Dataset> {
    let shape = x.shape();
    let rows = shape[0];
    let cols = shape[1];
    let values = x.as_slice()?.to_vec();
    let dataset = Dataset::from_flat(rows, cols, values).map_err(to_py_value_error)?;
    let sparse_sets = encoded_sparse_sets(rows, sparse_offsets, sparse_ids)?
        .into_iter()
        .map(SparseSetColumn::new)
        .collect::<Vec<_>>();
    let schema = feature_schema_json
        .map(|payload| serde_json::from_str::<FeatureSchema>(&payload))
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("invalid feature_schema: {err}")))?;
    let dataset = dataset
        .with_sparse_sets(sparse_sets)
        .map_err(to_py_value_error)?;
    match schema {
        Some(schema) => dataset.with_schema(schema).map_err(to_py_value_error),
        None => Ok(dataset),
    }
}

fn encoded_sparse_sets(
    rows: usize,
    sparse_offsets: Option<Vec<Vec<usize>>>,
    sparse_ids: Option<Vec<Vec<u64>>>,
) -> PyResult<Vec<Vec<Vec<u64>>>> {
    let offsets = sparse_offsets.unwrap_or_default();
    let ids = sparse_ids.unwrap_or_default();
    if offsets.len() != ids.len() {
        return Err(PyValueError::new_err(
            "sparse_offsets and sparse_ids must contain the same number of columns",
        ));
    }
    let mut columns = Vec::with_capacity(offsets.len());
    for (column_index, (column_offsets, column_ids)) in offsets.into_iter().zip(ids).enumerate() {
        if column_offsets.len() != rows + 1 {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must have rows + 1 entries"
            )));
        }
        if column_offsets.first().copied() != Some(0) {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must start at 0"
            )));
        }
        if column_offsets.last().copied() != Some(column_ids.len()) {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} final offset must match sparse_ids length"
            )));
        }
        if column_offsets
            .windows(2)
            .any(|window| window[0] > window[1])
        {
            return Err(PyValueError::new_err(format!(
                "sparse_offsets column {column_index} must be non-decreasing"
            )));
        }
        let mut column = Vec::with_capacity(rows);
        for window in column_offsets.windows(2) {
            column.push(column_ids[window[0]..window[1]].to_vec());
        }
        columns.push(column);
    }
    Ok(columns)
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
