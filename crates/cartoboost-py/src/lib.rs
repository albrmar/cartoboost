use cartoboost_core::data::{FeatureSchema, SparseSetColumn};
use cartoboost_core::loss::{HuberLossConfig, LogL2LossConfig, LossConfig, QuantileLossConfig};
use cartoboost_core::tree::{FlatAxisPredictor, FuzzyKernel, LeafPredictorKind, SplitterKind};
use cartoboost_core::{Booster, BoosterConfig, CartoBoostError, Dataset, Model};
use cartoboost_neural::{
    build_embedding_table_artifact, compute_directional_features, fit_embedding_table,
    materialize_source_target_pair_nodes, validate_directed_metapath,
    write_embedding_table_artifact, ArtifactFallbackKind, EmbeddingTable, GraphSageConfig,
    GraphSageEncoder, HeteroGraph, HeteroGraphSageConfig, HeteroGraphSageEncoder, HeteroTypedEdge,
    HinSageConfig, HinSageEncoder, HinSageGraph, HomogeneousGraph, Node2VecConfig, Node2VecEncoder,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::path::PathBuf;

type StringTypedEdges = Vec<(String, String, String)>;

#[pyclass(name = "CartoBoostRegressor")]
#[derive(Clone, Debug)]
struct NativeCartoBoostRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    loss: String,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
    splitters: Vec<String>,
    leaf_predictor: String,
    linear_leaf_features: Vec<usize>,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy: bool,
    fuzzy_bandwidth: f64,
    fuzzy_kernel: String,
    monotonic_constraints: Vec<i8>,
    model: Option<Model>,
    flat_axis_predictor: Option<FlatAxisPredictor>,
}

#[pymethods]
impl NativeCartoBoostRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.05, max_depth=4, min_samples_leaf=20, min_gain=1e-8, loss="l2", quantile_alpha=0.5, huber_delta=1.0, log_offset=1.0, splitters=None, leaf_predictor="constant", linear_leaf_features=None, l2_regularization=1.0, constant_l2_regularization=0.0, fuzzy=false, fuzzy_bandwidth=0.0, fuzzy_kernel="linear", monotonic_constraints=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_gain: f64,
        loss: &str,
        quantile_alpha: f64,
        huber_delta: f64,
        log_offset: f64,
        splitters: Option<Vec<String>>,
        leaf_predictor: &str,
        linear_leaf_features: Option<Vec<usize>>,
        l2_regularization: f64,
        constant_l2_regularization: f64,
        fuzzy: bool,
        fuzzy_bandwidth: f64,
        fuzzy_kernel: &str,
        monotonic_constraints: Option<Vec<i8>>,
    ) -> PyResult<Self> {
        validate_params(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            l2_regularization,
            constant_l2_regularization,
            fuzzy_bandwidth,
            quantile_alpha,
            huber_delta,
            log_offset,
        )?;
        parse_loss(loss, quantile_alpha, huber_delta, log_offset)?;
        let splitters = splitters.unwrap_or_else(|| vec!["axis".to_string()]);
        parse_splitters(&splitters)?;
        parse_leaf_predictor(leaf_predictor)?;
        parse_fuzzy_kernel(fuzzy_kernel)?;

        Ok(Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            loss: loss.to_string(),
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor: leaf_predictor.to_string(),
            linear_leaf_features: linear_leaf_features.unwrap_or_default(),
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel: fuzzy_kernel.to_string(),
            monotonic_constraints: monotonic_constraints.unwrap_or_default(),
            model: None,
            flat_axis_predictor: None,
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
            loss: parse_loss(
                &self.loss,
                self.quantile_alpha,
                self.huber_delta,
                self.log_offset,
            )?,
            splitters,
            leaf_predictor,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            monotonic_constraints: self.monotonic_constraints.clone(),
        };
        self.set_model(
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
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
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
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        let predictions = if offsets.is_empty() && ids.is_empty() {
            if let Some(predictor) = &self.flat_axis_predictor {
                model
                    .validate_dense_flat_prediction_inputs(rows, cols, values)
                    .map_err(to_py_value_error)?;
                predictor.predict_flat(rows, cols, values)
            } else {
                model
                    .try_predict_flat(rows, cols, values, &offsets, &ids)
                    .map_err(to_py_value_error)?
            }
        } else {
            model
                .try_predict_flat(rows, cols, values, &offsets, &ids)
                .map_err(to_py_value_error)?
        };
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

    #[pyo3(signature = (x, sparse_sets=None))]
    fn predict_additive(
        &self,
        x: Vec<Vec<f64>>,
        sparse_sets: Option<Vec<Vec<Vec<u64>>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let dataset = dataset_from_parts(x, sparse_sets, None)?;
        model
            .try_predict_additive(&dataset)
            .map_err(to_py_value_error)
    }

    #[pyo3(signature = (x, sparse_offsets=None, sparse_ids=None))]
    fn predict_additive_arrays(
        &self,
        x: PyReadonlyArray2<'_, f64>,
        sparse_offsets: Option<Vec<Vec<usize>>>,
        sparse_ids: Option<Vec<Vec<u64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        let shape = x.shape();
        let rows = shape[0];
        let cols = shape[1];
        let values = x.as_slice()?;
        let offsets = sparse_offsets.unwrap_or_default();
        let ids = sparse_ids.unwrap_or_default();
        model
            .try_predict_additive_flat(rows, cols, values, &offsets, &ids)
            .map_err(to_py_value_error)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        model.save(path).map_err(to_py_error)
    }

    fn save_weights(&self, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("CartoBoostRegressor is not fitted"))?;
        model.save_weights(path).map_err(to_py_error)
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let model = Model::load(path).map_err(to_py_error)?;
        Self::from_model(model)
    }

    #[staticmethod]
    fn load_weights(path: PathBuf) -> PyResult<Self> {
        let model = Model::load_weights(path).map_err(to_py_error)?;
        Self::from_model(model)
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
    fn loss(&self) -> String {
        self.loss.clone()
    }

    #[getter]
    fn quantile_alpha(&self) -> f64 {
        self.quantile_alpha
    }

    #[getter]
    fn huber_delta(&self) -> f64 {
        self.huber_delta
    }

    #[getter]
    fn log_offset(&self) -> f64 {
        self.log_offset
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
    fn constant_l2_regularization(&self) -> f64 {
        self.constant_l2_regularization
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
    fn fuzzy_kernel(&self) -> String {
        self.fuzzy_kernel.clone()
    }

    #[getter]
    fn monotonic_constraints(&self) -> Vec<i8> {
        self.monotonic_constraints.clone()
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

impl NativeCartoBoostRegressor {
    fn from_model(model: Model) -> PyResult<Self> {
        let training_config = model.training_config.clone();
        let (
            max_depth,
            min_samples_leaf,
            min_gain,
            loss,
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            monotonic_constraints,
        ) = if let Some(config) = training_config {
            (
                config.max_depth,
                config.min_samples_leaf,
                config.min_gain,
                loss_name(&config.loss).to_string(),
                quantile_alpha(&config.loss),
                huber_delta(&config.loss),
                log_offset(&config.loss),
                splitter_names(&config.splitters),
                leaf_predictor_name(&config.leaf_predictor).to_string(),
                config.linear_leaf_features,
                config.linear_lambda_l2,
                config.constant_lambda_l2,
                config.fuzzy,
                config.fuzzy_bandwidth,
                fuzzy_kernel_name(config.fuzzy_kernel).to_string(),
                config.monotonic_constraints,
            )
        } else {
            (
                1,
                1,
                0.0,
                "l2".to_string(),
                0.5,
                1.0,
                1.0,
                vec!["axis".to_string()],
                "constant".to_string(),
                Vec::new(),
                1.0,
                0.0,
                false,
                0.0,
                "linear".to_string(),
                Vec::new(),
            )
        };
        Ok(Self {
            n_estimators: model.trees.len(),
            learning_rate: model.learning_rate,
            max_depth,
            min_samples_leaf,
            min_gain,
            loss,
            quantile_alpha,
            huber_delta,
            log_offset,
            splitters,
            leaf_predictor,
            linear_leaf_features,
            l2_regularization,
            constant_l2_regularization,
            fuzzy,
            fuzzy_bandwidth,
            fuzzy_kernel,
            monotonic_constraints,
            model: Some(model),
            flat_axis_predictor: None,
        })
        .map(|mut regressor| {
            regressor.refresh_prediction_cache();
            regressor
        })
    }

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
            loss: parse_loss(
                &self.loss,
                self.quantile_alpha,
                self.huber_delta,
                self.log_offset,
            )?,
            splitters,
            leaf_predictor,
            linear_leaf_features: self.linear_leaf_features.clone(),
            linear_lambda_l2: self.l2_regularization,
            constant_lambda_l2: self.constant_l2_regularization,
            fuzzy: self.fuzzy,
            fuzzy_bandwidth: self.fuzzy_bandwidth,
            fuzzy_kernel: parse_fuzzy_kernel(&self.fuzzy_kernel)?,
            monotonic_constraints: self.monotonic_constraints.clone(),
        };
        self.set_model(
            Booster::new(config)
                .fit(&dataset, &y, sample_weight.as_deref())
                .map_err(to_py_value_error)?,
        );
        Ok(())
    }

    fn set_model(&mut self, model: Model) {
        self.model = Some(model);
        self.refresh_prediction_cache();
    }

    fn refresh_prediction_cache(&mut self) {
        self.flat_axis_predictor = self.model.as_ref().and_then(Model::flat_axis_predictor);
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

#[pyclass(name = "NeuralEmbeddingFeatures")]
#[derive(Clone)]
struct NativeNeuralEmbeddingFeatures {
    dim: usize,
    fallback: ArtifactFallbackKind,
    random_state: Option<i64>,
    parent_resolution: Option<u8>,
    table: Option<EmbeddingTable>,
}

#[pymethods]
impl NativeNeuralEmbeddingFeatures {
    #[new]
    #[pyo3(signature = (dim, fallback="global_mean_vector", random_state=None, parent_resolution=None))]
    fn new(
        dim: usize,
        fallback: &str,
        random_state: Option<i64>,
        parent_resolution: Option<u8>,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be positive"));
        }

        let fallback = parse_embedding_fallback(fallback, parent_resolution)?;

        Ok(Self {
            dim,
            fallback,
            random_state,
            parent_resolution,
            table: None,
        })
    }

    #[pyo3(signature = (ids, target))]
    fn fit(
        &mut self,
        ids: PyReadonlyArray1<'_, u64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let ids = ids.as_slice()?;
        let target: Vec<f32> = target
            .as_slice()?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect();
        let random_state = self.random_state.map(|value| value as u64);

        let table =
            fit_embedding_table(self.dim, ids, &target, self.fallback.clone(), random_state)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.table = Some(table);
        Ok(())
    }

    #[pyo3(signature = (ids, target))]
    fn fit_transform(
        &mut self,
        ids: PyReadonlyArray1<'_, u64>,
        target: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let ids = ids.as_slice()?;
        let target: Vec<f32> = target
            .as_slice()?
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect();
        let random_state = self.random_state.map(|value| value as u64);
        let table =
            fit_embedding_table(self.dim, ids, &target, self.fallback.clone(), random_state)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.table = Some(table);

        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("fit_transform failed to set table"))?;
        let block = table
            .encode_ids(ids, "neural_embedding")
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut output = Vec::with_capacity(ids.len());
        for row in block.values.chunks_exact(block.dim) {
            output.push(row.to_vec());
        }
        Ok(output)
    }

    #[pyo3(signature = (ids))]
    fn transform(&self, ids: PyReadonlyArray1<'_, u64>) -> PyResult<Vec<Vec<f32>>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("transform called before fit or load"))?;

        let ids = ids.as_slice()?;
        let block = table
            .encode_ids(ids, "neural_embedding")
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut output = Vec::with_capacity(ids.len());
        for row in block.values.chunks_exact(block.dim) {
            output.push(row.to_vec());
        }
        Ok(output)
    }

    #[pyo3(signature = (path))]
    fn export(&self, path: String) -> PyResult<()> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("export called before fit or load"))?;

        let artifact = build_embedding_table_artifact(
            self.dim,
            table.rows().to_vec(),
            table.artifact_metadata().fallback.clone(),
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
        write_embedding_table_artifact(path, &artifact)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[classmethod]
    fn from_artifact(_cls: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let table =
            EmbeddingTable::load(path).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let metadata = table.artifact_metadata().clone();
        let parent_resolution = match metadata.fallback {
            ArtifactFallbackKind::ParentCell { parent_resolution } => Some(parent_resolution),
            _ => None,
        };

        Ok(Self {
            dim: metadata.dim,
            fallback: metadata.fallback,
            random_state: None,
            parent_resolution,
            table: Some(table),
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn fallback(&self) -> String {
        artifact_fallback_name(&self.fallback).to_string()
    }

    #[getter]
    fn random_state(&self) -> Option<i64> {
        self.random_state
    }

    #[getter]
    fn parent_resolution(&self) -> Option<u8> {
        self.parent_resolution
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.table.is_some()
    }

    fn artifact_rows(&self) -> PyResult<Vec<(u64, Vec<f32>)>> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("artifact_rows called before fit or load"))?;
        Ok(table
            .rows()
            .iter()
            .map(|row| (row.id, row.values.clone()))
            .collect())
    }
}

fn parse_embedding_fallback(
    value: &str,
    parent_resolution: Option<u8>,
) -> PyResult<ArtifactFallbackKind> {
    match value {
        "zero_vector" => Ok(ArtifactFallbackKind::ZeroVector),
        "global_mean_vector" => Ok(ArtifactFallbackKind::GlobalMeanVector),
        "parent_cell" => parent_resolution
            .map(|parent_resolution| ArtifactFallbackKind::ParentCell { parent_resolution })
            .ok_or_else(|| PyValueError::new_err("parent_resolution is required for parent_cell")),
        _ => Err(PyValueError::new_err(
            "fallback must be one of zero_vector, global_mean_vector, parent_cell",
        )),
    }
}

#[pyclass(name = "GraphSageEncoder")]
#[derive(Clone)]
struct NativeGraphSageEncoder {
    config: GraphSageConfig,
    encoder: GraphSageEncoder,
}

#[pymethods]
impl NativeGraphSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x5A17_9A4E_7F33_C0DE, add_self_loop=true, l2_regularization=1e-5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        add_self_loop: bool,
        l2_regularization: f32,
    ) -> PyResult<Self> {
        let config = GraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            add_self_loop,
            l2_regularization,
        };

        let encoder =
            GraphSageEncoder::new(config.clone(), input_dim).map_err(to_py_neural_error)?;

        Ok(Self { config, encoder })
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn fit(
        &mut self,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let graph = HomogeneousGraph::from_directed_edges(node_count, &edges)
            .map_err(to_py_neural_error)?;
        let mut model = GraphSageEncoder::new(self.config.clone(), self.encoder.input_dim())
            .map_err(to_py_neural_error)?;
        let embedding = model
            .fit(&graph, &node_features)
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = self
            .encoder
            .encode(&node_features)
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn encode_graph(
        &self,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let graph = HomogeneousGraph::from_directed_edges(node_count, &edges)
            .map_err(to_py_neural_error)?;
        let embedding = self
            .encoder
            .encode_graph(&graph, &node_features)
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, path: String) -> PyResult<()> {
        self.encoder
            .save_artifact_json(path)
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self) -> PyResult<String> {
        self.encoder.to_artifact_json().map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(_cls: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let encoder = GraphSageEncoder::load_artifact_json(path).map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self { encoder, config })
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }

    #[getter]
    fn config_seed(&self) -> u64 {
        self.config.seed
    }

    #[getter]
    fn config_epochs(&self) -> usize {
        self.config.epochs
    }

    #[getter]
    fn config_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    #[getter]
    fn config_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    #[getter]
    fn config_add_self_loop(&self) -> bool {
        self.config.add_self_loop
    }

    #[getter]
    fn config_l2_regularization(&self) -> f32 {
        self.config.l2_regularization
    }

    #[getter]
    fn hidden_dims(&self) -> Vec<usize> {
        self.config.hidden_dims.clone()
    }
}

#[pyclass(name = "Node2VecEncoder")]
#[derive(Clone)]
struct NativeNode2VecEncoder {
    config: Node2VecConfig,
    encoder: Node2VecEncoder,
}

#[pymethods]
impl NativeNode2VecEncoder {
    #[new]
    #[pyo3(signature = (dim=16, walk_length=16, walks_per_node=8, window_size=5, epochs=3, learning_rate=0.025, min_learning_rate=0.0001, negative_samples=5, p=1.0, q=1.0, seed=0xA2B2_C2D2_E2F2_1234, l2_regularization=0.0, normalize=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        walk_length: usize,
        walks_per_node: usize,
        window_size: usize,
        epochs: usize,
        learning_rate: f32,
        min_learning_rate: f32,
        negative_samples: usize,
        p: f32,
        q: f32,
        seed: u64,
        l2_regularization: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let config = Node2VecConfig {
            dim,
            walk_length,
            walks_per_node,
            window_size,
            epochs,
            learning_rate,
            min_learning_rate,
            negative_samples,
            p,
            q,
            seed,
            l2_regularization,
            normalize,
        };
        let encoder = Node2VecEncoder::new(config.clone()).map_err(to_py_neural_error)?;
        Ok(Self { config, encoder })
    }

    #[pyo3(signature = (node_count, edges, edge_weights=None))]
    fn fit(
        &mut self,
        node_count: usize,
        edges: Vec<(usize, usize)>,
        edge_weights: Option<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let mut model = Node2VecEncoder::new(self.config.clone()).map_err(to_py_neural_error)?;
        let embedding = model
            .fit(node_count, &edges, edge_weights.as_deref())
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self) -> PyResult<Vec<Vec<f32>>> {
        let embedding = self.encoder.encode().map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, path: String) -> PyResult<()> {
        self.encoder
            .save_artifact_json(path)
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self) -> PyResult<String> {
        self.encoder.to_artifact_json().map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(_cls: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let encoder = Node2VecEncoder::load_artifact_json(path).map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self { encoder, config })
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.encoder.node_count()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }

    #[getter]
    fn config_seed(&self) -> u64 {
        self.config.seed
    }

    #[getter]
    fn config_epochs(&self) -> usize {
        self.config.epochs
    }

    #[getter]
    fn config_learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    #[getter]
    fn config_negative_samples(&self) -> usize {
        self.config.negative_samples
    }

    #[getter]
    fn config_p(&self) -> f32 {
        self.config.p
    }

    #[getter]
    fn config_q(&self) -> f32 {
        self.config.q
    }
}

#[pyclass(name = "HeteroGraphSageEncoder")]
#[derive(Clone)]
struct NativeHeteroGraphSageEncoder {
    config: HeteroGraphSageConfig,
    relation_count: usize,
    encoder: HeteroGraphSageEncoder,
}

#[pymethods]
impl NativeHeteroGraphSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, relation_count, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0x0D1A_2A3B_4C5D_6E7F, l2_regularization=1e-5))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        relation_count: usize,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
    ) -> PyResult<Self> {
        let config = HeteroGraphSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
        };
        let encoder = HeteroGraphSageEncoder::new(config.clone(), input_dim, relation_count)
            .map_err(to_py_neural_error)?;
        Ok(Self {
            config,
            relation_count,
            encoder,
        })
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn fit(
        &mut self,
        node_count: usize,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HeteroGraph::from_typed_edges(node_count, self.relation_count, &typed_edges)
            .map_err(to_py_neural_error)?;
        let mut model = HeteroGraphSageEncoder::new(
            self.config.clone(),
            self.encoder.input_dim(),
            self.relation_count,
        )
        .map_err(to_py_neural_error)?;
        let embedding = model
            .fit(&graph, &node_features)
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = self
            .encoder
            .encode(&node_features)
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    #[pyo3(signature = (node_count, edges, node_features))]
    fn encode_graph(
        &self,
        node_count: usize,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HeteroGraph::from_typed_edges(node_count, self.relation_count, &typed_edges)
            .map_err(to_py_neural_error)?;
        let embedding = self
            .encoder
            .encode_graph(&graph, &node_features)
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, path: String) -> PyResult<()> {
        self.encoder
            .save_artifact_json(path)
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self) -> PyResult<String> {
        self.encoder.to_artifact_json().map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(_cls: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let encoder =
            HeteroGraphSageEncoder::load_artifact_json(path).map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self {
            relation_count: encoder.relation_count(),
            config,
            encoder,
        })
    }

    #[getter]
    fn relation_count(&self) -> usize {
        self.relation_count
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }
}

#[pyclass(name = "HinSageEncoder")]
#[derive(Clone)]
struct NativeHinSageEncoder {
    config: HinSageConfig,
    node_type_count: usize,
    edge_type_triples: Vec<(usize, usize, usize)>,
    encoder: HinSageEncoder,
}

#[pymethods]
impl NativeHinSageEncoder {
    #[new]
    #[pyo3(signature = (input_dim, node_type_count, edge_type_triples, hidden_dims=None, epochs=20, learning_rate=0.05, negative_samples=4, seed=0xA11C_E5A6_5EED_1234, l2_regularization=1e-5, neighbor_samples=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        node_type_count: usize,
        edge_type_triples: Vec<(usize, usize, usize)>,
        hidden_dims: Option<Vec<usize>>,
        epochs: usize,
        learning_rate: f32,
        negative_samples: usize,
        seed: u64,
        l2_regularization: f32,
        neighbor_samples: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let config = HinSageConfig {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![16]),
            epochs,
            learning_rate,
            negative_samples,
            seed,
            l2_regularization,
            neighbor_samples: neighbor_samples.unwrap_or_default(),
        };
        let encoder = HinSageEncoder::new(
            config.clone(),
            input_dim,
            node_type_count,
            edge_type_triples.clone(),
        )
        .map_err(to_py_neural_error)?;
        Ok(Self {
            config,
            node_type_count,
            edge_type_triples,
            encoder,
        })
    }

    #[pyo3(signature = (node_types, edges, node_features))]
    fn fit(
        &mut self,
        node_types: Vec<usize>,
        edges: Vec<(usize, usize, usize)>,
        node_features: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let typed_edges = edges
            .into_iter()
            .map(|(source, target, relation)| HeteroTypedEdge {
                source,
                target,
                relation,
            })
            .collect::<Vec<_>>();
        let graph = HinSageGraph::from_typed_schema(
            node_types,
            self.node_type_count,
            self.edge_type_triples.len(),
            self.edge_type_triples.clone(),
            typed_edges,
        )
        .map_err(to_py_neural_error)?;
        let mut model = HinSageEncoder::new(
            self.config.clone(),
            self.encoder.input_dim(),
            self.node_type_count,
            self.edge_type_triples.clone(),
        )
        .map_err(to_py_neural_error)?;
        let embedding = model
            .fit(&graph, &node_features)
            .map_err(to_py_neural_error)?;
        self.encoder = model;
        Ok(embedding.into_inner())
    }

    fn encode(&self, node_features: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let embedding = self
            .encoder
            .encode(&node_features)
            .map_err(to_py_neural_error)?;
        Ok(embedding.into_inner())
    }

    fn link_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Vec<Vec<f32>>> {
        self.encoder
            .link_embeddings(&embeddings, &pairs)
            .map_err(to_py_neural_error)
    }

    fn loss_curve(&self) -> Vec<f32> {
        self.encoder.loss_curve().values().to_vec()
    }

    fn save_artifact_json(&self, path: String) -> PyResult<()> {
        self.encoder
            .save_artifact_json(path)
            .map_err(to_py_neural_error)
    }

    fn to_artifact_json(&self) -> PyResult<String> {
        self.encoder.to_artifact_json().map_err(to_py_neural_error)
    }

    #[classmethod]
    fn load_artifact_json(_cls: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let encoder = HinSageEncoder::load_artifact_json(path).map_err(to_py_neural_error)?;
        let config = encoder.config();
        Ok(Self {
            node_type_count: encoder.node_type_count(),
            edge_type_triples: encoder.edge_type_triples().to_vec(),
            config,
            encoder,
        })
    }

    #[getter]
    fn node_type_count(&self) -> usize {
        self.node_type_count
    }

    #[getter]
    fn relation_count(&self) -> usize {
        self.edge_type_triples.len()
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    #[getter]
    fn output_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    #[getter]
    fn edge_type_triples(&self) -> Vec<(usize, usize, usize)> {
        self.edge_type_triples.clone()
    }

    #[getter]
    fn neighbor_samples(&self) -> Vec<usize> {
        self.config.neighbor_samples.clone()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        !self.encoder.loss_curve().values().is_empty()
    }
}

#[pyfunction]
#[pyo3(signature = (node_count, edges, embeddings, edge_weights=None, edge_timestamps=None, feature_prefix="graph", requested_features=None))]
fn graph_compute_directional_features(
    node_count: usize,
    edges: Vec<(usize, usize)>,
    embeddings: Vec<Vec<f32>>,
    edge_weights: Option<Vec<f32>>,
    edge_timestamps: Option<Vec<f32>>,
    feature_prefix: &str,
    requested_features: Option<Vec<String>>,
) -> PyResult<(Vec<Vec<f32>>, Vec<String>)> {
    let requested_features = requested_features.unwrap_or_default();
    let block = compute_directional_features(
        node_count,
        &edges,
        &embeddings,
        edge_weights.as_deref(),
        edge_timestamps.as_deref(),
        feature_prefix,
        &requested_features,
    )
    .map_err(to_py_neural_error)?;
    Ok((block.values, block.feature_names))
}

#[pyfunction]
fn graph_validate_directed_metapath(
    steps: Vec<String>,
    edge_types: Vec<(String, String, String)>,
) -> PyResult<()> {
    validate_directed_metapath(&steps, &edge_types).map_err(to_py_neural_error)
}

#[pyfunction]
#[pyo3(signature = (edges, source_to_pair_relation="source_to_pair", pair_to_target_relation="pair_to_target", pair_node_prefix="od_pair", include_original_edges=true))]
fn graph_materialize_source_target_pair_nodes(
    edges: Vec<(String, String, String)>,
    source_to_pair_relation: &str,
    pair_to_target_relation: &str,
    pair_node_prefix: &str,
    include_original_edges: bool,
) -> PyResult<(StringTypedEdges, Vec<String>)> {
    let expansion = materialize_source_target_pair_nodes(
        &edges,
        source_to_pair_relation,
        pair_to_target_relation,
        pair_node_prefix,
        include_original_edges,
    )
    .map_err(to_py_neural_error)?;
    Ok((expansion.edges, expansion.pair_node_ids))
}

fn artifact_fallback_name(fallback: &ArtifactFallbackKind) -> &'static str {
    match fallback {
        ArtifactFallbackKind::ZeroVector => "zero_vector",
        ArtifactFallbackKind::GlobalMeanVector => "global_mean_vector",
        ArtifactFallbackKind::ParentCell { .. } => "parent_cell",
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

fn parse_fuzzy_kernel(name: &str) -> PyResult<FuzzyKernel> {
    match name {
        "linear" | "triangular" => Ok(FuzzyKernel::Linear),
        "gaussian" => Ok(FuzzyKernel::Gaussian),
        "exponential" => Ok(FuzzyKernel::Exponential),
        "bisquare" => Ok(FuzzyKernel::Bisquare),
        "epanechnikov" => Ok(FuzzyKernel::Epanechnikov),
        "tricube" => Ok(FuzzyKernel::Tricube),
        _ => Err(PyValueError::new_err(format!(
            "unknown fuzzy_kernel {name:?}; expected 'linear', 'gaussian', 'exponential', 'bisquare', 'epanechnikov', or 'tricube'"
        ))),
    }
}

fn fuzzy_kernel_name(kernel: FuzzyKernel) -> &'static str {
    match kernel {
        FuzzyKernel::Linear => "linear",
        FuzzyKernel::Gaussian => "gaussian",
        FuzzyKernel::Exponential => "exponential",
        FuzzyKernel::Bisquare => "bisquare",
        FuzzyKernel::Epanechnikov => "epanechnikov",
        FuzzyKernel::Tricube => "tricube",
    }
}

fn parse_loss(
    name: &str,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
) -> PyResult<LossConfig> {
    match name {
        "l2" | "squared_error" => Ok(LossConfig::L2),
        "l1" | "mae" | "absolute_error" | "least_absolute_deviation" | "lad" => Ok(LossConfig::L1),
        "huber" => {
            if !huber_delta.is_finite() || huber_delta <= 0.0 {
                return Err(PyValueError::new_err(
                    "huber_delta must be positive and finite",
                ));
            }
            Ok(LossConfig::Huber(HuberLossConfig { delta: huber_delta }))
        }
        "log_l2" | "log" | "log_squared_error" => {
            if !log_offset.is_finite() || log_offset <= 0.0 {
                return Err(PyValueError::new_err(
                    "log_offset must be positive and finite",
                ));
            }
            if (log_offset - 1.0).abs() > 1e-12 {
                return Err(PyValueError::new_err(
                    "log_l2 currently supports log_offset=1.0",
                ));
            }
            Ok(LossConfig::LogL2(LogL2LossConfig { offset: log_offset }))
        }
        "quantile" | "pinball" => {
            if !quantile_alpha.is_finite() || quantile_alpha <= 0.0 || quantile_alpha >= 1.0 {
                return Err(PyValueError::new_err(
                    "quantile_alpha must be finite and in (0, 1)",
                ));
            }
            Ok(LossConfig::Quantile(QuantileLossConfig {
                alpha: quantile_alpha,
            }))
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown loss {name:?}; expected 'l2', 'l1', 'huber', 'log_l2', or 'quantile'"
        ))),
    }
}

fn loss_name(loss: &LossConfig) -> &'static str {
    match loss {
        LossConfig::L2 => "l2",
        LossConfig::L1 => "l1",
        LossConfig::Huber(_) => "huber",
        LossConfig::LogL2(_) => "log_l2",
        LossConfig::Quantile(_) => "quantile",
    }
}

fn quantile_alpha(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::L2 | LossConfig::L1 | LossConfig::Huber(_) | LossConfig::LogL2(_) => 0.5,
        LossConfig::Quantile(config) => config.alpha,
    }
}

fn huber_delta(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::Huber(config) => config.delta,
        _ => 1.0,
    }
}

fn log_offset(loss: &LossConfig) -> f64 {
    match loss {
        LossConfig::LogL2(config) => config.offset,
        _ => 1.0,
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

#[allow(clippy::too_many_arguments)]
fn validate_params(
    n_estimators: usize,
    learning_rate: f64,
    _max_depth: usize,
    min_samples_leaf: usize,
    min_gain: f64,
    l2_regularization: f64,
    constant_l2_regularization: f64,
    fuzzy_bandwidth: f64,
    quantile_alpha: f64,
    huber_delta: f64,
    log_offset: f64,
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
    if !constant_l2_regularization.is_finite() || constant_l2_regularization < 0.0 {
        return Err(PyValueError::new_err(
            "constant_l2_regularization must be finite and non-negative",
        ));
    }
    if !fuzzy_bandwidth.is_finite() || fuzzy_bandwidth < 0.0 {
        return Err(PyValueError::new_err(
            "fuzzy_bandwidth must be finite and non-negative",
        ));
    }
    if !quantile_alpha.is_finite() || quantile_alpha <= 0.0 || quantile_alpha >= 1.0 {
        return Err(PyValueError::new_err(
            "quantile_alpha must be finite and in (0, 1)",
        ));
    }
    if !huber_delta.is_finite() || huber_delta <= 0.0 {
        return Err(PyValueError::new_err(
            "huber_delta must be positive and finite",
        ));
    }
    if !log_offset.is_finite() || log_offset <= 0.0 {
        return Err(PyValueError::new_err(
            "log_offset must be positive and finite",
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

#[derive(Clone)]
struct OverlayPoint {
    id: String,
    coordinates: (f64, f64),
    properties: serde_json::Map<String, Value>,
}

struct OverlayZone {
    id: String,
    priority: f64,
    bbox: (f64, f64, f64, f64),
    ring: Vec<(f64, f64)>,
}

#[pyfunction(signature = (points, zones, weights, origin=None, zone_priority_multiplier=true, kernel="none", bandwidth_meters=None, distance_alpha=0.0, precision=6, include_debug=false))]
#[allow(clippy::too_many_arguments)]
fn weighted_overlay(
    py: Python<'_>,
    points: Bound<'_, PyAny>,
    zones: Bound<'_, PyAny>,
    weights: Bound<'_, PyAny>,
    origin: Option<(f64, f64)>,
    zone_priority_multiplier: bool,
    kernel: &str,
    bandwidth_meters: Option<f64>,
    distance_alpha: f64,
    precision: usize,
    include_debug: bool,
) -> PyResult<Py<PyAny>> {
    let json_module = PyModule::import(py, "json")?;
    let points_payload = json_module
        .call_method1("dumps", (points,))?
        .extract::<String>()?;
    let zones_payload = json_module
        .call_method1("dumps", (zones,))?
        .extract::<String>()?;
    let weights_payload = json_module
        .call_method1("dumps", (weights,))?
        .extract::<String>()?;

    let points_value = serde_json::from_str::<Value>(&points_payload)
        .map_err(|err| PyValueError::new_err(format!("invalid points payload: {err}")))?;
    let zones_value = serde_json::from_str::<Value>(&zones_payload)
        .map_err(|err| PyValueError::new_err(format!("invalid zones payload: {err}")))?;
    let weights_value = serde_json::from_str::<Value>(&weights_payload)
        .map_err(|err| PyValueError::new_err(format!("invalid weights payload: {err}")))?;

    let result = weighted_overlay_impl(
        &points_value,
        &zones_value,
        &weights_value,
        origin,
        zone_priority_multiplier,
        kernel,
        bandwidth_meters,
        distance_alpha,
        precision,
        include_debug,
    )
    .map_err(PyValueError::new_err)?;

    let payload = serde_json::to_string(&result).map_err(|err| {
        PyValueError::new_err(format!("failed to serialize overlay result: {err}"))
    })?;
    Ok(json_module.call_method1("loads", (payload,))?.unbind())
}

#[allow(clippy::too_many_arguments)]
fn weighted_overlay_impl(
    points: &Value,
    zones: &Value,
    weights: &Value,
    origin: Option<(f64, f64)>,
    zone_priority_multiplier: bool,
    kernel: &str,
    bandwidth_meters: Option<f64>,
    distance_alpha: f64,
    precision: usize,
    include_debug: bool,
) -> Result<Value, String> {
    let overlay_points = parse_overlay_points(points)?;
    let overlay_zones = parse_overlay_zones(zones)?;
    let weight_map = weights
        .as_object()
        .ok_or_else(|| "weights must be a JSON object".to_string())?;

    let mut features = Vec::with_capacity(overlay_points.len());
    for point in &overlay_points {
        let zone = locate_zone(&overlay_zones, point.coordinates)?;
        let linear_score = weight_map.iter().try_fold(0.0, |score, (name, weight)| {
            let weight_value = weight
                .as_f64()
                .ok_or_else(|| format!("weight {name:?} must be numeric"))?;
            let property_value = point
                .properties
                .get(name)
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            Ok::<f64, String>(score + weight_value * property_value)
        })?;

        let priority = if zone_priority_multiplier {
            zone.priority
        } else {
            1.0
        };

        let spatial_term = if let Some(origin) = (kernel != "none" && distance_alpha != 0.0)
            .then_some(origin)
            .flatten()
        {
            let bandwidth =
                resolve_bandwidth(bandwidth_meters, point.coordinates, &overlay_points)?;
            let distance = haversine_meters(origin, point.coordinates);
            distance_alpha * kernel_weight(distance, bandwidth, kernel)?
        } else {
            0.0
        };

        let mut feature = json!({
            "id": point.id,
            "zone_id": zone.id,
            "boost_score": round_half_even(linear_score * priority * (1.0 + spatial_term), precision),
        });
        if include_debug {
            feature["debug"] = json!({
                "linear": round_half_even(linear_score, precision),
                "priority": round_half_even(priority, precision),
                "spatial_term": round_half_even(spatial_term, precision),
            });
        }
        features.push(feature);
    }

    features.sort_by(|left, right| {
        let right_score = right["boost_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        let left_score = left["boost_score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left["id"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(right["id"].as_str().unwrap_or(""))
            })
    });
    for (rank, feature) in features.iter_mut().enumerate() {
        feature["rank"] = json!(rank + 1);
    }

    let points_name = points
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| points.get("type").and_then(Value::as_str))
        .unwrap_or("points");
    let zones_name = zones
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| zones.get("type").and_then(Value::as_str))
        .unwrap_or("zones");

    let mut config = json!({
        "algorithm": "weighted_overlay",
        "weights": weights.clone(),
        "zone_priority_multiplier": zone_priority_multiplier,
        "rounding": {
            "places": precision,
            "mode": "half_even",
        },
    });
    if origin.is_some() || kernel != "none" || distance_alpha != 0.0 {
        config["distance_term"] = json!({
            "enabled": origin.is_some() && kernel != "none" && distance_alpha != 0.0,
            "source": if origin.is_some() { Value::String("origin".to_string()) } else { Value::Null },
            "kernel": kernel,
            "bandwidth_meters": bandwidth_meters,
            "distance_alpha": distance_alpha,
        });
    }

    Ok(json!({
        "schema_version": 1,
        "scenario": format!("{points_name}_x_{zones_name}"),
        "config": config,
        "features": features,
    }))
}

fn parse_overlay_points(points: &Value) -> Result<Vec<OverlayPoint>, String> {
    let features = points
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| "points must contain a features array".to_string())?;
    features
        .iter()
        .map(|feature| {
            let id = feature
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| "point features must provide an id".to_string())?;
            let cartometry = feature
                .get("cartometry")
                .ok_or_else(|| format!("point feature {id:?} is missing cartometry"))?;
            let cartometry_type = cartometry
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("point feature {id:?} cartometry is missing type"))?;
            if cartometry_type != "Point" {
                return Err(format!("point feature {id:?} must use Point cartometry"));
            }
            let coordinates = cartometry
                .get("coordinates")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("point feature {id:?} is missing coordinates"))?;
            if coordinates.len() < 2 {
                return Err(format!(
                    "point feature {id:?} must provide [x, y] coordinates"
                ));
            }
            let x = coordinates[0]
                .as_f64()
                .ok_or_else(|| format!("point feature {id:?} x coordinate must be numeric"))?;
            let y = coordinates[1]
                .as_f64()
                .ok_or_else(|| format!("point feature {id:?} y coordinate must be numeric"))?;
            let properties = feature
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            Ok(OverlayPoint {
                id: id.to_string(),
                coordinates: (x, y),
                properties,
            })
        })
        .collect()
}

fn parse_overlay_zones(zones: &Value) -> Result<Vec<OverlayZone>, String> {
    let features = zones
        .get("features")
        .and_then(Value::as_array)
        .ok_or_else(|| "zones must contain a features array".to_string())?;
    features
        .iter()
        .map(|feature| {
            let id = feature
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| "zone features must provide an id".to_string())?;
            let cartometry = feature
                .get("cartometry")
                .ok_or_else(|| format!("zone feature {id:?} is missing cartometry"))?;
            let cartometry_type = cartometry
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("zone feature {id:?} cartometry is missing type"))?;
            if cartometry_type != "Polygon" {
                return Err(format!("zone feature {id:?} must use Polygon cartometry"));
            }
            let rings = cartometry
                .get("coordinates")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("zone feature {id:?} is missing polygon coordinates"))?;
            let outer_ring = rings
                .first()
                .and_then(Value::as_array)
                .ok_or_else(|| format!("zone feature {id:?} is missing an outer ring"))?;
            let ring = outer_ring
                .iter()
                .map(|coordinate| {
                    let pair = coordinate.as_array().ok_or_else(|| {
                        format!("zone feature {id:?} ring coordinates must be arrays")
                    })?;
                    if pair.len() < 2 {
                        return Err(format!(
                            "zone feature {id:?} ring coordinates must have two values"
                        ));
                    }
                    let x = pair[0].as_f64().ok_or_else(|| {
                        format!("zone feature {id:?} x coordinate must be numeric")
                    })?;
                    let y = pair[1].as_f64().ok_or_else(|| {
                        format!("zone feature {id:?} y coordinate must be numeric")
                    })?;
                    Ok((x, y))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let bbox = bounding_box(&ring)?;
            let priority = feature
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|properties| properties.get("priority"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            Ok(OverlayZone {
                id: id.to_string(),
                priority,
                bbox,
                ring,
            })
        })
        .collect()
}

fn locate_zone(zones: &[OverlayZone], point: (f64, f64)) -> Result<&OverlayZone, String> {
    let (x, y) = point;
    zones
        .iter()
        .find(|zone| {
            let (min_x, min_y, max_x, max_y) = zone.bbox;
            min_x <= x
                && x <= max_x
                && min_y <= y
                && y <= max_y
                && point_in_polygon(point, &zone.ring)
        })
        .ok_or_else(|| format!("point ({x}, {y}) does not belong to any zone"))
}

fn bounding_box(ring: &[(f64, f64)]) -> Result<(f64, f64, f64, f64), String> {
    if ring.is_empty() {
        return Err("polygon ring must not be empty".to_string());
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in ring {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }
    Ok((min_x, min_y, max_x, max_y))
}

fn point_in_polygon(point: (f64, f64), ring: &[(f64, f64)]) -> bool {
    if ring.len() < 2 {
        return false;
    }
    let (x, y) = point;
    let mut inside = false;
    for index in 0..(ring.len() - 1) {
        let start = ring[index];
        let end = ring[index + 1];
        if point_on_segment(point, start, end) {
            return true;
        }
        let intersects = (start.1 > y) != (end.1 > y);
        if intersects {
            let slope_x =
                (end.0 - start.0) * (y - start.1) / ((end.1 - start.1).abs().max(1e-12)) + start.0;
            if x <= slope_x {
                inside = !inside;
            }
        }
    }
    inside
}

fn point_on_segment(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> bool {
    let cross = (point.0 - start.0) * (end.1 - start.1) - (point.1 - start.1) * (end.0 - start.0);
    if cross.abs() > 1e-9 {
        return false;
    }
    let min_x = start.0.min(end.0) - 1e-9;
    let max_x = start.0.max(end.0) + 1e-9;
    let min_y = start.1.min(end.1) - 1e-9;
    let max_y = start.1.max(end.1) + 1e-9;
    min_x <= point.0 && point.0 <= max_x && min_y <= point.1 && point.1 <= max_y
}

fn resolve_bandwidth(
    bandwidth_meters: Option<f64>,
    point: (f64, f64),
    points: &[OverlayPoint],
) -> Result<f64, String> {
    if let Some(bandwidth) = bandwidth_meters {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err("bandwidth_meters must be positive and finite".to_string());
        }
        return Ok(bandwidth);
    }
    let mut distances = points
        .iter()
        .filter(|candidate| candidate.coordinates != point)
        .map(|candidate| haversine_meters(point, candidate.coordinates))
        .collect::<Vec<_>>();
    if distances.is_empty() {
        return Ok(1.0);
    }
    distances.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Ok(distances[distances.len().min(3) - 1].max(1.0))
}

fn kernel_weight(distance_meters: f64, bandwidth_meters: f64, kernel: &str) -> Result<f64, String> {
    if !bandwidth_meters.is_finite() || bandwidth_meters <= 0.0 {
        return Err("bandwidth_meters must be positive and finite".to_string());
    }
    let ratio = distance_meters / bandwidth_meters;
    match kernel {
        "none" => Ok(0.0),
        "gaussian" => Ok((-0.5 * ratio * ratio).exp()),
        "bisquare" => {
            if ratio >= 1.0 {
                Ok(0.0)
            } else {
                Ok((1.0 - ratio * ratio).powi(2))
            }
        }
        "exponential" => Ok((-ratio).exp()),
        _ => Err(format!("unknown kernel {kernel:?}")),
    }
}

fn haversine_meters(origin: (f64, f64), destination: (f64, f64)) -> f64 {
    let lon1 = origin.0.to_radians();
    let lat1 = origin.1.to_radians();
    let lon2 = destination.0.to_radians();
    let lat2 = destination.1.to_radians();
    let dlon = lon2 - lon1;
    let dlat = lat2 - lat1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * 6_371_000.0 * a.sqrt().asin()
}

fn round_half_even(value: f64, precision: usize) -> f64 {
    let factor = 10_f64.powi(precision as i32);
    let scaled = value * factor;
    let sign = if scaled.is_sign_negative() { -1.0 } else { 1.0 };
    let scaled_abs = scaled.abs();
    let lower = scaled_abs.floor();
    let fraction = scaled_abs - lower;
    let rounded = if fraction > 0.5 + 1e-12 {
        lower + 1.0
    } else if fraction < 0.5 - 1e-12 || (lower as i64) % 2 == 0 {
        lower
    } else {
        lower + 1.0
    };
    sign * rounded / factor
}

fn to_py_value_error(err: CartoBoostError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_py_error(err: CartoBoostError) -> PyErr {
    match err {
        CartoBoostError::Io(_) => PyIOError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn to_py_neural_error(err: cartoboost_neural::NeuralError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeCartoBoostRegressor>()?;
    m.add_class::<NativeNeuralEmbeddingFeatures>()?;
    m.add_class::<NativeGraphSageEncoder>()?;
    m.add_class::<NativeNode2VecEncoder>()?;
    m.add_class::<NativeHeteroGraphSageEncoder>()?;
    m.add_class::<NativeHinSageEncoder>()?;
    m.add_function(wrap_pyfunction!(graph_compute_directional_features, m)?)?;
    m.add_function(wrap_pyfunction!(graph_validate_directed_metapath, m)?)?;
    m.add_function(wrap_pyfunction!(
        graph_materialize_source_target_pair_nodes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(weighted_overlay, m)?)?;
    Ok(())
}
