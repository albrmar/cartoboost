use crate::error::{NeuralError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const GRAPH_SAGE_ARTIFACT_TYPE: &str = "cartoboost.neural.graphsage_encoder";
pub const GRAPH_SAGE_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSageEncoderArtifact {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub model: GraphSageModelArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphSageModelArtifact {
    Homogeneous(HomogeneousGraphSageEncoderArtifact),
    Hetero(HeteroGraphSageEncoderArtifact),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HomogeneousGraphSageEncoderArtifact {
    pub input_dim: usize,
    pub output_dim: usize,
    pub config: GraphSageConfig,
    pub layers: Vec<GraphSageLayer>,
    pub loss_curve: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeteroGraphSageEncoderArtifact {
    pub input_dim: usize,
    pub output_dim: usize,
    pub relation_count: usize,
    pub config: HeteroGraphSageConfig,
    pub layers: Vec<HeteroGraphSageLayer>,
    pub loss_curve: Vec<f32>,
}

/// Hyper-parameters for homogeneous GraphSAGE-style layers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSageConfig {
    pub hidden_dims: Vec<usize>,
    pub epochs: usize,
    pub learning_rate: f32,
    pub negative_samples: usize,
    pub seed: u64,
    pub add_self_loop: bool,
    pub l2_regularization: f32,
}

impl Default for GraphSageConfig {
    fn default() -> Self {
        Self {
            hidden_dims: vec![16],
            epochs: 20,
            learning_rate: 0.05,
            negative_samples: 4,
            seed: 0x5A17_9A4E_7F33_C0DE,
            add_self_loop: true,
            l2_regularization: 1e-5,
        }
    }
}

/// Hyper-parameters for hetero-typed GraphSAGE-style layers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeteroGraphSageConfig {
    pub hidden_dims: Vec<usize>,
    pub epochs: usize,
    pub learning_rate: f32,
    pub negative_samples: usize,
    pub seed: u64,
    pub l2_regularization: f32,
}

impl Default for HeteroGraphSageConfig {
    fn default() -> Self {
        Self {
            hidden_dims: vec![16],
            epochs: 20,
            learning_rate: 0.05,
            negative_samples: 4,
            seed: 0x0D1A_2A3B_4C5D_6E7F,
            l2_regularization: 1e-5,
        }
    }
}

/// Per-epoch loss record returned after fit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSageLoss {
    epoch_losses: Vec<f32>,
}

impl GraphSageLoss {
    pub fn values(&self) -> &[f32] {
        &self.epoch_losses
    }

    pub fn final_loss(&self) -> Option<f32> {
        self.epoch_losses.last().copied()
    }
}

/// Directed homogeneous graph with neighbor lists by source node.
#[derive(Debug, Clone)]
pub struct HomogeneousGraph {
    node_count: usize,
    neighbors: Vec<Vec<usize>>,
    edges: Vec<(usize, usize)>,
}

impl HomogeneousGraph {
    /// Builds a graph from explicit directed edges.
    ///
    /// `node_count` must be positive and every edge endpoint must be in-range.
    pub fn from_directed_edges(node_count: usize, edges: &[(usize, usize)]) -> Result<Self> {
        if node_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "node_count must be positive for a homogeneous graph".to_string(),
            ));
        }
        let neighbors = build_directed_neighbors(node_count, edges)?;
        Ok(Self {
            node_count,
            neighbors,
            edges: edges.to_vec(),
        })
    }

    pub fn from_undirected_edges(node_count: usize, edges: &[(usize, usize)]) -> Result<Self> {
        if node_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "node_count must be positive for a homogeneous graph".to_string(),
            ));
        }
        let mut undirected = Vec::with_capacity(edges.len() * 2);
        for &(source, target) in edges {
            undirected.push((source, target));
            undirected.push((target, source));
        }
        let neighbors = build_directed_neighbors(node_count, &undirected)?;
        Ok(Self {
            node_count,
            neighbors,
            edges: undirected,
        })
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    pub fn neighbors(&self) -> &[Vec<usize>] {
        &self.neighbors
    }
}

/// A typed edge for heterogeneous graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroTypedEdge {
    pub source: usize,
    pub target: usize,
    pub relation: usize,
}

/// Directed heterogeneous graph grouped by relation index.
#[derive(Debug, Clone)]
pub struct HeteroGraph {
    node_count: usize,
    relation_count: usize,
    edges: Vec<HeteroTypedEdge>,
    neighbors: Vec<Vec<Vec<usize>>>,
}

impl HeteroGraph {
    /// Builds a relation-typed graph from typed edges.
    ///
    /// `node_count` and `relation_count` must be positive. Every edge index must be
    /// in-range.
    pub fn from_typed_edges(
        node_count: usize,
        relation_count: usize,
        edges: &[HeteroTypedEdge],
    ) -> Result<Self> {
        if node_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "node_count must be positive for a heterogeneous graph".to_string(),
            ));
        }
        if relation_count == 0 {
            return Err(NeuralError::InvalidArgument(
                "relation_count must be positive for a heterogeneous graph".to_string(),
            ));
        }

        let mut neighbors = vec![vec![Vec::new(); relation_count]; node_count];

        for edge in edges {
            validate_node_index(edge.source, node_count)?;
            validate_node_index(edge.target, node_count)?;
            validate_relation_index(edge.relation, relation_count)?;
            neighbors[edge.source][edge.relation].push(edge.target);
        }

        for row in &mut neighbors {
            for rel_neighbors in row {
                rel_neighbors.sort_unstable();
                rel_neighbors.dedup();
            }
        }

        Ok(Self {
            node_count,
            relation_count,
            edges: edges.to_vec(),
            neighbors,
        })
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn relation_count(&self) -> usize {
        self.relation_count
    }

    pub fn edges(&self) -> &[HeteroTypedEdge] {
        &self.edges
    }

    pub fn neighbors(&self) -> &[Vec<Vec<usize>>] {
        &self.neighbors
    }
}

#[derive(Debug, Clone)]
pub struct GraphSageEncoder {
    config: GraphSageConfig,
    layers: Vec<GraphSageLayer>,
    input_dim: usize,
    output_dim: usize,
    losses: Vec<f32>,
}

impl GraphSageEncoder {
    pub fn new(config: GraphSageConfig, input_dim: usize) -> Result<Self> {
        validate_input_dim(input_dim)?;
        validate_dimensions(&config.hidden_dims)?;

        let mut dims = Vec::with_capacity(config.hidden_dims.len() + 1);
        dims.push(input_dim);
        dims.extend(config.hidden_dims.iter().copied());

        let mut rng = SplitMix64::from_seed(config.seed);
        let mut layers = Vec::with_capacity(config.hidden_dims.len());
        for pair in dims.windows(2) {
            let in_dim = pair[0];
            let out_dim = pair[1];
            layers.push(GraphSageLayer::new(in_dim, out_dim, &mut rng));
        }

        let output_dim = dims.last().copied().unwrap_or(input_dim);

        Ok(Self {
            config,
            layers,
            input_dim,
            output_dim,
            losses: Vec::new(),
        })
    }

    /// Serializes the full encoder state (hyperparameters and learned weights).
    pub fn to_artifact(&self) -> GraphSageEncoderArtifact {
        GraphSageEncoderArtifact {
            artifact_type: GRAPH_SAGE_ARTIFACT_TYPE.to_string(),
            artifact_version: GRAPH_SAGE_ARTIFACT_VERSION,
            model: GraphSageModelArtifact::Homogeneous(HomogeneousGraphSageEncoderArtifact {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                config: self.config.clone(),
                layers: self.layers.clone(),
                loss_curve: self.losses.clone(),
            }),
        }
    }

    /// Serializes encoder state as pretty JSON.
    pub fn to_artifact_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.to_artifact())?)
    }

    /// Writes encoder artifact JSON to `path`.
    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_artifact_json()?)?;
        Ok(())
    }

    /// Reconstructs an encoder from a previous artifact payload.
    pub fn from_artifact(artifact: GraphSageEncoderArtifact) -> Result<Self> {
        validate_graphsage_artifact(&artifact)?;
        let GraphSageModelArtifact::Homogeneous(payload) = artifact.model else {
            return Err(NeuralError::InvalidArgument(
                "artifact model kind is not homogeneous".to_string(),
            ));
        };
        Ok(Self {
            config: payload.config,
            layers: payload.layers,
            input_dim: payload.input_dim,
            output_dim: payload.output_dim,
            losses: payload.loss_curve,
        })
    }

    /// Loads an encoder from an artifact JSON payload.
    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    /// Returns the encoder configuration copy.
    pub fn config(&self) -> GraphSageConfig {
        self.config.clone()
    }

    pub fn fit(
        &mut self,
        graph: &HomogeneousGraph,
        node_features: &[Vec<f32>],
    ) -> Result<GraphSageEmbedding> {
        validate_node_features(Some(graph.node_count()), self.input_dim, node_features)?;

        if self.layers.is_empty() {
            self.losses.clear();
            self.losses.resize(self.config.epochs, 0.0);
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }

        let node_count = graph.node_count();
        let neighbors = if self.config.add_self_loop {
            add_self_neighbors(graph.neighbors())
        } else {
            graph.neighbors().to_vec()
        };

        self.losses = Vec::with_capacity(self.config.epochs);
        let mut rng = SplitMix64::from_seed(self.config.seed);
        let effective_negative = if node_count > 1 {
            self.config.negative_samples
        } else {
            0
        };

        for _ in 0..self.config.epochs {
            let cache = forward_homogeneous(node_features, &self.layers, &neighbors)?;
            let mut grad = vec![vec![0.0_f32; self.output_dim]; node_count];

            let loss = if graph.edges().is_empty() {
                0.0
            } else {
                compute_link_loss(
                    cache
                        .representations
                        .last()
                        .expect("cache must include final embeddings"),
                    graph.edges(),
                    effective_negative,
                    node_count,
                    &mut rng,
                    &mut grad,
                )
            };
            self.losses.push(loss);

            if !graph.edges().is_empty() {
                apply_homogeneous_backward(
                    &mut self.layers,
                    &cache,
                    &neighbors,
                    &grad,
                    self.config.learning_rate,
                    self.config.l2_regularization,
                )?;
            }
        }

        Ok(GraphSageEmbedding::new(
            forward_homogeneous(node_features, &self.layers, &neighbors)?
                .representations
                .pop()
                .expect("cache must include final embeddings"),
        ))
    }

    pub fn encode(&self, node_features: &[Vec<f32>]) -> Result<GraphSageEmbedding> {
        validate_node_features(None, self.input_dim, node_features)?;
        if self.layers.is_empty() {
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }
        let node_count = node_features.len();
        if node_count == 0 {
            return Ok(GraphSageEmbedding::new(Vec::new()));
        }
        let dummy_neighbors = vec![Vec::new(); node_count];
        Ok(GraphSageEmbedding::new(
            forward_homogeneous(node_features, &self.layers, &dummy_neighbors)?
                .representations
                .last()
                .expect("cache must include final embeddings")
                .clone(),
        ))
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn loss_curve(&self) -> GraphSageLoss {
        GraphSageLoss {
            epoch_losses: self.losses.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeteroGraphSageEncoder {
    config: HeteroGraphSageConfig,
    layers: Vec<HeteroGraphSageLayer>,
    input_dim: usize,
    output_dim: usize,
    relation_count: usize,
    losses: Vec<f32>,
}

impl HeteroGraphSageEncoder {
    pub fn new(
        config: HeteroGraphSageConfig,
        input_dim: usize,
        relation_count: usize,
    ) -> Result<Self> {
        validate_input_dim(input_dim)?;
        validate_relation_count(relation_count)?;
        validate_dimensions(&config.hidden_dims)?;

        let mut dims = Vec::with_capacity(config.hidden_dims.len() + 1);
        dims.push(input_dim);
        dims.extend(config.hidden_dims.iter().copied());

        let mut rng = SplitMix64::from_seed(config.seed);
        let mut layers = Vec::with_capacity(config.hidden_dims.len());
        for pair in dims.windows(2) {
            let in_dim = pair[0];
            let out_dim = pair[1];
            layers.push(HeteroGraphSageLayer::new(
                in_dim,
                out_dim,
                relation_count,
                &mut rng,
            ));
        }

        let output_dim = dims.last().copied().unwrap_or(input_dim);

        Ok(Self {
            config,
            layers,
            input_dim,
            output_dim,
            relation_count,
            losses: Vec::new(),
        })
    }

    /// Serializes the full encoder state (hyperparameters and learned weights).
    pub fn to_artifact(&self) -> GraphSageEncoderArtifact {
        GraphSageEncoderArtifact {
            artifact_type: GRAPH_SAGE_ARTIFACT_TYPE.to_string(),
            artifact_version: GRAPH_SAGE_ARTIFACT_VERSION,
            model: GraphSageModelArtifact::Hetero(HeteroGraphSageEncoderArtifact {
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                relation_count: self.relation_count,
                config: self.config.clone(),
                layers: self.layers.clone(),
                loss_curve: self.losses.clone(),
            }),
        }
    }

    /// Serializes encoder state as pretty JSON.
    pub fn to_artifact_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.to_artifact())?)
    }

    /// Writes encoder artifact JSON to `path`.
    pub fn save_artifact_json(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_artifact_json()?)?;
        Ok(())
    }

    /// Reconstructs an encoder from a previous artifact payload.
    pub fn from_artifact(artifact: GraphSageEncoderArtifact) -> Result<Self> {
        validate_graphsage_artifact(&artifact)?;
        let GraphSageModelArtifact::Hetero(payload) = artifact.model else {
            return Err(NeuralError::InvalidArgument(
                "artifact model kind is not heterogeneous".to_string(),
            ));
        };
        Ok(Self {
            config: payload.config,
            layers: payload.layers,
            input_dim: payload.input_dim,
            output_dim: payload.output_dim,
            relation_count: payload.relation_count,
            losses: payload.loss_curve,
        })
    }

    /// Loads an encoder from an artifact JSON payload.
    pub fn load_artifact_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        let artifact = serde_json::from_str(&text)?;
        Self::from_artifact(artifact)
    }

    /// Returns the encoder configuration copy.
    pub fn config(&self) -> HeteroGraphSageConfig {
        self.config.clone()
    }

    pub fn fit(
        &mut self,
        graph: &HeteroGraph,
        node_features: &[Vec<f32>],
    ) -> Result<GraphSageEmbedding> {
        validate_node_features(Some(graph.node_count()), self.input_dim, node_features)?;

        if self.layers.is_empty() {
            self.losses.clear();
            self.losses.resize(self.config.epochs, 0.0);
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }

        let node_count = graph.node_count();
        let neighbors = graph.neighbors().to_vec();
        self.losses = Vec::with_capacity(self.config.epochs);
        let mut rng = SplitMix64::from_seed(self.config.seed);
        let effective_negative = if node_count > 1 {
            self.config.negative_samples
        } else {
            0
        };

        for _ in 0..self.config.epochs {
            let cache = forward_hetero(node_features, &self.layers, &neighbors)?;
            let mut grad = vec![vec![0.0_f32; self.output_dim]; node_count];

            let loss = if graph.edges().is_empty() {
                0.0
            } else {
                compute_link_loss_hetero(
                    cache
                        .representations
                        .last()
                        .expect("cache must include final embeddings"),
                    graph.edges(),
                    effective_negative,
                    node_count,
                    &mut rng,
                    &mut grad,
                )
            };
            self.losses.push(loss);

            if !graph.edges().is_empty() {
                apply_hetero_backward(
                    &mut self.layers,
                    &cache,
                    &neighbors,
                    &grad,
                    self.config.learning_rate,
                    self.config.l2_regularization,
                )?;
            }
        }

        Ok(GraphSageEmbedding::new(
            forward_hetero(node_features, &self.layers, &neighbors)?
                .representations
                .last()
                .expect("cache must include final embeddings")
                .clone(),
        ))
    }

    pub fn encode(&self, node_features: &[Vec<f32>]) -> Result<GraphSageEmbedding> {
        validate_node_features(None, self.input_dim, node_features)?;
        if self.layers.is_empty() {
            return Ok(GraphSageEmbedding::new(node_features.to_vec()));
        }
        let node_count = node_features.len();
        if node_count == 0 {
            return Ok(GraphSageEmbedding::new(Vec::new()));
        }
        let dummy_neighbors = vec![vec![Vec::new(); self.relation_count]; node_count];
        Ok(GraphSageEmbedding::new(
            forward_hetero(node_features, &self.layers, &dummy_neighbors)?
                .representations
                .last()
                .expect("cache must include final embeddings")
                .clone(),
        ))
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn relation_count(&self) -> usize {
        self.relation_count
    }

    pub fn loss_curve(&self) -> GraphSageLoss {
        GraphSageLoss {
            epoch_losses: self.losses.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphSageEmbedding {
    vectors: Vec<Vec<f32>>,
}

impl GraphSageEmbedding {
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        Self { vectors }
    }

    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    pub fn into_inner(self) -> Vec<Vec<f32>> {
        self.vectors
    }

    pub fn dim(&self) -> usize {
        self.vectors.first().map_or(0, |row| row.len())
    }

    pub fn node_count(&self) -> usize {
        self.vectors.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSageLayer {
    in_dim: usize,
    out_dim: usize,
    self_weight: Vec<f32>,
    neigh_weight: Vec<f32>,
    bias: Vec<f32>,
}

impl GraphSageLayer {
    fn new(in_dim: usize, out_dim: usize, rng: &mut SplitMix64) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        let mut self_weight = Vec::with_capacity(in_dim * out_dim);
        let mut neigh_weight = Vec::with_capacity(in_dim * out_dim);
        for _ in 0..(in_dim * out_dim) {
            self_weight.push((rng.next_unit() * 2.0 - 1.0) * scale * 0.1);
            neigh_weight.push((rng.next_unit() * 2.0 - 1.0) * scale * 0.1);
        }

        Self {
            in_dim,
            out_dim,
            self_weight,
            neigh_weight,
            bias: vec![0.0; out_dim],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeteroGraphSageLayer {
    in_dim: usize,
    out_dim: usize,
    self_weight: Vec<f32>,
    relation_weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl HeteroGraphSageLayer {
    fn new(in_dim: usize, out_dim: usize, relation_count: usize, rng: &mut SplitMix64) -> Self {
        let scale = 1.0 / (in_dim as f32).sqrt();
        let mut relation_weights = Vec::with_capacity(relation_count);
        for _ in 0..relation_count {
            let mut relation_weight = Vec::with_capacity(in_dim * out_dim);
            for _ in 0..(in_dim * out_dim) {
                relation_weight.push((rng.next_unit() * 2.0 - 1.0) * scale * 0.1);
            }
            relation_weights.push(relation_weight);
        }

        Self {
            in_dim,
            out_dim,
            self_weight: (0..in_dim * out_dim)
                .map(|_| (rng.next_unit() * 2.0 - 1.0) * scale * 0.1)
                .collect(),
            relation_weights,
            bias: vec![0.0; out_dim],
        }
    }
}

#[derive(Debug)]
struct HomogeneousLayerCache {
    preactivations: Vec<Vec<f32>>,
    neighborhood_means: Vec<Vec<f32>>,
    representations: Vec<Vec<f32>>,
}

#[derive(Debug)]
struct HeteroLayerCache {
    preactivations: Vec<Vec<f32>>,
    relation_means: Vec<Vec<Vec<f32>>>,
    representations: Vec<Vec<f32>>,
}

#[derive(Debug)]
struct GraphSageForwardCache {
    representations: Vec<Vec<Vec<f32>>>,
    layers: Vec<HomogeneousLayerCache>,
}

#[derive(Debug)]
struct HeteroForwardCache {
    representations: Vec<Vec<Vec<f32>>>,
    layers: Vec<HeteroLayerCache>,
}

fn forward_homogeneous(
    node_features: &[Vec<f32>],
    layers: &[GraphSageLayer],
    neighbors: &[Vec<usize>],
) -> Result<GraphSageForwardCache> {
    if node_features.is_empty() {
        return Ok(GraphSageForwardCache {
            representations: vec![Vec::new()],
            layers: Vec::new(),
        });
    }

    let mut representations = Vec::with_capacity(layers.len() + 1);
    representations.push(node_features.to_vec());
    let mut cache_layers = Vec::with_capacity(layers.len());

    for layer in layers {
        let current = representations
            .last()
            .expect("layer activation should exist before running forward");
        let mut means = vec![vec![0.0_f32; layer.in_dim]; node_features.len()];
        for (node, neighbor_ids) in neighbors.iter().enumerate() {
            if neighbor_ids.is_empty() {
                continue;
            }
            let inv = 1.0 / (neighbor_ids.len() as f32);
            let mut accumulator = vec![0.0_f32; layer.in_dim];
            for &neighbor in neighbor_ids {
                for (index, value) in current[neighbor].iter().enumerate() {
                    accumulator[index] += *value;
                }
            }
            for index in 0..layer.in_dim {
                means[node][index] = accumulator[index] * inv;
            }
        }

        let mut preactivations = vec![vec![0.0_f32; layer.out_dim]; node_features.len()];
        let mut next = vec![vec![0.0_f32; layer.out_dim]; node_features.len()];
        for node in 0..node_features.len() {
            for out in 0..layer.out_dim {
                let mut value = layer.bias[out];
                for input_index in 0..layer.in_dim {
                    let self_value = current[node][input_index];
                    let neigh_value = means[node][input_index];
                    let self_pos = input_index * layer.out_dim + out;
                    value += self_value * layer.self_weight[self_pos];
                    value += neigh_value * layer.neigh_weight[self_pos];
                }
                preactivations[node][out] = value;
                next[node][out] = if value > 0.0 { value } else { 0.0 };
            }
        }

        let current = current.to_vec();
        representations.push(next);
        cache_layers.push(HomogeneousLayerCache {
            preactivations,
            neighborhood_means: means,
            representations: current,
        });
    }

    Ok(GraphSageForwardCache {
        representations,
        layers: cache_layers,
    })
}

fn forward_hetero(
    node_features: &[Vec<f32>],
    layers: &[HeteroGraphSageLayer],
    neighbors: &[Vec<Vec<usize>>],
) -> Result<HeteroForwardCache> {
    if node_features.is_empty() {
        return Ok(HeteroForwardCache {
            representations: vec![Vec::new()],
            layers: Vec::new(),
        });
    }

    if layers.first().is_some_and(|layer| {
        layer.relation_weights.len() != neighbors.first().map_or(0, |row| row.len())
    }) {
        return Err(NeuralError::InvalidArgument(
            "relation count must match hetero neighbor tensor".to_string(),
        ));
    }

    let mut representations = Vec::with_capacity(layers.len() + 1);
    representations.push(node_features.to_vec());
    let mut cache_layers = Vec::with_capacity(layers.len());

    for layer in layers {
        let current = representations
            .last()
            .expect("layer activation should exist before running forward");
        let relation_count = neighbors.first().map_or(0, |row| row.len());
        let mut relation_means =
            vec![vec![vec![0.0_f32; layer.in_dim]; relation_count]; node_features.len()];

        for (node_neighbors, relation_slots) in neighbors
            .iter()
            .zip(relation_means.iter_mut())
            .take(current.len())
        {
            for (relation_means_row, neighbor_ids) in relation_slots
                .iter_mut()
                .zip(node_neighbors.iter().take(relation_count))
            {
                if neighbor_ids.is_empty() {
                    continue;
                }
                let inv = 1.0 / (neighbor_ids.len() as f32);
                for &neighbor in neighbor_ids {
                    for (relation_mean, input_value) in
                        relation_means_row.iter_mut().zip(current[neighbor].iter())
                    {
                        *relation_mean += *input_value * inv;
                    }
                }
            }
        }

        let mut preactivations = vec![vec![0.0_f32; layer.out_dim]; node_features.len()];
        let mut next = vec![vec![0.0_f32; layer.out_dim]; node_features.len()];

        for node in 0..node_features.len() {
            for out in 0..layer.out_dim {
                let mut value = layer.bias[out];
                for index in 0..layer.in_dim {
                    value += current[node][index] * layer.self_weight[index * layer.out_dim + out];
                    for (relation, means_by_relation) in
                        relation_means[node].iter().enumerate().take(relation_count)
                    {
                        value += means_by_relation[index]
                            * layer.relation_weights[relation][index * layer.out_dim + out];
                    }
                }
                preactivations[node][out] = value;
                next[node][out] = if value > 0.0 { value } else { 0.0 };
            }
        }

        let current = current.to_vec();
        representations.push(next);
        cache_layers.push(HeteroLayerCache {
            preactivations,
            relation_means,
            representations: current,
        });
    }

    Ok(HeteroForwardCache {
        representations,
        layers: cache_layers,
    })
}

#[allow(clippy::needless_range_loop)]
fn apply_homogeneous_backward(
    layers: &mut [GraphSageLayer],
    cache: &GraphSageForwardCache,
    neighbors: &[Vec<usize>],
    grad_output: &[Vec<f32>],
    learning_rate: f32,
    l2_regularization: f32,
) -> Result<()> {
    if cache.layers.is_empty() {
        return Ok(());
    }

    let mut upstream_grad = grad_output.to_vec();

    for layer_index in (0..layers.len()).rev() {
        let layer = &mut layers[layer_index];
        let layer_cache = &cache.layers[layer_index];
        let input = &layer_cache.representations;
        let pre = &layer_cache.preactivations;
        let means = &layer_cache.neighborhood_means;
        let out_dim = layer.out_dim;
        let in_dim = layer.in_dim;

        let mut next_grad = vec![vec![0.0_f32; in_dim]; input.len()];
        let mut self_grad = vec![0.0_f32; layer.self_weight.len()];
        let mut neigh_grad = vec![0.0_f32; layer.neigh_weight.len()];
        let mut bias_grad = vec![0.0_f32; layer.bias.len()];

        for node in 0..input.len() {
            for out in 0..out_dim {
                let grad = upstream_grad[node][out];
                if pre[node][out] <= 0.0 {
                    continue;
                }

                bias_grad[out] += grad;

                for index in 0..in_dim {
                    let idx = index * out_dim + out;
                    let self_value = input[node][index];
                    let neigh_value = means[node][index];
                    let g = grad;

                    self_grad[idx] += self_value * g;
                    neigh_grad[idx] += neigh_value * g;
                    next_grad[node][index] += layer.self_weight[idx] * g;
                }

                if !neighbors[node].is_empty() {
                    let inv = 1.0 / neighbors[node].len() as f32;
                    for &neighbor in &neighbors[node] {
                        for index in 0..in_dim {
                            let idx = index * out_dim + out;
                            next_grad[neighbor][index] += layer.neigh_weight[idx] * grad * inv;
                        }
                    }
                }
            }
        }

        for (index, weight) in layer.self_weight.iter_mut().enumerate() {
            *weight -= learning_rate * (self_grad[index] + l2_regularization * *weight);
        }
        for (index, weight) in layer.neigh_weight.iter_mut().enumerate() {
            *weight -= learning_rate * (neigh_grad[index] + l2_regularization * *weight);
        }
        for (index, value) in layer.bias.iter_mut().enumerate() {
            *value -= learning_rate * bias_grad[index];
        }

        upstream_grad = next_grad;
    }

    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn apply_hetero_backward(
    layers: &mut [HeteroGraphSageLayer],
    cache: &HeteroForwardCache,
    neighbors: &[Vec<Vec<usize>>],
    grad_output: &[Vec<f32>],
    learning_rate: f32,
    l2_regularization: f32,
) -> Result<()> {
    if cache.layers.is_empty() {
        return Ok(());
    }

    let relation_count = neighbors.first().map_or(0, |row| row.len());
    let mut upstream_grad = grad_output.to_vec();

    for layer_index in (0..layers.len()).rev() {
        let layer = &mut layers[layer_index];
        let layer_cache = &cache.layers[layer_index];
        let input = &layer_cache.representations;
        let pre = &layer_cache.preactivations;
        let means = &layer_cache.relation_means;

        let out_dim = layer.out_dim;
        let in_dim = layer.in_dim;
        let mut next_grad = vec![vec![0.0_f32; in_dim]; input.len()];
        let mut self_grad = vec![0.0_f32; layer.self_weight.len()];
        let mut relation_grad = vec![vec![0.0_f32; layer.in_dim * out_dim]; relation_count];
        let mut bias_grad = vec![0.0_f32; layer.bias.len()];

        for node in 0..input.len() {
            for out in 0..out_dim {
                let grad = upstream_grad[node][out];
                if pre[node][out] <= 0.0 {
                    continue;
                }

                bias_grad[out] += grad;

                for index in 0..in_dim {
                    let weight_index = index * out_dim + out;
                    self_grad[weight_index] += input[node][index] * grad;
                    next_grad[node][index] += layer.self_weight[weight_index] * grad;
                }

                for (relation, neighbors_for_relation) in
                    neighbors[node].iter().enumerate().take(relation_count)
                {
                    if neighbors_for_relation.is_empty() {
                        continue;
                    }
                    let inv = 1.0 / (neighbors_for_relation.len() as f32);
                    #[allow(clippy::needless_range_loop)]
                    for index in 0..in_dim {
                        let weight_index = index * out_dim + out;
                        relation_grad[relation][weight_index] +=
                            means[node][relation][index] * grad;
                        let relation_weight = layer.relation_weights[relation][weight_index];
                        for &neighbor in neighbors_for_relation {
                            next_grad[neighbor][index] += relation_weight * grad * inv;
                        }
                    }
                }
            }
        }

        for (index, weight) in layer.self_weight.iter_mut().enumerate() {
            *weight -= learning_rate * (self_grad[index] + l2_regularization * *weight);
        }
        for (relation_grad_row, relation_weights) in relation_grad
            .iter()
            .zip(layer.relation_weights.iter_mut())
            .take(relation_count)
        {
            for (index, weight) in relation_weights.iter_mut().enumerate() {
                *weight -= learning_rate * (relation_grad_row[index] + l2_regularization * *weight);
            }
        }
        for (index, value) in layer.bias.iter_mut().enumerate() {
            *value -= learning_rate * bias_grad[index];
        }

        upstream_grad = next_grad;
    }

    Ok(())
}

fn compute_link_loss(
    embeddings: &[Vec<f32>],
    edges: &[(usize, usize)],
    negative_samples: usize,
    node_count: usize,
    rng: &mut SplitMix64,
    grad: &mut [Vec<f32>],
) -> f32 {
    if edges.is_empty() || node_count == 0 {
        return 0.0;
    }
    let mut loss = 0.0_f32;
    let scale = if edges.is_empty() {
        1.0
    } else {
        1.0 / ((edges.len() * (1 + negative_samples).max(1)) as f32)
    };

    for &(left, right) in edges {
        let mut pos_score = 0.0_f32;
        for (left_value, right_value) in embeddings[left].iter().zip(embeddings[right].iter()) {
            pos_score += left_value * right_value;
        }
        let pos_prob = sigmoid(pos_score);
        let safe_pos = safe_prob(pos_prob);
        loss += -safe_pos.ln();
        let pos_grad = (pos_prob - 1.0) * scale;

        for index in 0..grad[left].len() {
            grad[left][index] += pos_grad * embeddings[right][index];
            grad[right][index] += pos_grad * embeddings[left][index];
        }

        if negative_samples == 0 {
            continue;
        }

        for _ in 0..negative_samples {
            let negative = sample_negative_node(rng, node_count, right);
            let mut neg_score = 0.0_f32;
            for (left_value, neg_value) in embeddings[left].iter().zip(embeddings[negative].iter())
            {
                neg_score += left_value * neg_value;
            }
            let neg_prob = sigmoid(neg_score);
            let safe_neg = (1.0 - neg_prob).max(f32::EPSILON);
            loss += -safe_neg.ln();
            let neg_grad = neg_prob * scale;

            for index in 0..grad[left].len() {
                grad[left][index] += neg_grad * embeddings[negative][index];
                grad[negative][index] += neg_grad * embeddings[left][index];
            }
        }
    }

    loss
}

fn compute_link_loss_hetero(
    embeddings: &[Vec<f32>],
    edges: &[HeteroTypedEdge],
    negative_samples: usize,
    node_count: usize,
    rng: &mut SplitMix64,
    grad: &mut [Vec<f32>],
) -> f32 {
    if edges.is_empty() || node_count == 0 {
        return 0.0;
    }

    let mut loss = 0.0_f32;
    let scale = if edges.is_empty() {
        1.0
    } else {
        1.0 / ((edges.len() * (1 + negative_samples).max(1)) as f32)
    };
    for edge in edges {
        let left = edge.source;
        let right = edge.target;
        let mut pos_score = 0.0_f32;
        for (left_value, right_value) in embeddings[left].iter().zip(embeddings[right].iter()) {
            pos_score += left_value * right_value;
        }
        let pos_prob = sigmoid(pos_score);
        let safe_pos = safe_prob(pos_prob);
        loss += -safe_pos.ln();
        let pos_grad = (pos_prob - 1.0) * scale;

        for index in 0..grad[left].len() {
            grad[left][index] += pos_grad * embeddings[right][index];
            grad[right][index] += pos_grad * embeddings[left][index];
        }

        if negative_samples == 0 {
            continue;
        }

        for _ in 0..negative_samples {
            let negative = sample_negative_node(rng, node_count, right);
            let mut neg_score = 0.0_f32;
            for (left_value, neg_value) in embeddings[left].iter().zip(embeddings[negative].iter())
            {
                neg_score += left_value * neg_value;
            }
            let neg_prob = sigmoid(neg_score);
            let safe_neg = (1.0 - neg_prob).max(f32::EPSILON);
            loss += -safe_neg.ln();
            let neg_grad = neg_prob * scale;

            for index in 0..grad[left].len() {
                grad[left][index] += neg_grad * embeddings[negative][index];
                grad[negative][index] += neg_grad * embeddings[left][index];
            }
        }
    }

    loss
}

fn safe_prob(probability: f32) -> f32 {
    probability.clamp(f32::EPSILON, 1.0 - f32::EPSILON)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn sample_negative_node(rng: &mut SplitMix64, node_count: usize, forbidden: usize) -> usize {
    if node_count <= 1 {
        return forbidden;
    }

    loop {
        let candidate = rng.next_usize(node_count);
        if candidate != forbidden {
            return candidate;
        }
    }
}

fn build_directed_neighbors(
    node_count: usize,
    edges: &[(usize, usize)],
) -> Result<Vec<Vec<usize>>> {
    let mut neighbors = vec![Vec::new(); node_count];
    for &(source, target) in edges {
        validate_node_index(source, node_count)?;
        validate_node_index(target, node_count)?;
        neighbors[source].push(target);
    }

    for row in &mut neighbors {
        row.sort_unstable();
        row.dedup();
    }

    Ok(neighbors)
}

fn add_self_neighbors(neighbors: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut with_self = Vec::with_capacity(neighbors.len());
    for (node, source_neighbors) in neighbors.iter().enumerate() {
        let mut neighbors = source_neighbors.clone();
        if !neighbors.contains(&node) {
            neighbors.push(node);
            neighbors.sort_unstable();
        }
        with_self.push(neighbors);
    }
    with_self
}

fn validate_node_features(
    expected_node_count: Option<usize>,
    input_dim: usize,
    features: &[Vec<f32>],
) -> Result<()> {
    if let Some(expected_nodes) = expected_node_count {
        if expected_nodes != features.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "expected {expected_nodes} rows of features, got {}",
                features.len(),
            )));
        }
    }

    if let Some((index, row)) = features
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != input_dim)
    {
        return Err(NeuralError::InvalidArgument(format!(
            "row {index} has width {}, expected {}",
            row.len(),
            input_dim,
        )));
    }

    Ok(())
}

fn validate_input_dim(input_dim: usize) -> Result<()> {
    if input_dim == 0 {
        return Err(NeuralError::InvalidArgument(
            "input feature dimension must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_dimensions(hidden_dims: &[usize]) -> Result<()> {
    if hidden_dims.contains(&0) {
        return Err(NeuralError::InvalidArgument(
            "hidden_dims must contain only positive values".to_string(),
        ));
    }

    Ok(())
}

fn validate_node_index(node: usize, node_count: usize) -> Result<()> {
    if node >= node_count {
        return Err(NeuralError::InvalidArgument(format!(
            "node id {node} exceeds graph size {node_count}",
        )));
    }
    Ok(())
}

fn validate_relation_index(relation: usize, relation_count: usize) -> Result<()> {
    if relation >= relation_count {
        return Err(NeuralError::InvalidArgument(format!(
            "relation id {relation} exceeds relation count {relation_count}"
        )));
    }
    Ok(())
}

fn validate_relation_count(relation_count: usize) -> Result<()> {
    if relation_count == 0 {
        return Err(NeuralError::InvalidArgument(
            "relation_count must be positive for a hetero model".to_string(),
        ));
    }
    Ok(())
}

fn validate_graphsage_artifact(artifact: &GraphSageEncoderArtifact) -> Result<()> {
    if artifact.artifact_type != GRAPH_SAGE_ARTIFACT_TYPE {
        return Err(NeuralError::InvalidArgument(format!(
            "unsupported artifact type {}",
            artifact.artifact_type,
        )));
    }

    if artifact.artifact_version != GRAPH_SAGE_ARTIFACT_VERSION {
        return Err(NeuralError::InvalidArgument(format!(
            "unsupported artifact version {}",
            artifact.artifact_version,
        )));
    }

    Ok(())
}

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn next_unit(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / ((u64::MAX as f64) + 1.0) as f32;
        self.next_u64() as f32 * SCALE
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as f64 % (max as f64)) as usize
    }
}
