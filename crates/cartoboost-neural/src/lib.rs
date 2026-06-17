pub mod artifact;
pub mod encoder;
mod error;
pub mod features;
pub mod graphsage;
mod trainer;

pub use artifact::{
    build_embedding_table_artifact, write_embedding_table_artifact, ArtifactFallbackKind,
    EmbeddingChecksum, EmbeddingIdType, EmbeddingRow, EmbeddingTable, EmbeddingTableArtifact,
    EmbeddingTableMetadata, FallbackStrategy,
};
pub use encoder::{EmbeddingTableEncoder, NeuralEncoder};
pub use error::{NeuralError, Result};
pub use features::NeuralFeatureBlock;
pub use graphsage::{
    GraphSageConfig, GraphSageEncoder, GraphSageEncoderArtifact, GraphSageModelArtifact,
    GraphSageLoss, HeteroGraph, HeteroGraphSageConfig, HeteroGraphSageEncoder,
    HeteroGraphSageEncoderArtifact, HeteroTypedEdge, HomogeneousGraph,
};
pub use trainer::fit_embedding_table;
