"""Graph encoders for CartoBoost-friendly feature construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .._native import (
    GraphSageEncoder as _NativeGraphSageEncoder,
    HeteroGraphSageEncoder as _NativeHeteroGraphSageEncoder,
)
from .builder import (
    HeterogeneousGraph,
    HomogeneousGraph,
    ensure_node_features_shape,
    normalize_homogeneous_graph,
    normalize_heterogeneous_graph,
)
from .features import GraphFeatureBundle


@dataclass(frozen=True)
class GraphSageConfig:
    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [16])
    epochs: int = 20
    learning_rate: float = 0.05
    negative_samples: int = 4
    seed: int = 0x5A17_9A4E_7F33_C0DE
    add_self_loop: bool = True
    l2_regularization: float = 1e-5


def _coerce_dim(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _coerce_hidden_dims(values: Sequence[int]) -> list[int]:
    if not values:
        return []
    hidden_dims = [int(v) for v in values]
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("hidden_dims must contain only positive values")
    return hidden_dims


class GraphSageFeatureEncoder:
    """High-level wrapper around the Rust GraphSageEncoder."""

    def __init__(self, config: GraphSageConfig) -> None:
        self.config = config
        self._encoder = _NativeGraphSageEncoder(
            input_dim=config.input_dim,
            hidden_dims=list(config.hidden_dims),
            epochs=config.epochs,
            learning_rate=float(config.learning_rate),
            negative_samples=int(config.negative_samples),
            seed=int(config.seed),
            add_self_loop=bool(config.add_self_loop),
            l2_regularization=float(config.l2_regularization),
        )
        self.graph: HomogeneousGraph | None = None

    def fit(
        self,
        node_features: Sequence[Sequence[float]],
        edges: Sequence[tuple[Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
    ) -> GraphFeatureBundle:
        graph = normalize_homogeneous_graph(
            edges=edges,
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
        )
        feature_rows = ensure_node_features_shape(node_features, graph.node_count)
        emb = self._encoder.fit(graph.node_count, graph.edges, feature_rows)
        self.graph = graph
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(graph.node_ids),
            feature_names=[f"graph_sage_homo_{index:02d}" for index in range(len(emb[0]))],
        )

    def encode(
        self,
        node_features: Sequence[Sequence[float]],
    ) -> GraphFeatureBundle:
        if self.graph is None:
            raise RuntimeError("GraphSageFeatureEncoder must be fitted first")
        feature_rows = ensure_node_features_shape(node_features, len(self.graph.node_ids))
        emb = self._encoder.encode(feature_rows)
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(self.graph.node_ids),
            feature_names=[f"graph_sage_homo_{index:02d}" for index in range(len(emb[0]))],
        )

    def loss_curve(self) -> list[float]:
        return self._encoder.loss_curve()

    def save_artifact_json(self, path: str) -> None:
        self._encoder.save_artifact_json(path)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GraphSageFeatureEncoder":
        input_dim = _coerce_dim(int(config["input_dim"]), "input_dim")
        hidden_dims = _coerce_hidden_dims(config.get("hidden_dims", [16]))
        return cls(
            GraphSageConfig(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                epochs=int(config.get("epochs", 20)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                negative_samples=int(config.get("negative_samples", 4)),
                seed=int(config.get("seed", 0x5A17_9A4E_7F33_C0DE)),
                add_self_loop=bool(config.get("add_self_loop", True)),
                l2_regularization=float(config.get("l2_regularization", 1e-5)),
            )
        )


@dataclass(frozen=True)
class HeteroGraphSageConfig:
    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [16])
    epochs: int = 20
    learning_rate: float = 0.05
    negative_samples: int = 4
    seed: int = 0x0D1A_2A3B_4C5D_6E7F
    l2_regularization: float = 1e-5


class HeteroGraphSageFeatureEncoder:
    """High-level wrapper around the Rust HeteroGraphSageEncoder."""

    def __init__(self, config: HeteroGraphSageConfig) -> None:
        self.config = config
        self._encoder = _NativeHeteroGraphSageEncoder(
            input_dim=config.input_dim,
            relation_count=1,
            hidden_dims=list(config.hidden_dims),
            epochs=config.epochs,
            learning_rate=float(config.learning_rate),
            negative_samples=int(config.negative_samples),
            seed=int(config.seed),
            l2_regularization=float(config.l2_regularization),
        )
        self.graph: HeterogeneousGraph | None = None

    def fit(
        self,
        node_features: Sequence[Sequence[float]],
        edges: Sequence[tuple[Any, Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        relation_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        materialize_reverse_edges: bool = False,
        reverse_relation_suffix: str = "_reverse",
        reverse_relation_map: Mapping[Any, Any] | None = None,
    ) -> GraphFeatureBundle:
        graph = normalize_heterogeneous_graph(
            edges=edges,
            node_ids=node_ids,
            relation_ids=relation_ids,
            node_count=node_count,
            directed=directed,
            materialize_reverse_edges=materialize_reverse_edges,
            reverse_relation_suffix=reverse_relation_suffix,
            reverse_relation_map=reverse_relation_map,
        )
        if self.config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if self._encoder.relation_count != graph.relation_count:
            # Keep relation count stable with the fitted topology.
            self._encoder = _NativeHeteroGraphSageEncoder(
                input_dim=self.config.input_dim,
                relation_count=graph.relation_count,
                hidden_dims=list(self.config.hidden_dims),
                epochs=self.config.epochs,
                learning_rate=float(self.config.learning_rate),
                negative_samples=int(self.config.negative_samples),
                seed=int(self.config.seed),
                l2_regularization=float(self.config.l2_regularization),
            )

        feature_rows = ensure_node_features_shape(node_features, graph.node_count)
        emb = self._encoder.fit(graph.node_count, graph.edges, feature_rows)
        self.graph = graph
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(graph.node_ids),
            feature_names=[
                f"graph_sage_hetero_{index:02d}" for index in range(len(emb[0]))
            ],
        )

    def encode(
        self,
        node_features: Sequence[Sequence[float]],
    ) -> GraphFeatureBundle:
        if self.graph is None:
            raise RuntimeError("HeteroGraphSageFeatureEncoder must be fitted first")
        feature_rows = ensure_node_features_shape(node_features, len(self.graph.node_ids))
        emb = self._encoder.encode(feature_rows)
        return GraphFeatureBundle(
            embeddings=emb,
            node_ids=list(self.graph.node_ids),
            feature_names=[
                f"graph_sage_hetero_{index:02d}" for index in range(len(emb[0]))
            ],
        )

    def loss_curve(self) -> list[float]:
        return self._encoder.loss_curve()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "HeteroGraphSageFeatureEncoder":
        input_dim = _coerce_dim(int(config["input_dim"]), "input_dim")
        hidden_dims = _coerce_hidden_dims(config.get("hidden_dims", [16]))
        return cls(
            HeteroGraphSageConfig(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                epochs=int(config.get("epochs", 20)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                negative_samples=int(config.get("negative_samples", 4)),
                seed=int(config.get("seed", 0x0D1A_2A3B_4C5D_6E7F)),
                l2_regularization=float(config.get("l2_regularization", 1e-5)),
            )
        )


class GraphFeatureTransformer:
    """Small configuration-first API for graph feature extraction."""

    def __init__(
        self,
        *,
        use_hetero: bool = False,
        sage_kwargs: Mapping[str, Any] | None = None,
        hetero_kwargs: Mapping[str, Any] | None = None,
        directionality: Mapping[str, Any] | None = None,
    ) -> None:
        self.use_hetero = bool(use_hetero)
        self.sage_kwargs = dict(sage_kwargs or {})
        self.hetero_kwargs = dict(hetero_kwargs or {})
        self.directionality = dict(directionality or {})
        self.encoder: GraphSageFeatureEncoder | HeteroGraphSageFeatureEncoder | None = None
        self._target_input_dim: int | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GraphFeatureTransformer":
        if not isinstance(cfg, Mapping):
            raise TypeError("graph config must be a mapping")
        graph_cfg = cfg.get("graph_embeddings")
        if graph_cfg is None:
            graph_cfg = cfg.get("graph_sage", {})
        if not isinstance(graph_cfg, Mapping):
            raise TypeError("graph configuration must be a mapping")

        encoder_cfg = graph_cfg.get("encoder", graph_cfg)
        if not isinstance(encoder_cfg, Mapping):
            raise TypeError("graph encoder configuration must be a mapping")

        family = str(encoder_cfg.get("family", "graphsage")).lower()
        if family not in {"graphsage", "hinsage", "sage"}:
            raise ValueError(f"unsupported graph family {family!r}")

        use_hetero = bool(encoder_cfg.get("hetero", graph_cfg.get("hetero", False)))
        if use_hetero:
            return cls(
                use_hetero=True,
                hetero_kwargs=dict(encoder_cfg),
                directionality=dict(graph_cfg.get("directionality", {})),
            )
        return cls(
            use_hetero=False,
            sage_kwargs=dict(encoder_cfg),
            directionality=dict(graph_cfg.get("directionality", {})),
        )

    def fit_transform(
        self,
        node_features: Sequence[Sequence[float]],
        edges: Sequence[tuple[Any, Any]] | Sequence[tuple[Any, Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        relation_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
    ) -> GraphFeatureBundle:
        feature_rows = [list(row) for row in node_features]
        if not feature_rows:
            raise ValueError("node_features must not be empty")
        self._target_input_dim = len(feature_rows[0])
        self._validate_input_dim()

        if self.use_hetero:
            return self._fit_hetero(
                feature_rows,
                edges=edges,  # type: ignore[arg-type]
                node_ids=node_ids,
                relation_ids=relation_ids,
                node_count=node_count,
                directed=directed,
                directionality=self.directionality,
            )
        return self._fit_homo(
            feature_rows,
            edges=edges,  # type: ignore[arg-type]
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
            directionality=self.directionality,
        )

    def _ensure_encoder(self) -> GraphSageFeatureEncoder | HeteroGraphSageFeatureEncoder:
        if self._target_input_dim is None:
            raise RuntimeError("GraphFeatureTransformer.fit_transform has not been initialized")
        if self.encoder is not None:
            return self.encoder

        if self.use_hetero:
            cfg = dict(self.hetero_kwargs)
            cfg.setdefault("input_dim", self._target_input_dim)
            self.encoder = HeteroGraphSageFeatureEncoder.from_config(
                cfg,
            )
            return self.encoder

        cfg = dict(self.sage_kwargs)
        cfg.setdefault("input_dim", self._target_input_dim)
        if self._target_input_dim is None:
            raise RuntimeError("input dimension could not be inferred")
        self.encoder = GraphSageFeatureEncoder.from_config(cfg)
        return self.encoder

    def _validate_input_dim(self) -> None:
        if self._target_input_dim is None or self._target_input_dim <= 0:
            raise ValueError("node features must have at least one column")
        if self.use_hetero:
            configured = self.hetero_kwargs.get("input_dim")
        else:
            configured = self.sage_kwargs.get("input_dim")
        if configured is None:
            return

        configured_dim = _coerce_dim(int(configured), "configured input_dim")
        if configured_dim != self._target_input_dim:
            raise ValueError(
                f"configured input_dim {configured_dim} does not match feature width "
                f"{self._target_input_dim}"
            )

    def _fit_homo(
        self,
        feature_rows: list[list[float]],
        edges: Sequence[tuple[Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        directionality: Mapping[str, Any] | None = None,
    ) -> GraphFeatureBundle:
        encoder = self._ensure_encoder()
        if not isinstance(encoder, GraphSageFeatureEncoder):
            raise RuntimeError("configured encoder mismatch; expected homogeneous")
        del directionality
        return encoder.fit(
            node_features=feature_rows,
            edges=edges,
            node_ids=node_ids,
            node_count=node_count,
            directed=directed,
        )

    def _fit_hetero(
        self,
        feature_rows: list[list[float]],
        edges: Sequence[tuple[Any, Any, Any]],
        *,
        node_ids: Sequence[Any] | None = None,
        relation_ids: Sequence[Any] | None = None,
        node_count: int | None = None,
        directed: bool = True,
        directionality: Mapping[str, Any] | None = None,
    ) -> GraphFeatureBundle:
        encoder = self._ensure_encoder()
        if not isinstance(encoder, HeteroGraphSageFeatureEncoder):
            raise RuntimeError("configured encoder mismatch; expected hetero")

        directionality = dict(directionality or {})
        materialize_reverse_edges = bool(directionality.get("materialize_reverse_edges", False))
        reverse_relation_suffix = str(
            directionality.get("reverse_relation_suffix", "_reverse"),
        )
        reverse_relation_map = directionality.get("reverse_relation_map")
        if reverse_relation_map is not None and not isinstance(reverse_relation_map, Mapping):
            raise TypeError("reverse_relation_map must be a mapping when provided")

        return encoder.fit(
            node_features=feature_rows,
            edges=edges,
            node_ids=node_ids,
            relation_ids=relation_ids,
            node_count=node_count,
            directed=directed,
            materialize_reverse_edges=materialize_reverse_edges,
            reverse_relation_suffix=reverse_relation_suffix,
            reverse_relation_map=reverse_relation_map,
        )
