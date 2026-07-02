from __future__ import annotations

from typing import Any

import torch

from app.knowledge_transfer.alignment_network import AlignmentNetwork
from app.knowledge_transfer.modality_graph import ModalityGraph
from app.knowledge_transfer.validation import validate_no_nan, validate_prototype_size


class CrossModalMapper:
    def __init__(self, graph: ModalityGraph):
        self._graph = graph
        self._networks: dict[tuple[str, str], AlignmentNetwork] = {}

    def add_mapping_network(
        self, source: str, target: str, network: AlignmentNetwork
    ) -> None:
        key = (source, target)
        if key in self._networks:
            raise ValueError(f"Mapping {source} -> {target} already exists")
        if network.source_dim != self._graph.get_embedding_dim(source):
            raise ValueError(
                f"Network source_dim {network.source_dim} does not match "
                f"graph dim {self._graph.get_embedding_dim(source)} for {source}"
            )
        if network.target_dim != self._graph.get_embedding_dim(target):
            raise ValueError(
                f"Network target_dim {network.target_dim} does not match "
                f"graph dim {self._graph.get_embedding_dim(target)} for {target}"
            )
        self._networks[key] = network
        self._graph.add_mapping(source, target)

    def has_mapping(self, source: str, target: str) -> bool:
        return (source, target) in self._networks

    def get_mapping_network(self, source: str, target: str) -> AlignmentNetwork | None:
        return self._networks.get((source, target))

    def get_or_create_mapping_network(
        self,
        source: str,
        target: str,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        mapper_type: str = "linear",
    ) -> AlignmentNetwork:
        existing = self.get_mapping_network(source, target)
        if existing is not None:
            return existing

        source_dim = self._graph.get_embedding_dim(source)
        target_dim = self._graph.get_embedding_dim(target)
        if source_dim is None or target_dim is None:
            raise ValueError(
                f"Cannot determine dimensions for {source} -> {target}. "
                f"Set embedding dims first."
            )

        network = AlignmentNetwork(
            source_dim=source_dim,
            target_dim=target_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            mapper_type=mapper_type,
        )
        self.add_mapping_network(source, target, network)
        return network

    def translate(
        self,
        source_modality: str,
        target_modality: str,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        validate_prototype_size(embedding)
        validate_no_nan(embedding, "source_embedding")

        if source_modality == target_modality:
            return embedding.clone()

        network = self.get_mapping_network(source_modality, target_modality)
        if network is None:
            path = self._graph.find_path(source_modality, target_modality)
            if path is None or len(path) < 2:
                raise ValueError(
                    f"No mapping or path from {source_modality} to {target_modality}"
                )
            return self._translate_along_path(embedding, path)

        return network(embedding)

    def _translate_along_path(
        self, embedding: torch.Tensor, path: list[str]
    ) -> torch.Tensor:
        current = embedding
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            network = self.get_mapping_network(src, tgt)
            if network is None:
                raise ValueError(f"Missing mapping network: {src} -> {tgt}")
            current = network(current)
        return current

    def batch_translate(
        self,
        source_modality: str,
        target_modality: str,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        validate_prototype_size(embeddings, "batch_embeddings")
        validate_no_nan(embeddings, "batch_embeddings")

        network = self.get_mapping_network(source_modality, target_modality)
        if network is None:
            path = self._graph.find_path(source_modality, target_modality)
            if path is None or len(path) < 2:
                raise ValueError(
                    f"No mapping or path from {source_modality} to {target_modality}"
                )
            return self._batch_translate_along_path(embeddings, path)
        return network(embeddings)

    def _batch_translate_along_path(
        self, embeddings: torch.Tensor, path: list[str]
    ) -> torch.Tensor:
        current = embeddings
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            network = self.get_mapping_network(src, tgt)
            if network is None:
                raise ValueError(f"Missing mapping network: {src} -> {tgt}")
            current = network(current)
        return current

    def available_mappings(self) -> list[tuple[str, str]]:
        return sorted(self._networks.keys())

    def mapping_count(self) -> int:
        return len(self._networks)

    def to_config(self) -> dict[str, Any]:
        return {
            "mappings": [
                {"source": s, "target": t, **n.to_config()}
                for (s, t), n in self._networks.items()
            ]
        }
