from __future__ import annotations

from typing import Any

from app.knowledge_transfer.alignment_network import AlignmentNetwork
from app.knowledge_transfer.contrastive_alignment import (
    ContrastiveAlignmentLoss,
    InfoNCELoss,
    TripletLoss,
)
from app.knowledge_transfer.cross_modal_mapper import CrossModalMapper
from app.knowledge_transfer.inference import InferenceEngine
from app.knowledge_transfer.modality_graph import ModalityGraph
from app.knowledge_transfer.prototype_generator import PrototypeGenerator
from app.knowledge_transfer.registry import TransferRegistry
from app.knowledge_transfer.similarity import Similarity
from app.knowledge_transfer.transfer_loss import TransferLoss
from app.knowledge_transfer.utils import TransferLogger


class TransferFactory:
    @staticmethod
    def create_graph_with_modalities(
        modalities: dict[str, int],
        mappings: list[tuple[str, str]] | None = None,
    ) -> ModalityGraph:
        graph = ModalityGraph()
        for mod, dim in modalities.items():
            graph.add_modality(mod, dim)
        if mappings:
            for source, target in mappings:
                graph.add_mapping(source, target)
        return graph

    @staticmethod
    def create_default_mapper(
        modalities: dict[str, int],
        mappings: list[tuple[str, str]],
        mapper_type: str = "linear",
        activation: str = "relu",
        hidden_dims: list[int] | None = None,
    ) -> CrossModalMapper:
        graph = TransferFactory.create_graph_with_modalities(modalities, mappings)
        mapper = CrossModalMapper(graph)

        for source, target in mappings:
            network = AlignmentNetwork(
                source_dim=modalities[source],
                target_dim=modalities[target],
                hidden_dims=hidden_dims,
                activation=activation,
                mapper_type=mapper_type,
            )
            mapper.add_mapping_network(source, target, network)

        return mapper

    @staticmethod
    def create_default_generator(
        modalities: dict[str, int],
        mappings: list[tuple[str, str]],
        mapper_type: str = "linear",
        activation: str = "relu",
        hidden_dims: list[int] | None = None,
    ) -> PrototypeGenerator:
        mapper = TransferFactory.create_default_mapper(
            modalities=modalities,
            mappings=mappings,
            mapper_type=mapper_type,
            activation=activation,
            hidden_dims=hidden_dims,
        )
        return PrototypeGenerator(
            mapper=mapper,
            graph=mapper._graph,
        )

    @staticmethod
    def create_default_inference(
        modalities: dict[str, int],
        mappings: list[tuple[str, str]],
        mapper_type: str = "linear",
        activation: str = "relu",
        hidden_dims: list[int] | None = None,
    ) -> InferenceEngine:
        mapper = TransferFactory.create_default_mapper(
            modalities=modalities,
            mappings=mappings,
            mapper_type=mapper_type,
            activation=activation,
            hidden_dims=hidden_dims,
        )
        generator = PrototypeGenerator(
            mapper=mapper,
            graph=mapper._graph,
        )
        return InferenceEngine(
            mapper=mapper,
            graph=mapper._graph,
            generator=generator,
        )

    @staticmethod
    def create_loss(
        loss_type: str = "info_nce",
        temperature: float = 0.07,
        margin: float = 1.0,
    ) -> Any:
        if loss_type == "info_nce":
            return InfoNCELoss(temperature=temperature)
        elif loss_type == "triplet":
            return TripletLoss(margin=margin)
        elif loss_type == "contrastive":
            return ContrastiveAlignmentLoss(margin=margin)
        else:
            raise ValueError(
                f"Unknown loss type '{loss_type}'. "
                f"Choose from: info_nce, triplet, contrastive"
            )

    @staticmethod
    def create_transfer_loss(
        alignment_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        similarity_weight: float = 0.5,
        consistency_weight: float = 0.5,
        similarity_metric: str = "cosine",
    ) -> TransferLoss:
        return TransferLoss(
            alignment_weight=alignment_weight,
            reconstruction_weight=reconstruction_weight,
            similarity_weight=similarity_weight,
            consistency_weight=consistency_weight,
            similarity_metric=similarity_metric,
        )

    @staticmethod
    def create_from_config(config: dict[str, Any]) -> dict[str, Any]:
        modalities = config.get("modalities", {})
        mappings = config.get("mappings", [])
        mapper_type = config.get("mapper_type", "linear")
        activation = config.get("activation", "relu")
        hidden_dims = config.get("hidden_dims")

        mapper = TransferFactory.create_default_mapper(
            modalities=modalities,
            mappings=mappings,
            mapper_type=mapper_type,
            activation=activation,
            hidden_dims=hidden_dims,
        )
        generator = PrototypeGenerator(
            mapper=mapper,
            graph=mapper._graph,
        )
        inference = InferenceEngine(
            mapper=mapper,
            graph=mapper._graph,
            generator=generator,
        )
        loss_config = config.get("loss", {})
        loss = TransferFactory.create_loss(
            loss_type=loss_config.get("type", "info_nce"),
            temperature=loss_config.get("temperature", 0.07),
            margin=loss_config.get("margin", 1.0),
        )
        transfer_loss = TransferFactory.create_transfer_loss(
            alignment_weight=config.get("alignment_weight", 1.0),
            reconstruction_weight=config.get("reconstruction_weight", 1.0),
            similarity_weight=config.get("similarity_weight", 0.5),
            consistency_weight=config.get("consistency_weight", 0.5),
            similarity_metric=config.get("similarity_metric", "cosine"),
        )
        return {
            "mapper": mapper,
            "generator": generator,
            "inference": inference,
            "loss": loss,
            "transfer_loss": transfer_loss,
        }
