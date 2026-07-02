from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from app.knowledge_transfer.alignment_network import AlignmentNetwork
from app.knowledge_transfer.contrastive_alignment import (
    ContrastiveAlignmentLoss,
    InfoNCELoss,
    TripletLoss,
)
from app.knowledge_transfer.similarity import Similarity
from app.knowledge_transfer.transfer_loss import TransferLoss


class TransferRegistry:
    def __init__(self):
        self._alignment_networks: dict[str, type[AlignmentNetwork]] = {
            "default": AlignmentNetwork,
        }
        self._loss_functions: dict[str, type[nn.Module]] = {
            "info_nce": InfoNCELoss,
            "triplet": TripletLoss,
            "contrastive": ContrastiveAlignmentLoss,
        }
        self._similarity_metrics: dict[str, str] = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "dot": "dot",
        }
        self._mapper_types: dict[str, str] = {
            "linear": "linear",
            "mlp": "mlp",
        }
        self._activations: dict[str, str] = {
            "relu": "relu",
            "tanh": "tanh",
            "gelu": "gelu",
        }
        self._custom_factories: dict[str, Callable[..., Any]] = {}

    def register_alignment_network(
        self, name: str, cls: type[AlignmentNetwork]
    ) -> None:
        if name in self._alignment_networks:
            raise ValueError(f"Alignment network '{name}' is already registered")
        self._alignment_networks[name] = cls

    def get_alignment_network(self, name: str) -> type[AlignmentNetwork]:
        if name not in self._alignment_networks:
            raise ValueError(
                f"Unknown alignment network '{name}'. "
                f"Available: {self.list_alignment_networks()}"
            )
        return self._alignment_networks[name]

    def list_alignment_networks(self) -> list[str]:
        return sorted(self._alignment_networks.keys())

    def register_loss_function(self, name: str, cls: type[nn.Module]) -> None:
        if name in self._loss_functions:
            raise ValueError(f"Loss function '{name}' is already registered")
        self._loss_functions[name] = cls

    def get_loss_function(self, name: str) -> type[nn.Module]:
        if name not in self._loss_functions:
            raise ValueError(
                f"Unknown loss function '{name}'. "
                f"Available: {self.list_loss_functions()}"
            )
        return self._loss_functions[name]

    def list_loss_functions(self) -> list[str]:
        return sorted(self._loss_functions.keys())

    def register_similarity_metric(self, name: str) -> None:
        if name in self._similarity_metrics:
            raise ValueError(f"Similarity metric '{name}' is already registered")
        self._similarity_metrics[name] = name

    def get_similarity_metric(self, name: str) -> str:
        if name not in self._similarity_metrics:
            raise ValueError(
                f"Unknown similarity metric '{name}'. "
                f"Available: {self.list_similarity_metrics()}"
            )
        return self._similarity_metrics[name]

    def list_similarity_metrics(self) -> list[str]:
        return sorted(self._similarity_metrics.keys())

    def list_mapper_types(self) -> list[str]:
        return sorted(self._mapper_types.keys())

    def list_activations(self) -> list[str]:
        return sorted(self._activations.keys())

    def register_component(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._custom_factories:
            raise ValueError(f"Component '{name}' is already registered")
        self._custom_factories[name] = factory

    def get_component(self, name: str, **kwargs: Any) -> Any:
        if name not in self._custom_factories:
            raise ValueError(
                f"Unknown component '{name}'. Available: {self.list_components()}"
            )
        return self._custom_factories[name](**kwargs)

    def list_components(self) -> list[str]:
        return sorted(self._custom_factories.keys())

    def to_config(self) -> dict[str, Any]:
        return {
            "alignment_networks": self.list_alignment_networks(),
            "loss_functions": self.list_loss_functions(),
            "similarity_metrics": self.list_similarity_metrics(),
            "mapper_types": self.list_mapper_types(),
            "activations": self.list_activations(),
            "custom_components": self.list_components(),
        }
