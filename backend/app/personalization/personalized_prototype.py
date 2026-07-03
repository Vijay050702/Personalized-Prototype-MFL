from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PersonalizedPrototype:
    client_id: str
    class_id: int
    modality: str
    local_prototype: list[float] | None = None
    global_prototype: list[float] | None = None
    cross_modal_prototype: list[float] | None = None
    personalized_prototype: list[float] | None = None
    fusion_weights: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    embedding_dim: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tensor(self) -> torch.Tensor:
        if self.personalized_prototype is None:
            raise ValueError("No personalized prototype vector available")
        return torch.tensor(self.personalized_prototype, dtype=torch.float32)

    def has_local(self) -> bool:
        return self.local_prototype is not None

    def has_global(self) -> bool:
        return self.global_prototype is not None

    def has_cross_modal(self) -> bool:
        return self.cross_modal_prototype is not None

    def available_sources(self) -> list[str]:
        sources: list[str] = []
        if self.has_local():
            sources.append("local")
        if self.has_global():
            sources.append("global")
        if self.has_cross_modal():
            sources.append("cross_modal")
        return sources

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "class_id": self.class_id,
            "modality": self.modality,
            "local_prototype": self.local_prototype,
            "global_prototype": self.global_prototype,
            "cross_modal_prototype": self.cross_modal_prototype,
            "personalized_prototype": self.personalized_prototype,
            "fusion_weights": self.fusion_weights,
            "confidence": self.confidence,
            "embedding_dim": self.embedding_dim,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
