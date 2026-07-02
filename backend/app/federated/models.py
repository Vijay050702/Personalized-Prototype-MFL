from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
from pydantic import BaseModel, Field, field_validator


class ClientPrototypePackage(BaseModel):
    client_id: str
    round_id: int
    modality: str
    class_id: int
    prototype_vector: list[float]
    sample_count: int = Field(ge=1)
    embedding_dim: int = Field(ge=1)
    timestamp: float = Field(default_factory=time.time)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prototype_vector")
    @classmethod
    def _validate_vector_not_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("prototype_vector must not be empty")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def _validate_dim_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("embedding_dim must be >= 1")
        return v

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.prototype_vector, dtype=torch.float32)

    def package_id(self) -> str:
        return f"{self.client_id}_r{self.round_id}_{self.modality}_c{self.class_id}"


class AggregatedPrototype(BaseModel):
    class_id: int
    modality: str
    prototype_vector: list[float]
    embedding_dim: int
    sample_count: int
    confidence: float
    variance: float = 0.0
    num_contributors: int = 0
    round_id: int = 0
    aggregated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.prototype_vector, dtype=torch.float32)


class AggregationRound(BaseModel):
    round_id: int
    participating_clients: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    class_ids: list[int] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    num_packages_received: int = 0
    num_aggregated: int = 0
    status: str = "pending"

    @property
    def duration(self) -> float:
        if self.completed_at is not None and self.started_at is not None:
            return self.completed_at - self.started_at
        return 0.0

    def complete(self) -> None:
        self.completed_at = time.time()
        self.status = "completed"


class ModalityCompletenessReport(BaseModel):
    available_modalities: list[str] = Field(default_factory=list)
    missing_modalities: list[str] = Field(default_factory=list)
    total_possible: int = 0
    completeness_ratio: float = 0.0


class DivergenceReport(BaseModel):
    client_id: str
    modality: str
    class_id: int
    divergence_score: float
    divergence_metric: str


@dataclass
class FederatedState:
    current_round: int = 0
    total_clients_ever: set[str] = field(default_factory=set)
    packages_received: int = 0
    rounds_completed: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_round": self.current_round,
            "total_clients_ever": len(self.total_clients_ever),
            "packages_received": self.packages_received,
            "rounds_completed": self.rounds_completed,
            "started_at": self.started_at,
        }


@dataclass
class WeightedPrototype:
    prototype_vector: torch.Tensor
    weight: float
    sample_count: int
    client_id: str
    class_id: int
    modality: str
    confidence: float
