from __future__ import annotations

import time
import uuid
from typing import Any

import torch

from app.prototypes.utils import check_nan, validate_embedding


class Prototype:
    def __init__(
        self,
        embedding: torch.Tensor,
        class_id: int,
        modality: str = "shared",
        prototype_id: str | None = None,
        sample_count: int = 1,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ):
        validate_embedding(embedding)
        check_nan(embedding, "Prototype embedding")
        self._prototype_id = prototype_id or str(uuid.uuid4())
        self._class_id = class_id
        self._modality = modality
        self._embedding = embedding.detach().clone()
        self._sample_count = sample_count
        self._confidence = confidence
        self._timestamp = time.time()
        self._metadata = metadata or {}

    @property
    def prototype_id(self) -> str:
        return self._prototype_id

    @property
    def class_id(self) -> int:
        return self._class_id

    @class_id.setter
    def class_id(self, value: int) -> None:
        self._class_id = value

    @property
    def modality(self) -> str:
        return self._modality

    @property
    def embedding(self) -> torch.Tensor:
        return self._embedding

    @embedding.setter
    def embedding(self, value: torch.Tensor) -> None:
        validate_embedding(value)
        check_nan(value, "Prototype embedding")
        self._embedding = value.detach().clone()

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @sample_count.setter
    def sample_count(self, value: int) -> None:
        self._sample_count = max(0, value)

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self._confidence = max(0.0, min(1.0, value))

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def update(
        self,
        embedding: torch.Tensor,
        sample_count: int = 1,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._embedding = embedding.detach().clone()
        self._sample_count += sample_count
        if confidence is not None:
            self.confidence = confidence
        if metadata:
            self._metadata.update(metadata)
        self._timestamp = time.time()

    def clone(self) -> Prototype:
        return Prototype(
            embedding=self._embedding.clone(),
            class_id=self._class_id,
            modality=self._modality,
            prototype_id=str(uuid.uuid4()),
            sample_count=self._sample_count,
            confidence=self._confidence,
            metadata=dict(self._metadata),
        )

    def distance(
        self, other: Prototype | torch.Tensor, metric: str = "euclidean"
    ) -> torch.Tensor:
        if isinstance(other, Prototype):
            other = other.embedding
        if metric == "euclidean":
            return torch.nn.functional.pairwise_distance(
                self._embedding.unsqueeze(0), other.unsqueeze(0)
            )
        elif metric == "cosine":
            return 1.0 - torch.nn.functional.cosine_similarity(
                self._embedding.unsqueeze(0), other.unsqueeze(0)
            )
        elif metric == "manhattan":
            return torch.abs(self._embedding - other).sum()
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def similarity(
        self, other: Prototype | torch.Tensor, metric: str = "cosine"
    ) -> torch.Tensor:
        if isinstance(other, Prototype):
            other = other.embedding
        if metric == "cosine":
            return torch.nn.functional.cosine_similarity(
                self._embedding.unsqueeze(0), other.unsqueeze(0)
            )
        elif metric == "dot":
            return (self._embedding * other).sum()
        elif metric == "euclidean":
            return 1.0 / (
                1.0
                + torch.nn.functional.pairwise_distance(
                    self._embedding.unsqueeze(0), other.unsqueeze(0)
                )
            )
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def normalize(self) -> None:
        norm = self._embedding.norm(p=2)
        if norm > 0:
            self._embedding = self._embedding / norm

    def to_dict(self) -> dict[str, Any]:
        return {
            "prototype_id": self._prototype_id,
            "class_id": self._class_id,
            "modality": self._modality,
            "embedding": self._embedding.detach().cpu().tolist(),
            "sample_count": self._sample_count,
            "confidence": self._confidence,
            "timestamp": self._timestamp,
            "metadata": self._metadata,
            "embedding_dim": self._embedding.size(0),
        }

    def __repr__(self) -> str:
        return (
            f"Prototype(id={self._prototype_id[:8]}, "
            f"class={self._class_id}, "
            f"modality={self._modality}, "
            f"dim={self._embedding.size(0)}, "
            f"samples={self._sample_count}, "
            f"conf={self._confidence:.3f})"
        )
