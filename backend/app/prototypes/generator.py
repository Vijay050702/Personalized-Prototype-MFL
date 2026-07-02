from __future__ import annotations

from typing import Any

import torch

from app.core.logging import logger
from app.prototypes.prototype import Prototype
from app.prototypes.repository import PrototypeRepository


class PrototypeGenerator:
    def __init__(self, strategy: str = "centroid"):
        self._strategy = strategy

    @property
    def strategy(self) -> str:
        return self._strategy

    def generate_from_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_id: int,
        modality: str = "shared",
        **kwargs: Any,
    ) -> Prototype:
        class_mask = labels == class_id
        class_embeddings = embeddings[class_mask]
        if class_embeddings.size(0) == 0:
            raise ValueError(f"No embeddings found for class {class_id}")

        if self._strategy == "centroid":
            proto_embedding = class_embeddings.mean(dim=0)
        elif self._strategy == "weighted_centroid":
            weights = kwargs.get("weights", None)
            if weights is not None:
                class_weights = weights[class_mask].float()
                class_weights = class_weights / class_weights.sum()
                proto_embedding = (class_embeddings * class_weights.unsqueeze(1)).sum(
                    dim=0
                )
            else:
                proto_embedding = class_embeddings.mean(dim=0)
        elif self._strategy == "median":
            proto_embedding = class_embeddings.median(dim=0).values
        else:
            raise ValueError(f"Unknown generation strategy: {self._strategy}")

        confidence = kwargs.get("confidence", 1.0)
        metadata = kwargs.get("metadata", {})
        prototype = Prototype(
            embedding=proto_embedding,
            class_id=class_id,
            modality=modality,
            sample_count=class_embeddings.size(0),
            confidence=confidence,
            metadata=metadata,
        )
        logger.info(f"Generated prototype for class {class_id} using {self._strategy}")
        return prototype

    def generate_all(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        modality: str = "shared",
        **kwargs: Any,
    ) -> list[Prototype]:
        class_ids = sorted(torch.unique(labels).tolist())
        prototypes = []
        for cid in class_ids:
            proto = self.generate_from_embeddings(
                embeddings, labels, cid, modality=modality, **kwargs
            )
            prototypes.append(proto)
        logger.info(
            f"Generated {len(prototypes)} prototypes for {len(class_ids)} classes"
        )
        return prototypes

    def generate_from_repository(
        self,
        repository: PrototypeRepository,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        modality: str = "shared",
        **kwargs: Any,
    ) -> int:
        prototypes = self.generate_all(embeddings, labels, modality=modality, **kwargs)
        for proto in prototypes:
            existing = repository.by_class(proto.class_id)
            for old in existing:
                repository.remove(old.prototype_id)
            repository.store(proto)
        return len(prototypes)

    def incremental_update(
        self,
        prototype: Prototype,
        new_embeddings: torch.Tensor,
        alpha: float = 0.5,
    ) -> Prototype:
        if new_embeddings.dim() == 1:
            new_embeddings = new_embeddings.unsqueeze(0)
        avg_new = new_embeddings.mean(dim=0)
        updated = (1.0 - alpha) * prototype.embedding + alpha * avg_new
        prototype.embedding = updated
        prototype.sample_count += new_embeddings.size(0)
        logger.debug(f"Incremental update for prototype {prototype.prototype_id}")
        return prototype

    def batch_update(
        self,
        prototypes: list[Prototype],
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> list[Prototype]:
        for proto in prototypes:
            class_mask = labels == proto.class_id
            class_embeddings = embeddings[class_mask]
            if class_embeddings.size(0) > 0:
                self.incremental_update(proto, class_embeddings, **kwargs)
        return prototypes
