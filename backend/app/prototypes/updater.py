from __future__ import annotations

from typing import Any

import torch

from app.core.logging import logger
from app.prototypes.prototype import Prototype


class PrototypeUpdater:
    def __init__(self, strategy: str = "ema"):
        self._strategy = strategy

    @property
    def strategy(self) -> str:
        return self._strategy

    def update(
        self,
        prototype: Prototype,
        new_embedding: torch.Tensor,
        **kwargs: Any,
    ) -> Prototype:
        if self._strategy == "ema":
            return self._ema_update(prototype, new_embedding, kwargs.get("alpha", 0.9))
        elif self._strategy == "moving_average":
            return self._moving_average_update(prototype, new_embedding)
        elif self._strategy == "replacement":
            return self._replacement_update(prototype, new_embedding)
        elif self._strategy == "weighted":
            return self._weighted_update(
                prototype, new_embedding, kwargs.get("weight", 0.5)
            )
        elif self._strategy == "adaptive":
            return self._adaptive_update(prototype, new_embedding)
        else:
            raise ValueError(f"Unknown update strategy: {self._strategy}")

    def _ema_update(
        self, prototype: Prototype, new_embedding: torch.Tensor, alpha: float = 0.9
    ) -> Prototype:
        updated = alpha * prototype.embedding + (1.0 - alpha) * new_embedding
        prototype.embedding = updated
        prototype.sample_count += 1
        logger.debug(f"EMA update for {prototype.prototype_id} (alpha={alpha})")
        return prototype

    def _moving_average_update(
        self, prototype: Prototype, new_embedding: torch.Tensor
    ) -> Prototype:
        n = prototype.sample_count
        updated = (prototype.embedding * n + new_embedding) / (n + 1)
        prototype.embedding = updated
        prototype.sample_count += 1
        logger.debug(f"Moving average update for {prototype.prototype_id}")
        return prototype

    def _replacement_update(
        self, prototype: Prototype, new_embedding: torch.Tensor
    ) -> Prototype:
        prototype.embedding = new_embedding
        prototype.sample_count = 1
        logger.debug(f"Replacement update for {prototype.prototype_id}")
        return prototype

    def _weighted_update(
        self, prototype: Prototype, new_embedding: torch.Tensor, weight: float = 0.5
    ) -> Prototype:
        updated = (1.0 - weight) * prototype.embedding + weight * new_embedding
        prototype.embedding = updated
        prototype.sample_count += 1
        logger.debug(f"Weighted update for {prototype.prototype_id} (weight={weight})")
        return prototype

    def _adaptive_update(
        self, prototype: Prototype, new_embedding: torch.Tensor
    ) -> Prototype:
        dist = torch.nn.functional.pairwise_distance(
            prototype.embedding.unsqueeze(0), new_embedding.unsqueeze(0)
        )
        alpha = torch.sigmoid(-dist).item()
        updated = alpha * prototype.embedding + (1.0 - alpha) * new_embedding
        prototype.embedding = updated
        prototype.sample_count += 1
        logger.debug(
            f"Adaptive update for {prototype.prototype_id} (alpha={alpha:.4f})"
        )
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
            class_embs = embeddings[class_mask]
            if class_embs.size(0) > 0:
                avg_emb = class_embs.mean(dim=0)
                self.update(proto, avg_emb, **kwargs)
        return prototypes
