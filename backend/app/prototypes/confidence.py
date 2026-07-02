from __future__ import annotations

from typing import Any

import torch

from app.prototypes.prototype import Prototype
from app.prototypes.similarity import SimilarityEngine


class ConfidenceEstimator:
    def __init__(
        self,
        similarity_engine: SimilarityEngine | None = None,
        base_threshold: float = 0.5,
    ):
        self._similarity = similarity_engine or SimilarityEngine(metric="cosine")
        self._base_threshold = base_threshold

    def estimate(
        self,
        prototype: Prototype,
        embedding: torch.Tensor | None = None,
    ) -> float:
        factors: list[float] = []

        factors.append(self._sample_count_factor(prototype))
        factors.append(self._base_confidence_factor(prototype))

        if embedding is not None:
            factors.append(self._distance_factor(prototype, embedding))

        if len(factors) == 0:
            return prototype.confidence

        return float(torch.tensor(factors).mean())

    def _sample_count_factor(self, prototype: Prototype) -> float:
        return min(1.0, prototype.sample_count / 100.0)

    def _base_confidence_factor(self, prototype: Prototype) -> float:
        return prototype.confidence

    def _distance_factor(self, prototype: Prototype, embedding: torch.Tensor) -> float:
        sim = self._similarity.similarity(prototype.embedding, embedding)
        return float(torch.sigmoid(sim * 5.0 - 2.5))

    def batch_estimate(
        self,
        prototypes: list[Prototype],
        embeddings: torch.Tensor | None = None,
    ) -> list[float]:
        confs = []
        for i, proto in enumerate(prototypes):
            emb = embeddings[i] if embeddings is not None else None
            confs.append(self.estimate(proto, emb))
        return confs

    def stability_score(
        self, prototype: Prototype, history: list[dict[str, Any]]
    ) -> float:
        if len(history) < 2:
            return 1.0
        confidences = [h.get("confidence", 0.0) for h in history]
        if len(confidences) <= 1:
            return 1.0
        variance = float(torch.tensor(confidences, dtype=torch.float32).var())
        return 1.0 / (1.0 + variance)

    def normalized_confidence(
        self, raw_score: float, min_val: float = 0.0, max_val: float = 1.0
    ) -> float:
        if max_val <= min_val:
            return 0.0
        return max(0.0, min(1.0, (raw_score - min_val) / (max_val - min_val)))
