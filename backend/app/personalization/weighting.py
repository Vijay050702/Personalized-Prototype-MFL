from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.knowledge_transfer.similarity import Similarity
from app.personalization.validation import (
    validate_fusion_sources,
    validate_weights_sum_to_one,
)


class WeightCalculator:
    def __init__(
        self,
        strategy: str = "fixed",
        fixed_weights: dict[str, float] | None = None,
        similarity_metric: str = "cosine",
        temperature: float = 1.0,
    ):
        if strategy not in {
            "fixed",
            "confidence",
            "similarity",
            "adaptive",
            "learnable",
        }:
            raise ValueError(
                f"Unknown weighting strategy '{strategy}'. "
                f"Choose from: fixed, confidence, similarity, adaptive, learnable"
            )
        self._strategy = strategy
        self._fixed_weights = fixed_weights or {}
        self._similarity = Similarity(metric=similarity_metric)
        self._temperature = temperature

        if strategy == "learnable":
            self._learnable_weights = nn.Parameter(torch.ones(3) / 3.0)
        else:
            self._learnable_weights = None

    def compute(
        self,
        sources: list[str],
        confidences: dict[str, float] | None = None,
        embeddings: dict[str, torch.Tensor] | None = None,
        adaptive_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        validate_fusion_sources(sources)

        if self._strategy == "fixed":
            weights = self._fixed(sources)
        elif self._strategy == "confidence":
            if confidences is None:
                raise ValueError("Confidences required for confidence strategy")
            weights = self._confidence_weight(sources, confidences)
        elif self._strategy == "similarity":
            if embeddings is None:
                raise ValueError("Embeddings required for similarity strategy")
            weights = self._similarity_weight(sources, embeddings)
        elif self._strategy == "adaptive":
            if adaptive_weights is None:
                raise ValueError("Adaptive weights required for adaptive strategy")
            weights = self._adaptive(sources, adaptive_weights)
        elif self._strategy == "learnable":
            weights = self._learnable(sources)
        else:
            n = len(sources)
            weights = {s: 1.0 / n for s in sources}

        validate_weights_sum_to_one(weights, sources)
        return weights

    def _fixed(self, sources: list[str]) -> dict[str, float]:
        if self._fixed_weights:
            return {s: self._fixed_weights.get(s, 0.0) for s in sources}
        n = len(sources)
        return {s: 1.0 / n for s in sources}

    def _confidence_weight(
        self,
        sources: list[str],
        confidences: dict[str, float],
    ) -> dict[str, float]:
        vals = torch.tensor(
            [confidences.get(s, 0.0) for s in sources],
            dtype=torch.float32,
        )
        vals = F.softmax(vals / self._temperature, dim=0)
        return {s: vals[i].item() for i, s in enumerate(sources)}

    def _similarity_weight(
        self,
        sources: list[str],
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        if len(sources) == 1:
            return {sources[0]: 1.0}

        sim_matrix = torch.zeros(len(sources), len(sources))
        for i, s1 in enumerate(sources):
            for j, s2 in enumerate(sources):
                if i != j and s1 in embeddings and s2 in embeddings:
                    sim_matrix[i, j] = self._similarity.compute(
                        embeddings[s1], embeddings[s2]
                    )

        avg_sim = sim_matrix.mean(dim=1)
        vals = F.softmax(avg_sim / self._temperature, dim=0)
        return {s: vals[i].item() for i, s in enumerate(sources)}

    def _adaptive(
        self,
        sources: list[str],
        adaptive_weights: dict[str, float],
    ) -> dict[str, float]:
        return {s: adaptive_weights.get(s, 0.0) for s in sources}

    def _learnable(self, sources: list[str]) -> dict[str, float]:
        if self._learnable_weights is None:
            n = len(sources)
            return {s: 1.0 / n for s in sources}
        vals = F.softmax(
            self._learnable_weights[: len(sources)] / self._temperature, dim=0
        )
        return {s: vals[i].item() for i, s in enumerate(sources)}

    @property
    def strategy(self) -> str:
        return self._strategy

    def to_config(self) -> dict[str, Any]:
        return {
            "strategy": self._strategy,
            "fixed_weights": self._fixed_weights,
            "similarity_metric": self._similarity.metric,
            "temperature": self._temperature,
        }
