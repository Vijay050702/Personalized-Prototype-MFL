from __future__ import annotations

from typing import Any

import torch

from app.knowledge_transfer.similarity import Similarity
from app.personalization.validation import validate_confidence_range


class PersonalizedConfidence:
    def __init__(
        self,
        similarity_metric: str = "cosine",
        stability_window: int = 5,
    ):
        self._similarity = Similarity(metric=similarity_metric)
        self._stability_window = stability_window

    def estimate(
        self,
        base_confidence: float,
        prototype_variance: float | None = None,
        consistency: float | None = None,
        similarity_score: float | None = None,
        history_confidences: list[float] | None = None,
    ) -> float:
        factors: list[float] = []

        factors.append(max(0.0, min(1.0, base_confidence)))

        if prototype_variance is not None:
            var_factor = 1.0 / (1.0 + prototype_variance)
            factors.append(max(0.0, min(1.0, var_factor)))

        if consistency is not None:
            factors.append(max(0.0, min(1.0, consistency)))

        if similarity_score is not None:
            factors.append(max(0.0, min(1.0, similarity_score)))

        if history_confidences is not None and len(history_confidences) >= 2:
            stability = self._stability(history_confidences)
            factors.append(stability)

        if not factors:
            return base_confidence

        confidence = sum(factors) / len(factors)
        validate_confidence_range(confidence)
        return confidence

    def batch_estimate(
        self,
        base_confidences: list[float],
        prototype_variances: list[float | None] | None = None,
        consistencies: list[float | None] | None = None,
        similarity_scores: list[float | None] | None = None,
        history_list: list[list[float]] | None = None,
    ) -> list[float]:
        n = len(base_confidences)
        results: list[float] = []
        for i in range(n):
            var = prototype_variances[i] if prototype_variances else None
            con = consistencies[i] if consistencies else None
            sim = similarity_scores[i] if similarity_scores else None
            hist = history_list[i] if history_list else None
            results.append(
                self.estimate(
                    base_confidence=base_confidences[i],
                    prototype_variance=var,
                    consistency=con,
                    similarity_score=sim,
                    history_confidences=hist,
                )
            )
        return results

    def _stability(self, history: list[float]) -> float:
        if len(history) < 2:
            return 1.0
        recent = history[-self._stability_window :]
        vals = torch.tensor(recent, dtype=torch.float32)
        variance = vals.var().item() if len(recent) > 1 else 0.0
        return 1.0 / (1.0 + variance)
