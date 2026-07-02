from __future__ import annotations

import math
from typing import Any

import torch

from app.federated.divergence import DivergenceCalculator
from app.federated.models import ClientPrototypePackage


class AdaptiveWeightCalculator:
    def __init__(
        self,
        temperature: float = 1.0,
        divergence_weight: float = 0.5,
        sample_weight: float = 0.3,
        completeness_weight: float = 0.2,
        divergence_calculator: DivergenceCalculator | None = None,
    ):
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")
        self._temperature = temperature
        self._divergence_weight = divergence_weight
        self._sample_weight = sample_weight
        self._completeness_weight = completeness_weight
        self._divergence = divergence_calculator or DivergenceCalculator(
            metric="cosine"
        )

    @property
    def temperature(self) -> float:
        return self._temperature

    def compute_weight(
        self,
        package: ClientPrototypePackage,
        completeness_ratio: float,
        divergence_score: float | None = None,
    ) -> float:
        sample_factor = self._sample_factor(package.sample_count)
        completeness_factor = completeness_ratio
        if divergence_score is not None:
            divergence_factor = self._divergence_factor(divergence_score)
        else:
            divergence_factor = 1.0

        raw = (
            self._sample_weight * sample_factor
            + self._completeness_weight * completeness_factor
            + self._divergence_weight * divergence_factor
        )
        return raw / (
            self._sample_weight + self._completeness_weight + self._divergence_weight
        )

    def compute_normalized_weights(
        self,
        packages: list[ClientPrototypePackage],
        completeness_ratios: dict[str, float],
        divergences: dict[str, float] | None = None,
    ) -> list[float]:
        if not packages:
            return []

        raw_weights: list[float] = []
        for pkg in packages:
            cr = completeness_ratios.get(pkg.client_id, 1.0)
            dv = divergences.get(pkg.client_id) if divergences else None
            raw_weights.append(self.compute_weight(pkg, cr, dv))

        return self._softmax_normalize(raw_weights)

    def _sample_factor(self, sample_count: int) -> float:
        return 1.0 - math.exp(-sample_count / 100.0)

    def _divergence_factor(self, divergence: float) -> float:
        return math.exp(-divergence / self._temperature)

    def _softmax_normalize(self, weights: list[float]) -> list[float]:
        if not weights:
            return []
        t = torch.tensor(weights, dtype=torch.float32)
        scaled = t / self._temperature
        normalized = torch.softmax(scaled, dim=0)
        return normalized.tolist()

    def to_config(self) -> dict[str, Any]:
        return {
            "temperature": self._temperature,
            "divergence_weight": self._divergence_weight,
            "sample_weight": self._sample_weight,
            "completeness_weight": self._completeness_weight,
            "divergence_metric": self._divergence.metric,
        }
