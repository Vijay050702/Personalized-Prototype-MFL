from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.knowledge_transfer.similarity import Similarity
from app.personalization.validation import validate_shape_match


class FusionLoss(nn.Module):
    def __init__(self, similarity_metric: str = "cosine"):
        super().__init__()
        self._similarity = Similarity(metric=similarity_metric)

    def forward(
        self,
        fused: torch.Tensor,
        targets: list[torch.Tensor],
        weights: list[float],
    ) -> torch.Tensor:
        if not targets or not weights:
            raise ValueError("Targets and weights must not be empty")
        if len(targets) != len(weights):
            raise ValueError(
                f"Number of targets ({len(targets)}) must match "
                f"number of weights ({len(weights)})"
            )

        losses: list[torch.Tensor] = []
        for target, w in zip(targets, weights):
            validate_shape_match(fused, target, "fused", "target")
            sim = self._similarity.compute(fused, target)
            losses.append(w * (1.0 - sim))

        return sum(losses) / len(losses)


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        validate_shape_match(prediction, target, "prediction", "target")
        return F.mse_loss(prediction, target)


class PersonalizationLoss(nn.Module):
    def __init__(self, similarity_metric: str = "cosine"):
        super().__init__()
        self._similarity = Similarity(metric=similarity_metric)

    def forward(
        self,
        personalized: torch.Tensor,
        local: torch.Tensor | None = None,
        global_p: torch.Tensor | None = None,
        cross_modal: torch.Tensor | None = None,
        lam_local: float = 1.0,
        lam_global: float = 1.0,
        lam_cross: float = 1.0,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)
        count = 0

        if local is not None:
            validate_shape_match(personalized, local, "personalized", "local")
            sim = self._similarity.compute(personalized, local)
            loss = loss + lam_local * (1.0 - sim)
            count += 1

        if global_p is not None:
            validate_shape_match(personalized, global_p, "personalized", "global")
            sim = self._similarity.compute(personalized, global_p)
            loss = loss + lam_global * (1.0 - sim)
            count += 1

        if cross_modal is not None:
            validate_shape_match(
                personalized, cross_modal, "personalized", "cross_modal"
            )
            sim = self._similarity.compute(personalized, cross_modal)
            loss = loss + lam_cross * (1.0 - sim)
            count += 1

        if count == 0:
            return torch.tensor(0.0)

        return loss / count


class PrototypeRegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        prototype: torch.Tensor,
        norm_weight: float = 1e-4,
    ) -> torch.Tensor:
        return norm_weight * prototype.norm(p=2)


class AdaptiveWeightingLoss(nn.Module):
    def __init__(self, regularization_strength: float = 0.01):
        super().__init__()
        self._reg_strength = regularization_strength

    def forward(
        self,
        weights: torch.Tensor,
        source_losses: torch.Tensor,
    ) -> torch.Tensor:
        if weights.numel() != source_losses.numel():
            raise ValueError(
                f"Weights ({weights.numel()}) and source_losses "
                f"({source_losses.numel()}) must have the same number of elements"
            )
        weighted_loss = (weights * source_losses).sum()
        entropy = -(weights * torch.log(weights + 1e-8)).sum()
        return weighted_loss - self._reg_strength * entropy


class CombinedPersonalizationLoss(nn.Module):
    def __init__(
        self,
        fusion_weight: float = 1.0,
        consistency_weight: float = 1.0,
        personalization_weight: float = 1.0,
        regularization_weight: float = 0.1,
        adaptive_weighting_weight: float = 0.1,
        similarity_metric: str = "cosine",
    ):
        super().__init__()
        self.fusion_loss = FusionLoss(similarity_metric=similarity_metric)
        self.consistency_loss = ConsistencyLoss()
        self.personalization_loss = PersonalizationLoss(
            similarity_metric=similarity_metric
        )
        self.regularization_loss = PrototypeRegularizationLoss()
        self.adaptive_weighting_loss = AdaptiveWeightingLoss()

        self._fusion_weight = fusion_weight
        self._consistency_weight = consistency_weight
        self._personalization_weight = personalization_weight
        self._regularization_weight = regularization_weight
        self._adaptive_weighting_weight = adaptive_weighting_weight

    def forward(
        self,
        fused: torch.Tensor,
        targets: list[torch.Tensor],
        target_weights: list[float],
        local: torch.Tensor | None = None,
        global_p: torch.Tensor | None = None,
        cross_modal: torch.Tensor | None = None,
        consistency_target: torch.Tensor | None = None,
        fusion_weights: torch.Tensor | None = None,
        source_losses: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        losses["fusion"] = (
            self.fusion_loss(fused, targets, target_weights) * self._fusion_weight
        )

        if consistency_target is not None:
            losses["consistency"] = (
                self.consistency_loss(fused, consistency_target)
                * self._consistency_weight
            )

        losses["personalization"] = (
            self.personalization_loss(
                personalized=fused,
                local=local,
                global_p=global_p,
                cross_modal=cross_modal,
            )
            * self._personalization_weight
        )

        losses["regularization"] = (
            self.regularization_loss(fused) * self._regularization_weight
        )

        if fusion_weights is not None and source_losses is not None:
            losses["adaptive_weighting"] = (
                self.adaptive_weighting_loss(fusion_weights, source_losses)
                * self._adaptive_weighting_weight
            )

        total = sum(losses.values())
        losses["total"] = total
        return losses

    def to_config(self) -> dict[str, Any]:
        return {
            "fusion_weight": self._fusion_weight,
            "consistency_weight": self._consistency_weight,
            "personalization_weight": self._personalization_weight,
            "regularization_weight": self._regularization_weight,
            "adaptive_weighting_weight": self._adaptive_weighting_weight,
        }
