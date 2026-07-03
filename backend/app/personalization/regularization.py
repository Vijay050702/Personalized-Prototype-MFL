from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.knowledge_transfer.similarity import Similarity
from app.personalization.validation import validate_shape_match


class PrototypeConsistencyRegularization(nn.Module):
    def __init__(self, similarity_metric: str = "cosine"):
        super().__init__()
        self._similarity = Similarity(metric=similarity_metric)

    def forward(
        self,
        personalized: torch.Tensor,
        local: torch.Tensor | None,
        global_p: torch.Tensor | None,
        lam_local: float = 1.0,
        lam_global: float = 1.0,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)
        if local is not None:
            validate_shape_match(personalized, local, "personalized", "local")
            local_sim = self._similarity.compute(personalized, local)
            loss = loss + lam_local * (1.0 - local_sim)

        if global_p is not None:
            validate_shape_match(personalized, global_p, "personalized", "global")
            global_sim = self._similarity.compute(personalized, global_p)
            loss = loss + lam_global * (1.0 - global_sim)

        return loss


class FusionSmoothnessRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        fusion_weights: torch.Tensor,
    ) -> torch.Tensor:
        if fusion_weights.numel() <= 1:
            return torch.tensor(0.0)
        return torch.var(fusion_weights)


class PrototypeStabilityRegularization(nn.Module):
    def __init__(self, similarity_metric: str = "cosine"):
        super().__init__()
        self._similarity = Similarity(metric=similarity_metric)

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor | None,
    ) -> torch.Tensor:
        if previous is None:
            return torch.tensor(0.0)
        validate_shape_match(current, previous, "current", "previous")
        sim = self._similarity.compute(current, previous)
        return 1.0 - sim


class TemporalConsistencyRegularization(nn.Module):
    def __init__(self, similarity_metric: str = "cosine"):
        super().__init__()
        self._similarity = Similarity(metric=similarity_metric)
        self._previous: dict[str, torch.Tensor] = {}

    def forward(
        self,
        current: torch.Tensor,
        key: str = "default",
    ) -> torch.Tensor:
        if key not in self._previous:
            self._previous[key] = current.clone().detach()
            return torch.tensor(0.0)
        prev = self._previous[key]
        sim = self._similarity.compute(current, prev)
        self._previous[key] = current.clone().detach()
        return 1.0 - sim
