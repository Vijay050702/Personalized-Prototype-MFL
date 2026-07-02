from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.prototypes.prototype import Prototype


class PrototypeCompactnessLoss(nn.Module):
    def __init__(self, margin: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self._margin = margin
        self._reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        prototypes: list[Prototype],
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        for i, label in enumerate(labels):
            matching = [p for p in prototypes if p.class_id == label.item()]
            if not matching:
                continue
            proto_emb = matching[0].embedding.to(embeddings.device)
            dist = F.pairwise_distance(
                embeddings[i].unsqueeze(0), proto_emb.unsqueeze(0)
            ).squeeze()
            loss = loss + dist
            count += 1
        if count == 0:
            return loss
        loss = loss / count
        return loss


class PrototypeSeparationLoss(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self._margin = margin
        self._reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        prototypes: list[Prototype],
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        for i, label in enumerate(labels):
            other_protos = [p for p in prototypes if p.class_id != label.item()]
            if not other_protos:
                continue
            other_embs = torch.stack([p.embedding for p in other_protos]).to(
                embeddings.device
            )
            dists = torch.cdist(embeddings[i].unsqueeze(0), other_embs, p=2.0)
            margin_dist = torch.clamp(self._margin - dists, min=0.0)
            loss = loss + margin_dist.sum()
            count += 1
        if count == 0:
            return loss
        loss = loss / count
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, alpha: float = 0.5):
        super().__init__()
        self._num_classes = num_classes
        self._embedding_dim = embedding_dim
        self._alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        prototypes: list[Prototype] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size = embeddings.size(0)
        centers_batch = self.centers[labels]
        loss = (embeddings - centers_batch).pow(2).sum(dim=1).mean()
        return loss


class PrototypeConsistencyLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self._temperature = temperature

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        student_norm = F.normalize(student_embeddings, p=2, dim=1)
        teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1)
        sim = (student_norm * teacher_norm).sum(dim=1)
        loss = 1.0 - sim.mean()
        return loss


class PrototypeDiversityLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self._margin = margin

    def forward(
        self,
        prototypes: list[Prototype],
        **kwargs: Any,
    ) -> torch.Tensor:
        if len(prototypes) < 2:
            return torch.tensor(0.0)

        embeddings = torch.stack([p.embedding for p in prototypes])
        normalized = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = normalized @ normalized.T

        mask = torch.ones_like(sim_matrix) - torch.eye(
            len(prototypes), device=sim_matrix.device
        )
        pair_sims = sim_matrix * mask
        loss = torch.clamp(pair_sims - self._margin, min=0.0).sum()
        n_pairs = mask.sum()
        if n_pairs > 0:
            loss = loss / n_pairs
        return loss
