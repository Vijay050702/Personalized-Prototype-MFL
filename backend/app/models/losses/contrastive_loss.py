from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean",
        margin: float | None = None,
    ):
        super().__init__()
        self._temperature = temperature
        self._reduction = reduction
        self._margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = embeddings @ embeddings.T / self._temperature

        if labels is not None:
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            pos_mask = mask - torch.eye(
                labels.size(0), device=labels.device, dtype=torch.float
            )
            neg_mask = 1.0 - mask

            exp_sim = torch.exp(sim_matrix)
            pos_sum = (exp_sim * pos_mask).sum(dim=1)
            neg_sum = (exp_sim * neg_mask).sum(dim=1)
            total = pos_sum + neg_sum

            pos_count = pos_mask.sum(dim=1)
            valid = pos_count > 0
            log_prob = torch.log(pos_sum / (total + 1e-8) + 1e-8)
            loss = -(log_prob * valid.float()).sum() / (valid.sum() + 1e-8)
        else:
            labels = torch.arange(embeddings.size(0), device=embeddings.device)
            criterion = nn.CrossEntropyLoss(reduction=self._reduction)
            loss = criterion(sim_matrix, labels)

        if self._margin is not None:
            loss = torch.clamp(loss, min=self._margin)

        return loss


class EmbeddingSimilarityLoss(nn.Module):
    def __init__(
        self,
        similarity_type: str = "cosine",
        reduction: str = "mean",
    ):
        super().__init__()
        self._similarity_type = similarity_type
        self._reduction = reduction

        if similarity_type == "cosine":
            self.sim_fn = nn.CosineEmbeddingLoss(reduction=reduction)
        elif similarity_type == "mse":
            self.sim_fn = nn.MSELoss(reduction=reduction)
        elif similarity_type == "l1":
            self.sim_fn = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown similarity_type: {similarity_type}")

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._similarity_type == "cosine":
            if targets is None:
                targets = torch.ones(embeddings_a.size(0), device=embeddings_a.device)
            return self.sim_fn(embeddings_a, embeddings_b, targets)
        return self.sim_fn(embeddings_a, embeddings_b)
