from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")
        self._temperature = temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negatives_norm = F.normalize(negatives, p=2, dim=1)

        pos_sim = (anchor_norm * positive_norm).sum(dim=1) / self._temperature
        neg_sim = anchor_norm @ negatives_norm.T / self._temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)

    def to_config(self) -> dict[str, float]:
        return {"temperature": self._temperature}


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        if margin < 0.0:
            raise ValueError(f"Margin must be >= 0, got {margin}")
        self._margin = margin

    @property
    def margin(self) -> float:
        return self._margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        neg_dist = (anchor - negative).pow(2).sum(dim=1)
        losses = F.relu(pos_dist - neg_dist + self._margin)
        return losses.mean()

    def to_config(self) -> dict[str, float]:
        return {"margin": self._margin}


class ContrastiveAlignmentLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        if margin < 0.0:
            raise ValueError(f"Margin must be >= 0, got {margin}")
        self._margin = margin

    @property
    def margin(self) -> float:
        return self._margin

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = embeddings_a.device
        batch_size = embeddings_a.size(0)

        sim = embeddings_a @ embeddings_b.T
        dist = 1.0 - sim

        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos_mask = label_eq.float()
        neg_mask = (~label_eq).float()

        pos_loss = (pos_mask * dist).sum() / (pos_mask.sum() + 1e-8)
        neg_loss = (neg_mask * F.relu(self._margin - dist)).sum() / (
            neg_mask.sum() + 1e-8
        )
        return pos_loss + neg_loss

    def to_config(self) -> dict[str, float]:
        return {"margin": self._margin}
