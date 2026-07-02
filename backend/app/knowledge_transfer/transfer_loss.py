from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.knowledge_transfer.similarity import Similarity
from app.knowledge_transfer.validation import validate_shape_match


class TransferLoss(nn.Module):
    def __init__(
        self,
        alignment_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        similarity_weight: float = 0.5,
        consistency_weight: float = 0.5,
        similarity_metric: str = "cosine",
    ):
        super().__init__()
        self._alignment_weight = alignment_weight
        self._reconstruction_weight = reconstruction_weight
        self._similarity_weight = similarity_weight
        self._consistency_weight = consistency_weight
        self._similarity = Similarity(metric=similarity_metric)

    @property
    def alignment_weight(self) -> float:
        return self._alignment_weight

    @property
    def reconstruction_weight(self) -> float:
        return self._reconstruction_weight

    @property
    def similarity_weight(self) -> float:
        return self._similarity_weight

    @property
    def consistency_weight(self) -> float:
        return self._consistency_weight

    def alignment_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        validate_shape_match(source, target)
        sim = self._similarity.compute(source, target)
        aligned = (labels > 0).float()
        return F.mse_loss(sim, aligned)

    def reconstruction_loss(
        self,
        input_embedding: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        validate_shape_match(input_embedding, reconstructed)
        return F.mse_loss(reconstructed, input_embedding)

    def similarity_preservation_loss(
        self,
        original_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        translated_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if not original_pairs or not translated_pairs:
            return torch.tensor(0.0)

        original_sims: list[torch.Tensor] = []
        translated_sims: list[torch.Tensor] = []
        for (a, b), (c, d) in zip(original_pairs, translated_pairs):
            original_sims.append(self._similarity.compute(a, b).unsqueeze(0))
            translated_sims.append(self._similarity.compute(c, d).unsqueeze(0))

        orig = torch.cat(original_sims)
        trans = torch.cat(translated_sims)
        return F.mse_loss(trans, orig.detach())

    def consistency_loss(
        self,
        forward_result: torch.Tensor,
        backward_result: torch.Tensor,
    ) -> torch.Tensor:
        validate_shape_match(forward_result, backward_result)
        return F.mse_loss(forward_result, backward_result)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor,
        reconstructed: torch.Tensor | None = None,
        original_pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        translated_pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        forward_cycle: torch.Tensor | None = None,
        backward_cycle: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        losses["alignment"] = (
            self.alignment_loss(source, target, labels) * self._alignment_weight
        )

        if reconstructed is not None:
            losses["reconstruction"] = (
                self.reconstruction_loss(source, reconstructed)
                * self._reconstruction_weight
            )

        if original_pairs is not None and translated_pairs is not None:
            losses["similarity"] = (
                self.similarity_preservation_loss(original_pairs, translated_pairs)
                * self._similarity_weight
            )

        if forward_cycle is not None and backward_cycle is not None:
            losses["consistency"] = (
                self.consistency_loss(forward_cycle, backward_cycle)
                * self._consistency_weight
            )

        total = sum(losses.values())
        losses["total"] = total
        return losses

    def to_config(self) -> dict[str, Any]:
        return {
            "alignment_weight": self._alignment_weight,
            "reconstruction_weight": self._reconstruction_weight,
            "similarity_weight": self._similarity_weight,
            "consistency_weight": self._consistency_weight,
            "similarity_metric": self._similarity.metric,
        }
