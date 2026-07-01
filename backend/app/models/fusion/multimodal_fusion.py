from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.data.modality import MODALITY_KEYS, NUM_MODALITIES
from app.models.base.base_model import BaseModel
from app.models.fusion.concat_fusion import ConcatFusion
from app.models.fusion.attention_fusion import AttentionFusion
from app.models.fusion.fusion_strategy import FusionStrategy


class MultimodalFusion(BaseModel):
    def __init__(
        self,
        embed_dim: int = 512,
        strategy: str = "concat",
        num_heads: int = 4,
        dropout: float = 0.1,
        projection_dim: int | None = None,
    ):
        super().__init__()
        self._strategy_name = strategy
        self._embed_dim = embed_dim

        if strategy == "concat":
            self.fusion: FusionStrategy = ConcatFusion(
                projected_dim=projection_dim, dropout=dropout
            )
        elif strategy == "attention":
            self.fusion = AttentionFusion(
                embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
            )
        elif strategy == "weighted":
            self.fusion = WeightedFusion(embed_dim=embed_dim, dropout=dropout)
        else:
            raise ValueError(
                f"Unknown fusion strategy: {strategy}. Choose 'concat', 'attention', or 'weighted'."
            )

        self._projection: nn.Module | None = None
        if projection_dim is not None and strategy != "concat":
            self._projection = nn.Linear(embed_dim, projection_dim)

        self._modality_projectors = nn.ModuleDict(
            {mod: nn.Identity() for mod in MODALITY_KEYS}
        )

    @property
    def output_dim(self) -> int:
        if hasattr(self.fusion, "output_dim") and self.fusion.output_dim is not None:
            return self.fusion.output_dim  # type: ignore
        return self._embed_dim

    def set_modality_projector(self, modality: str, projector: nn.Module) -> None:
        self._modality_projectors[modality] = projector

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        projected = {}
        for mod in MODALITY_KEYS:
            emb = embeddings.get(mod)
            if emb is not None:
                projected[mod] = self._modality_projectors[mod](emb)
        fused = self.fusion.forward(projected, modality_mask=modality_mask, **kwargs)
        if self._projection is not None:
            fused = self._projection(fused)
        return fused


class WeightedFusion(BaseModel, FusionStrategy):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self.weights = nn.Parameter(torch.ones(NUM_MODALITIES))
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self._embed_dim

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        valid_mods = [m for m in MODALITY_KEYS if embeddings.get(m) is not None]
        if not valid_mods:
            raise ValueError("No valid embeddings to fuse")

        stacked = torch.stack([embeddings[m] for m in valid_mods], dim=0)
        weight_values = F.softmax(self.weights[: len(valid_mods)], dim=0)
        weight_values = weight_values.view(-1, 1, 1)

        fused = (stacked * weight_values).sum(dim=0)
        fused = self.dropout(fused)
        return fused
