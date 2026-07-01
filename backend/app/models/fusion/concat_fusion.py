from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.models.base.base_model import BaseModel
from app.models.fusion.fusion_strategy import FusionStrategy


class ConcatFusion(BaseModel, FusionStrategy):
    def __init__(
        self,
        projected_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._projected_dim = projected_dim
        self._dropout_module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def output_dim(self) -> int | None:
        return self._projected_dim

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        valid_embeddings = [emb for emb in embeddings.values() if emb is not None]
        if not valid_embeddings:
            raise ValueError("No valid embeddings to fuse")
        x = torch.cat(valid_embeddings, dim=-1)
        x = self._dropout_module(x)
        return x
