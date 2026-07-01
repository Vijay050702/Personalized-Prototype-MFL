from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.models.base.base_model import BaseModel
from app.models.fusion.fusion_strategy import FusionStrategy
from app.data.modality import MODALITY_KEYS


class AttentionFusion(BaseModel, FusionStrategy):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim

        self.modality_projections = nn.ModuleDict(
            {mod: nn.Linear(embed_dim, embed_dim) for mod in MODALITY_KEYS}
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        projected: list[torch.Tensor] = []
        mod_order: list[str] = []
        for mod in MODALITY_KEYS:
            emb = embeddings.get(mod)
            if emb is None:
                continue
            proj = self.modality_projections[mod](emb)
            projected.append(proj)
            mod_order.append(mod)

        if not projected:
            raise ValueError("No valid embeddings to fuse")

        if len(projected) == 1:
            return projected[0]

        stacked = torch.stack(projected, dim=1)
        query = stacked[:, :1, :]
        key = stacked
        value = stacked

        key_padding_mask: torch.Tensor | None = None
        if modality_mask is not None:
            mod_indices = [i for i, mod in enumerate(MODALITY_KEYS) if mod in mod_order]
            present = modality_mask[:, mod_indices]
            key_padding_mask = ~present

        attn_out, _ = self.cross_attn(
            query, key, value, key_padding_mask=key_padding_mask
        )
        attn_out = self.layer_norm(attn_out + query)
        attn_out = self.dropout(attn_out)
        fused = attn_out.squeeze(1)

        return fused

    @property
    def output_dim(self) -> int:
        return self._embed_dim
