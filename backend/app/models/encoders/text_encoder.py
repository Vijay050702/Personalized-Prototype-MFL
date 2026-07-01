from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from app.models.base.base_encoder import BaseEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TextEncoder(BaseEncoder):
    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_dim: int = 256,
        output_dim: int | None = None,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        padding_idx: int = 0,
        normalize: bool = False,
    ):
        super().__init__(embedding_dim=embedding_dim, output_dim=output_dim)
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._max_seq_length = max_seq_length
        self._padding_idx = padding_idx
        self._normalize = normalize

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.pos_encoder = PositionalEncoding(
            embedding_dim, max_len=max_seq_length, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_proj: nn.Module | None = None
        if embedding_dim != self._output_dim:
            self.output_proj = nn.Linear(embedding_dim, self._output_dim)

    def encode(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        src_key_padding_mask: torch.Tensor | None = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        elif self._padding_idx is not None:
            src_key_padding_mask = x.abs().sum(dim=-1) == 0
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=1)
        if self.output_proj is not None:
            x = self.output_proj(x)
        if self._normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        return self.encode(x, mask=mask, **kwargs)
