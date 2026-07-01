from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.models.base.base_encoder import BaseEncoder


class AudioEncoder(BaseEncoder):
    def __init__(
        self,
        embedding_dim: int = 256,
        output_dim: int | None = None,
        in_channels: int = 1,
        base_filters: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        normalize: bool = False,
    ):
        super().__init__(embedding_dim=embedding_dim, output_dim=output_dim)
        self._pooling = pooling
        self._normalize = normalize

        layers: list[nn.Module] = []
        in_ch = in_channels
        for i in range(num_layers):
            out_ch = base_filters * (2 ** min(i, 3))
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Linear(in_ch, self._output_dim)

    def encode(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.cnn(x)
        if self._pooling == "mean":
            x = x.mean(dim=-1)
        elif self._pooling == "max":
            x = x.max(dim=-1).values
        else:
            x = x.mean(dim=-1)
        x = self.projection(x)
        if self._normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.encode(x, **kwargs)
