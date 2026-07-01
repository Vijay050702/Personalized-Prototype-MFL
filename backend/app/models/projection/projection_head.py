from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.models.base.base_model import BaseModel


class ProjectionHead(BaseModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        normalize_output: bool = True,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._normalize_output = normalize_output

        dims = [input_dim]
        if num_layers > 1:
            h_dim = hidden_dim or input_dim
            for _ in range(num_layers - 1):
                dims.append(h_dim)
        dims.append(output_dim)

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.mlp(x)
        if self._normalize_output:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x
