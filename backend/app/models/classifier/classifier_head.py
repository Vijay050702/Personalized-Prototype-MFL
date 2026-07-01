from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.models.base.base_model import BaseModel


class ClassifierHead(BaseModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes

        layers: list[nn.Module] = []
        in_dim = input_dim
        if hidden_dims:
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
