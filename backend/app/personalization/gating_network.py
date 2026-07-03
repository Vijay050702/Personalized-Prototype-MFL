from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_sources: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        normalization: str = "softmax",
        temperature: float = 1.0,
    ):
        super().__init__()
        if normalization not in {"softmax", "sigmoid"}:
            raise ValueError(
                f"Unknown normalization '{normalization}'. "
                f"Choose from: softmax, sigmoid"
            )
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self._num_sources = num_sources
        self._normalization = normalization
        self._temperature = temperature

        dims = [input_dim] + (hidden_dims or [32]) + [num_sources]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._net(x)
        if self._normalization == "softmax":
            weights = F.softmax(logits / self._temperature, dim=-1)
        else:
            weights = torch.sigmoid(logits / self._temperature)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights

    @property
    def num_sources(self) -> int:
        return self._num_sources

    @property
    def normalization(self) -> str:
        return self._normalization

    @property
    def temperature(self) -> float:
        return self._temperature

    def to_config(self) -> dict[str, Any]:
        return {
            "input_dim": self._net[0].in_features,
            "num_sources": self._num_sources,
            "normalization": self._normalization,
            "temperature": self._temperature,
        }
