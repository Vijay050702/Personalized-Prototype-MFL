from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.knowledge_transfer.validation import validate_mapping_dimensions


class AlignmentNetwork(nn.Module):
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        mapper_type: str = "linear",
    ):
        super().__init__()
        validate_mapping_dimensions(source_dim, target_dim)

        if mapper_type not in {"linear", "mlp"}:
            raise ValueError(
                f"Unknown mapper_type '{mapper_type}'. Choose from: linear, mlp"
            )
        if activation not in {"relu", "tanh", "gelu"}:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from: relu, tanh, gelu"
            )

        self._source_dim = source_dim
        self._target_dim = target_dim
        self._mapper_type = mapper_type

        act_fn = self._get_activation(activation)

        if mapper_type == "linear":
            self._network = nn.Linear(source_dim, target_dim)
        else:
            layers: list[nn.Module] = []
            dims = [source_dim]
            if hidden_dims:
                dims.extend(hidden_dims)
            dims.append(target_dim)
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(act_fn())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self._network = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name: str) -> type[nn.Module]:
        mapping: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        return mapping[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network(x)

    @property
    def source_dim(self) -> int:
        return self._source_dim

    @property
    def target_dim(self) -> int:
        return self._target_dim

    @property
    def mapper_type(self) -> str:
        return self._mapper_type

    def to_config(self) -> dict[str, Any]:
        return {
            "source_dim": self._source_dim,
            "target_dim": self._target_dim,
            "mapper_type": self._mapper_type,
        }
