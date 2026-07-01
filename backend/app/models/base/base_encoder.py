from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from app.models.base.base_model import BaseModel


class BaseEncoder(BaseModel):
    def __init__(self, embedding_dim: int, output_dim: int | None = None):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._output_dim = output_dim or embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.encode(x, **kwargs)
