from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class FusionStrategy(ABC):
    @abstractmethod
    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass
