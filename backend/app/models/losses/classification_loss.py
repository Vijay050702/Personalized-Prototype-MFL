from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self._reduction = reduction
        self._label_smoothing = label_smoothing
        self._class_weights = class_weights

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.ce(logits, targets)
