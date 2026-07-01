from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from app.core.logging import logger


class BaseModel(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._device: torch.device = torch.device("cpu")

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def to(self, device: torch.device | str | None = None, **kwargs: Any) -> BaseModel:
        if isinstance(device, str):
            device = torch.device(device)
        if device is not None:
            self._device = device
        return super().to(device=device, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._device

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def freeze_module(self, module_name: str) -> None:
        module = getattr(self, module_name, None)
        if module is None:
            raise AttributeError(
                f"Module '{module_name}' not found in {type(self).__name__}"
            )
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module_name: str) -> None:
        module = getattr(self, module_name, None)
        if module is None:
            raise AttributeError(
                f"Module '{module_name}' not found in {type(self).__name__}"
            )
        for param in module.parameters():
            param.requires_grad = True

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str, strict: bool = True) -> None:
        state_dict = torch.load(path, map_location=self._device, weights_only=True)
        self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Model loaded from {path}")

    def get_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        return total_norm**0.5

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(params={self.num_parameters:,}, "
            f"trainable={self.num_trainable_parameters:,})"
        )
