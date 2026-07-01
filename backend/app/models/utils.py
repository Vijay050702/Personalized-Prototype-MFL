from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from app.core.logging import logger


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def log_model_summary(model: nn.Module, model_name: str | None = None) -> None:
    name = model_name or type(model).__name__
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    non_trainable = total - trainable
    logger.info(
        f"Model: {name} | "
        f"Total params: {total:,} | "
        f"Trainable: {trainable:,} | "
        f"Non-trainable: {non_trainable:,}"
    )
    logger.info(f"{'Module':<30} {'Params':<15} {'Trainable':<10}")
    logger.info("-" * 55)
    for name_param, module in model.named_children():
        p = count_parameters(module, trainable_only=False)
        t = count_parameters(module, trainable_only=True)
        logger.info(f"{name_param:<30} {p:<15,} {str(t != 0):<10}")


def get_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        if isinstance(device, str):
            return torch.device(device)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def estimate_memory_usage(
    model: nn.Module, input_shape: tuple[int, ...] | None = None
) -> dict[str, float]:
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_memory = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    total_model_memory = param_memory + grad_memory

    forward_memory = 0.0
    if input_shape is not None:
        sample_input = torch.zeros(input_shape)
        try:
            forward_memory = sample_input.numel() * sample_input.element_size() * 2
        except Exception:
            pass

    return {
        "parameters_mb": param_memory / (1024**2),
        "gradients_mb": grad_memory / (1024**2),
        "total_model_mb": total_model_memory / (1024**2),
        "estimate_forward_mb": forward_memory / (1024**2),
        "total_estimate_mb": (total_model_memory + forward_memory) / (1024**2),
    }


class Timer:
    def __init__(self, name: str = "", log_on_exit: bool = True):
        self._name = name
        self._log_on_exit = log_on_exit
        self._start: float | None = None
        self._elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            self._elapsed = time.perf_counter() - self._start
        if self._log_on_exit and self._name:
            logger.info(f"Timer [{self._name}]: {self._elapsed:.4f}s")

    @property
    def elapsed(self) -> float:
        return self._elapsed
