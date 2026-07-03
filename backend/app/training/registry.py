from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn
import torch.optim as optim


class TrainingRegistry:
    _optimizers: dict[str, type[optim.Optimizer]] = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
    }
    _schedulers: dict[str, type] = {
        "step_lr": optim.lr_scheduler.StepLR,
        "cosine_annealing": optim.lr_scheduler.CosineAnnealingLR,
        "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau,
    }
    _losses: dict[str, Callable[..., nn.Module]] = {}
    _hooks: dict[str, type] = {}
    _callbacks: dict[str, type] = {}
    _metrics: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register_optimizer(
        cls, name: str, optimizer_cls: type[optim.Optimizer]
    ) -> None:
        cls._optimizers[name] = optimizer_cls

    @classmethod
    def get_optimizer(cls, name: str) -> type[optim.Optimizer]:
        if name not in cls._optimizers:
            raise ValueError(
                f"Unknown optimizer '{name}'. Available: {list(cls._optimizers.keys())}"
            )
        return cls._optimizers[name]

    @classmethod
    def list_optimizers(cls) -> list[str]:
        return list(cls._optimizers.keys())

    @classmethod
    def register_scheduler(cls, name: str, scheduler_cls: type) -> None:
        cls._schedulers[name] = scheduler_cls

    @classmethod
    def get_scheduler(cls, name: str) -> type:
        if name not in cls._schedulers:
            raise ValueError(
                f"Unknown scheduler '{name}'. Available: {list(cls._schedulers.keys())}"
            )
        return cls._schedulers[name]

    @classmethod
    def list_schedulers(cls) -> list[str]:
        return list(cls._schedulers.keys())

    @classmethod
    def register_loss(cls, name: str, loss_fn: Callable[..., nn.Module]) -> None:
        cls._losses[name] = loss_fn

    @classmethod
    def get_loss(cls, name: str) -> Callable[..., nn.Module]:
        if name not in cls._losses:
            raise ValueError(
                f"Unknown loss '{name}'. Available: {list(cls._losses.keys())}"
            )
        return cls._losses[name]

    @classmethod
    def list_losses(cls) -> list[str]:
        return list(cls._losses.keys())

    @classmethod
    def register_metric(cls, name: str, metric_fn: Callable[..., Any]) -> None:
        cls._metrics[name] = metric_fn

    @classmethod
    def get_metric(cls, name: str) -> Callable[..., Any]:
        if name not in cls._metrics:
            raise ValueError(
                f"Unknown metric '{name}'. Available: {list(cls._metrics.keys())}"
            )
        return cls._metrics[name]

    @classmethod
    def list_metrics(cls) -> list[str]:
        return list(cls._metrics.keys())
