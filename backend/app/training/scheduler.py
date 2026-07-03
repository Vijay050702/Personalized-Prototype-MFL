from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.optim as optim

from app.training.registry import TrainingRegistry


class WarmupWrapper:
    def __init__(
        self,
        scheduler: optim.lr_scheduler.LRScheduler,
        warmup_steps: int = 0,
        warmup_lr: float = 1e-6,
    ) -> None:
        self._scheduler = scheduler
        self._warmup_steps = warmup_steps
        self._warmup_lr = warmup_lr
        self._step_count: int = 0
        self._base_lrs: list[float] = [
            group["lr"] for group in scheduler.optimizer.param_groups
        ]

    def step(self) -> None:
        self._step_count += 1
        if self._step_count <= self._warmup_steps:
            alpha = self._step_count / max(self._warmup_steps, 1)
            for i, group in enumerate(self._scheduler.optimizer.param_groups):
                if i < len(self._base_lrs):
                    group["lr"] = self._warmup_lr + alpha * (
                        self._base_lrs[i] - self._warmup_lr
                    )
        else:
            self._scheduler.step()

    @property
    def scheduler(self) -> optim.lr_scheduler.LRScheduler:
        return self._scheduler

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self._warmup_steps,
            "warmup_lr": self._warmup_lr,
            "step_count": self._step_count,
            "base_lrs": self._base_lrs,
            "scheduler_state": self._scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._warmup_steps = state_dict["warmup_steps"]
        self._warmup_lr = state_dict["warmup_lr"]
        self._step_count = state_dict["step_count"]
        self._base_lrs = state_dict["base_lrs"]
        self._scheduler.load_state_dict(state_dict["scheduler_state"])


class SchedulerFactory:
    @staticmethod
    def create(
        optimizer: optim.Optimizer,
        scheduler_type: str = "cosine_annealing",
        warmup_steps: int = 0,
        warmup_lr: float = 1e-6,
        **kwargs: Any,
    ) -> WarmupWrapper | optim.lr_scheduler.LRScheduler:
        scheduler_cls = TrainingRegistry.get_scheduler(scheduler_type)
        default_kwargs: dict[str, Any] = {}

        if scheduler_type == "step_lr":
            default_kwargs["step_size"] = kwargs.get("step_size", 30)
            default_kwargs["gamma"] = kwargs.get("gamma", 0.1)
        elif scheduler_type == "cosine_annealing":
            default_kwargs["T_max"] = kwargs.get("t_max", 100)
            default_kwargs["eta_min"] = kwargs.get("eta_min", 0.0)
        elif scheduler_type == "reduce_on_plateau":
            default_kwargs["mode"] = kwargs.get("mode", "max")
            default_kwargs["factor"] = kwargs.get("factor", 0.1)
            default_kwargs["patience"] = kwargs.get("patience", 10)
            default_kwargs["threshold"] = kwargs.get("threshold", 1e-4)
            default_kwargs["cooldown"] = kwargs.get("cooldown", 0)
            default_kwargs["min_lr"] = kwargs.get("min_lr", 0.0)

        scheduler = scheduler_cls(optimizer, **default_kwargs)

        if warmup_steps > 0:
            return WarmupWrapper(
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                warmup_lr=warmup_lr,
            )
        return scheduler
