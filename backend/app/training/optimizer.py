from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from app.training.registry import TrainingRegistry


class FedProxOptimizer(optim.Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        mu: float = 0.01,
        **kwargs: Any,
    ) -> None:
        defaults: dict[str, Any] = {"lr": lr, "mu": mu, **kwargs}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            mu = group["mu"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "global_params" in group:
                    global_p = group["global_params"].get(id(p))
                    if global_p is not None:
                        p.grad.data += mu * (p.data - global_p.data)
                p.data.add_(p.grad.data, alpha=-group["lr"])
        return loss


class OptimizerFactory:
    @staticmethod
    def create(
        model: nn.Module,
        optimizer_type: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        **kwargs: Any,
    ) -> optim.Optimizer:
        if optimizer_type == "fedprox":
            return FedProxOptimizer(
                model.parameters(),
                lr=lr,
                mu=kwargs.get("mu", 0.01),
                weight_decay=weight_decay,
            )

        optimizer_cls = TrainingRegistry.get_optimizer(optimizer_type)
        base_kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}

        if optimizer_type == "sgd":
            base_kwargs["momentum"] = momentum

        return optimizer_cls(model.parameters(), **base_kwargs, **kwargs)

    @staticmethod
    def create_fedprox(
        model: nn.Module,
        lr: float = 1e-3,
        mu: float = 0.01,
        weight_decay: float = 0.0,
    ) -> FedProxOptimizer:
        return FedProxOptimizer(
            model.parameters(),
            lr=lr,
            mu=mu,
            weight_decay=weight_decay,
        )

    @staticmethod
    def set_global_params(
        optimizer: optim.Optimizer,
        model: nn.Module,
    ) -> None:
        if not isinstance(optimizer, FedProxOptimizer):
            return
        global_params: dict[int, nn.Parameter] = {}
        for p in model.parameters():
            global_params[id(p)] = p.data.clone().detach()
        for group in optimizer.param_groups:
            group["global_params"] = global_params
