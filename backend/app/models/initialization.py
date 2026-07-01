from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from app.core.logging import logger


def xavier_uniform(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def xavier_normal(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def kaiming_uniform(module: nn.Module, a: float = 0.0, mode: str = "fan_in") -> None:
    if (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv1d)
        or isinstance(module, nn.Conv2d)
    ):
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def kaiming_normal(module: nn.Module, a: float = 0.0, mode: str = "fan_in") -> None:
    if (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv1d)
        or isinstance(module, nn.Conv2d)
    ):
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def normal_init(module: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    if (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv1d)
        or isinstance(module, nn.Conv2d)
    ):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv1d)
        or isinstance(module, nn.Conv2d)
    ):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


_INITIALIZERS: dict[str, Callable[[nn.Module], None]] = {
    "xavier_uniform": lambda m: xavier_uniform(m),
    "xavier_normal": lambda m: xavier_normal(m),
    "kaiming_uniform": lambda m: kaiming_uniform(m),
    "kaiming_normal": lambda m: kaiming_normal(m),
    "normal": lambda m: normal_init(m),
    "orthogonal": lambda m: orthogonal_init(m),
}


def register_initializer(name: str, fn: Callable[[nn.Module], None]) -> None:
    _INITIALIZERS[name] = fn


def get_initializer(name: str) -> Callable[[nn.Module], None]:
    if name not in _INITIALIZERS:
        raise ValueError(
            f"Unknown initializer '{name}'. Available: {list(_INITIALIZERS.keys())}"
        )
    return _INITIALIZERS[name]


def initialize_weights(
    model: nn.Module,
    init_type: str = "kaiming_uniform",
    skip_modules: list[str] | None = None,
    **kwargs: Any,
) -> None:
    init_fn = get_initializer(init_type)
    skip = set(skip_modules or [])
    for name, module in model.named_modules():
        if any(name.startswith(s) for s in skip):
            continue
        init_fn(module)
    logger.info(f"Initialized {type(model).__name__} with {init_type}")
