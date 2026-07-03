from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn


def compute_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: int = 1
) -> torch.Tensor:
    preds = outputs.topk(top_k, dim=1).indices
    correct = preds.eq(targets.view(-1, 1).expand_as(preds))
    return correct.any(dim=1).float().mean()


def to_device(
    data: Any,
    device: torch.device,
) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    return data


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    return total_norm**0.5


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm, norm_type=norm_type
    )


def flatten_model_state(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for param in state_dict.values():
        vectors.append(param.data.view(-1))
    return torch.cat(vectors)


def unflatten_model_state(
    flat: torch.Tensor,
    reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    idx = 0
    for name, param in reference.items():
        numel = param.numel()
        state_dict[name] = flat[idx : idx + numel].view(param.shape).clone()
        idx += numel
    return state_dict


class Timer:
    def __init__(self) -> None:
        self._start: float = 0.0
        self._elapsed: float = 0.0
        self._running: bool = False

    def start(self) -> None:
        self._start = time.time()
        self._running = True

    def stop(self) -> float:
        if self._running:
            self._elapsed += time.time() - self._start
            self._running = False
        return self._elapsed

    def reset(self) -> None:
        self._start = 0.0
        self._elapsed = 0.0
        self._running = False

    @property
    def elapsed(self) -> float:
        if self._running:
            return self._elapsed + (time.time() - self._start)
        return self._elapsed


def merge_configs(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "rounds" not in config:
        errors.append("Missing required config: rounds")
    if "clients" not in config:
        errors.append("Missing required config: clients")
    if "model" not in config:
        errors.append("Missing required config: model")
    if "dataset" not in config:
        errors.append("Missing required config: dataset")
    return errors
