from __future__ import annotations

import time
from typing import Any

import torch

from app.core.logging import logger


def validate_embedding(embedding: torch.Tensor) -> None:
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(embedding)}")
    if embedding.dim() != 1:
        raise ValueError(f"Expected 1-D embedding, got {embedding.dim()}-D tensor")
    if embedding.numel() == 0:
        raise ValueError("Embedding is empty")


def check_nan(tensor: torch.Tensor, name: str = "tensor") -> None:
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")


def validate_class_id(class_id: int, num_classes: int | None = None) -> None:
    if not isinstance(class_id, (int,)):
        raise TypeError(f"class_id must be int, got {type(class_id)}")
    if class_id < 0:
        raise ValueError(f"class_id must be non-negative, got {class_id}")
    if num_classes is not None and class_id >= num_classes:
        raise ValueError(f"class_id {class_id} out of range [0, {num_classes - 1}]")


def validate_prototype_list(prototypes: list[Any]) -> None:
    from app.prototypes.prototype import Prototype

    for p in prototypes:
        if not isinstance(p, Prototype):
            raise TypeError(f"Expected Prototype, got {type(p)}")


def validate_similarity_metric(metric: str) -> None:
    valid = {"cosine", "euclidean", "manhattan", "dot"}
    if metric not in valid:
        raise ValueError(f"Invalid similarity metric '{metric}'. Choose from {valid}")


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / (a.norm(p=2, dim=1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return a_norm @ b_norm.T


def euclidean_distance_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2.0)


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
