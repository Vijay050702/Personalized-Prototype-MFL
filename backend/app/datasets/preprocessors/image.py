from __future__ import annotations

from typing import Any

import numpy as np

from app.datasets.base import BasePreprocessor
from app.datasets.transforms import normalize_mean_std, resize


class ImagePreprocessor(BasePreprocessor):
    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
    ):
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.augment = augment

    def fit(self, samples: list[dict[str, np.ndarray]]) -> None:
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        if data.dtype != np.float32 and data.dtype != np.float64:
            data = data.astype(np.float32) / 255.0
        if data.shape[:2] != self.target_size:
            data = resize(data, self.target_size)
        data = normalize_mean_std(data, self.mean, self.std)
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        if data.ndim == 3 and data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
        if data.ndim == 3:
            data = np.transpose(data, (2, 0, 1))
        return data

    def __repr__(self) -> str:
        return f"ImagePreprocessor(target_size={self.target_size})"
