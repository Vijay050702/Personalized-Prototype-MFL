from __future__ import annotations

from typing import Any

import numpy as np

from app.datasets.base import BasePreprocessor
from app.datasets.transforms import normalize_min_max, sliding_window


class SensorPreprocessor(BasePreprocessor):
    def __init__(
        self,
        window_size: int = 128,
        stride: int = 64,
        normalize: bool = True,
        interpolation_method: str = "linear",
    ):
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.interpolation_method = interpolation_method

    def fit(self, samples: list[dict[str, np.ndarray]]) -> None:
        pass

    def _interpolate(self, data: np.ndarray, target_length: int) -> np.ndarray:
        n = len(data)
        if n == target_length:
            return data
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_length)
        return np.column_stack(
            [np.interp(x_new, x_old, data[:, i]) for i in range(data.shape[1])]
        )

    def process(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr = sliding_window(arr, self.window_size, self.stride)
        if self.normalize:
            arr = np.array([normalize_min_max(w) for w in arr])
        return arr

    def __repr__(self) -> str:
        return f"SensorPreprocessor(window={self.window_size}, stride={self.stride})"
