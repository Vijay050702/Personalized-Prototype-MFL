from __future__ import annotations

from typing import Any

import numpy as np
import torch

from app.data.validation import validate_sample_tensor


class SensorLoader:
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def load(self, data: np.ndarray | torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype=self.dtype)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(dtype=self.dtype)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if tensor.numel() > 0:
            validate_sample_tensor(tensor, "sensor")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def ensure_windowed(
        self, tensor: torch.Tensor, window_size: int, stride: int | None = None
    ) -> torch.Tensor:
        if stride is None:
            stride = window_size
        if tensor.size(0) < window_size:
            return tensor.unsqueeze(0)
        windows: list[torch.Tensor] = []
        for start in range(0, tensor.size(0) - window_size + 1, stride):
            windows.append(tensor[start : start + window_size])
        return torch.stack(windows, dim=0) if windows else tensor.unsqueeze(0)

    def normalize_channels(
        self, tensor: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        if tensor.ndim == 1:
            mean = tensor.mean()
            std = tensor.std()
            return (tensor - mean) / (std + eps)
        for c in range(tensor.shape[-1]):
            col = tensor[..., c]
            mean = col.mean()
            std = col.std()
            tensor[..., c] = (col - mean) / (std + eps)
        return tensor
