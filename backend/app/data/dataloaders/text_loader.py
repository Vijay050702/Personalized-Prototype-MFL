from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TextLoader:
    def __init__(self, dtype: torch.dtype = torch.long):
        self.dtype = dtype

    def load(
        self, data: np.ndarray | torch.Tensor | list[int], **kwargs: Any
    ) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype=self.dtype)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(dtype=self.dtype)
        elif isinstance(data, (list, tuple)):
            tensor = torch.tensor(data, dtype=self.dtype)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

    def truncate(self, tensor: torch.Tensor, max_length: int) -> torch.Tensor:
        if tensor.size(0) > max_length:
            return tensor[:max_length]
        return tensor

    def pad_or_truncate(
        self, tensor: torch.Tensor, length: int, padding_value: int = 0
    ) -> torch.Tensor:
        current = tensor.size(0)
        if current > length:
            return tensor[:length]
        if current < length:
            pad = torch.full((length - current,), padding_value, dtype=self.dtype)
            return torch.cat([tensor, pad], dim=0)
        return tensor
