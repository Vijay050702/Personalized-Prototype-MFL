from __future__ import annotations

from typing import Any

import numpy as np
import torch

from app.data.validation import validate_sample_tensor


class ImageLoader:
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
            validate_sample_tensor(tensor, "image")

        return tensor

    def ensure_channels(self, tensor: torch.Tensor, channels: int = 3) -> torch.Tensor:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).repeat(channels, 1, 1)
        elif tensor.ndim == 3 and tensor.size(0) == 1 and channels > 1:
            tensor = tensor.repeat(channels, 1, 1)
        return tensor

    def ensure_shape(
        self,
        tensor: torch.Tensor,
        target_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if target_shape is not None and tensor.shape != target_shape:
            from torch.nn.functional import interpolate

            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            tensor = interpolate(
                tensor, size=target_shape[-2:], mode="bilinear", align_corners=False
            )
            tensor = tensor.squeeze(0)
        return tensor
