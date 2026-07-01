from __future__ import annotations

from typing import Any

import numpy as np
import torch

from app.data.validation import validate_sample_tensor


class AudioLoader:
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
            validate_sample_tensor(tensor, "audio")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def resample(
        self, tensor: torch.Tensor, orig_sr: int, target_sr: int
    ) -> torch.Tensor:
        if orig_sr == target_sr:
            return tensor
        from torch.nn.functional import interpolate

        orig_len = tensor.size(-1)
        target_len = int(orig_len * target_sr / orig_sr)
        tensor = tensor.unsqueeze(0)
        tensor = interpolate(
            tensor, size=target_len, mode="linear", align_corners=False
        )
        tensor = tensor.squeeze(0)
        return tensor

    def ensure_length(
        self, tensor: torch.Tensor, target_length: int, pad_value: float = 0.0
    ) -> torch.Tensor:
        current = tensor.size(-1)
        if current > target_length:
            return tensor[..., :target_length]
        if current < target_length:
            pad = torch.full(
                (tensor.size(0), target_length - current),
                pad_value,
                dtype=self.dtype,
            )
            return torch.cat([tensor, pad], dim=-1)
        return tensor
