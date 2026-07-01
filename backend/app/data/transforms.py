from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        pass


class IdentityTransform(BaseTransform):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class TypeCastTransform(BaseTransform):
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self.dtype)


class ImageTransform(BaseTransform):
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
        mean: tuple[float, ...] = (0.5,),
        std: tuple[float, ...] = (0.5,),
    ):
        self.dtype = dtype
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        t = tensor.to(dtype=self.dtype)
        if self.normalize and t.numel() > 0:
            ndim = t.ndim
            if ndim == 2:
                t = t.unsqueeze(0)
            for c in range(t.shape[0]):
                mean_c = self.mean[c] if c < len(self.mean) else 0.5
                std_c = self.std[c] if c < len(self.std) else 0.5
                t[c] = (t[c] - mean_c) / std_c
        return t


class TextTransform(BaseTransform):
    def __init__(self, dtype: torch.dtype = torch.long):
        self.dtype = dtype

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self.dtype)


class AudioTransform(BaseTransform):
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
    ):
        self.dtype = dtype
        self.normalize = normalize

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        t = tensor.to(dtype=self.dtype)
        if self.normalize and t.numel() > 0 and t.std() > 0:
            t = (t - t.mean()) / t.std()
        return t


class SensorTransform(BaseTransform):
    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
    ):
        self.dtype = dtype
        self.normalize = normalize

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        t = tensor.to(dtype=self.dtype)
        if self.normalize and t.numel() > 0:
            if t.ndim == 1:
                if t.std() > 0:
                    t = (t - t.mean()) / t.std()
            else:
                for c in range(t.shape[-1]):
                    col = t[..., c]
                    if col.std() > 0:
                        t[..., c] = (col - col.mean()) / col.std()
        return t


class ComposeTransform(BaseTransform):
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

    def __repr__(self) -> str:
        return f"ComposeTransform({self.transforms})"
