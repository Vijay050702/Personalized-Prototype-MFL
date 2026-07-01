from __future__ import annotations

from enum import Enum
from typing import Sequence

import torch

MODALITY_KEYS = ("image", "text", "audio", "sensor")
NUM_MODALITIES = len(MODALITY_KEYS)


class Modality(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    SENSOR = "sensor"

    @classmethod
    def all(cls) -> list[Modality]:
        return [cls.IMAGE, cls.TEXT, cls.AUDIO, cls.SENSOR]

    @classmethod
    def from_str(cls, value: str) -> Modality:
        return cls(value.lower())


def modality_mask_from_dict(
    sample: dict[str, torch.Tensor | object],
) -> dict[str, bool]:
    mask: dict[str, bool] = {}
    for mod in MODALITY_KEYS:
        missing_key = f"{mod}_missing"
        if missing_key in sample and sample[missing_key]:
            mask[mod] = False
        elif mod in sample:
            val = sample[mod]
            if isinstance(val, torch.Tensor):
                mask[mod] = val.numel() > 0
            elif hasattr(val, "__len__"):
                mask[mod] = len(val) > 0
            else:
                mask[mod] = val is not None
        else:
            mask[mod] = False
    return mask


def modality_mask_to_tensor(mask: dict[str, bool]) -> torch.Tensor:
    return torch.tensor(
        [mask.get(mod, False) for mod in MODALITY_KEYS], dtype=torch.bool
    )


def available_modalities(mask: dict[str, bool]) -> list[str]:
    return [mod for mod, avail in mask.items() if avail]


def missing_modalities(mask: dict[str, bool]) -> list[str]:
    return [mod for mod, avail in mask.items() if not avail]


def all_modalities_available(mask: dict[str, bool]) -> bool:
    return all(mask.get(mod, False) for mod in MODALITY_KEYS)
