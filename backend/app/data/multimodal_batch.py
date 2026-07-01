from __future__ import annotations

from typing import Any

import torch

from app.data.modality import MODALITY_KEYS


class MultimodalBatch:
    def __init__(
        self,
        images: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        sensor: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        client_ids: list[str] | None = None,
        modality_masks: torch.Tensor | None = None,
        sample_ids: list[int | str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.images = images
        self.text = text
        self.audio = audio
        self.sensor = sensor
        self.labels = labels
        self.client_ids = client_ids or []
        self.modality_masks = modality_masks
        self.sample_ids = sample_ids or []
        self.metadata = metadata or {}

    @property
    def batch_size(self) -> int:
        if self.labels is not None:
            return len(self.labels)
        if self.images is not None:
            return self.images.size(0)
        return 0

    def get_modality_tensor(self, modality: str) -> torch.Tensor | None:
        mapping = {
            "image": self.images,
            "text": self.text,
            "audio": self.audio,
            "sensor": self.sensor,
        }
        return mapping.get(modality)

    def set_modality_tensor(self, modality: str, tensor: torch.Tensor) -> None:
        setattr(self, modality if modality != "image" else "images", tensor)

    def has_modality(self, modality: str) -> bool:
        tensor = self.get_modality_tensor(modality)
        return tensor is not None

    @property
    def available_modalities(self) -> list[str]:
        return [m for m in MODALITY_KEYS if self.has_modality(m)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "text": self.text,
            "audio": self.audio,
            "sensor": self.sensor,
            "labels": self.labels,
            "client_ids": self.client_ids,
            "modality_masks": self.modality_masks,
            "sample_ids": self.sample_ids,
            "metadata": self.metadata,
            "batch_size": self.batch_size,
        }

    def pin_memory(self) -> MultimodalBatch:
        pinned = MultimodalBatch(
            images=self.images.pin_memory() if self.images is not None else None,
            text=self.text.pin_memory() if self.text is not None else None,
            audio=self.audio.pin_memory() if self.audio is not None else None,
            sensor=self.sensor.pin_memory() if self.sensor is not None else None,
            labels=self.labels.pin_memory() if self.labels is not None else None,
            client_ids=self.client_ids,
            modality_masks=(
                self.modality_masks.pin_memory()
                if self.modality_masks is not None
                else None
            ),
            sample_ids=self.sample_ids,
            metadata=self.metadata,
        )
        return pinned

    def to(self, device: torch.device | str) -> MultimodalBatch:
        mapped = MultimodalBatch(
            images=self.images.to(device) if self.images is not None else None,
            text=self.text.to(device) if self.text is not None else None,
            audio=self.audio.to(device) if self.audio is not None else None,
            sensor=self.sensor.to(device) if self.sensor is not None else None,
            labels=self.labels.to(device) if self.labels is not None else None,
            client_ids=self.client_ids,
            modality_masks=(
                self.modality_masks.to(device)
                if self.modality_masks is not None
                else None
            ),
            sample_ids=self.sample_ids,
            metadata=self.metadata,
        )
        return mapped

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self) -> str:
        return (
            f"MultimodalBatch(size={self.batch_size}, "
            f"modalities={self.available_modalities})"
        )
