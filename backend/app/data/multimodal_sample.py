from __future__ import annotations

from typing import Any

import torch

from app.data.modality import (
    MODALITY_KEYS,
    modality_mask_from_dict,
    modality_mask_to_tensor,
)


class MultimodalSample:
    def __init__(
        self,
        sample_id: int | str,
        label: int,
        client_id: str = "server",
        image: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        sensor: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.sample_id = sample_id
        self.label = label
        self.client_id = client_id

        self._tensors: dict[str, torch.Tensor] = {}
        if image is not None:
            self._tensors["image"] = image
        if text is not None:
            self._tensors["text"] = text
        if audio is not None:
            self._tensors["audio"] = audio
        if sensor is not None:
            self._tensors["sensor"] = sensor

        self.metadata = metadata or {}

    @property
    def image(self) -> torch.Tensor | None:
        return self._tensors.get("image")

    @image.setter
    def image(self, value: torch.Tensor | None) -> None:
        if value is None:
            self._tensors.pop("image", None)
        else:
            self._tensors["image"] = value

    @property
    def text(self) -> torch.Tensor | None:
        return self._tensors.get("text")

    @text.setter
    def text(self, value: torch.Tensor | None) -> None:
        if value is None:
            self._tensors.pop("text", None)
        else:
            self._tensors["text"] = value

    @property
    def audio(self) -> torch.Tensor | None:
        return self._tensors.get("audio")

    @audio.setter
    def audio(self, value: torch.Tensor | None) -> None:
        if value is None:
            self._tensors.pop("audio", None)
        else:
            self._tensors["audio"] = value

    @property
    def sensor(self) -> torch.Tensor | None:
        return self._tensors.get("sensor")

    @sensor.setter
    def sensor(self, value: torch.Tensor | None) -> None:
        if value is None:
            self._tensors.pop("sensor", None)
        else:
            self._tensors["sensor"] = value

    @property
    def available_modalities(self) -> list[str]:
        return list(self._tensors.keys())

    @property
    def missing_modalities(self) -> list[str]:
        return [m for m in MODALITY_KEYS if m not in self._tensors]

    @property
    def modality_mask(self) -> dict[str, bool]:
        return {m: m in self._tensors for m in MODALITY_KEYS}

    @property
    def modality_mask_tensor(self) -> torch.Tensor:
        return modality_mask_to_tensor(self.modality_mask)

    def get_tensor(self, modality: str) -> torch.Tensor | None:
        return self._tensors.get(modality)

    def set_tensor(self, modality: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            self._tensors.pop(modality, None)
        else:
            self._tensors[modality] = tensor

    def has_modality(self, modality: str) -> bool:
        return modality in self._tensors

    def num_available(self) -> int:
        return len(self._tensors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "client_id": self.client_id,
            "modality_mask": self.modality_mask,
            "metadata": self.metadata,
            **{mod: self._tensors.get(mod) for mod in MODALITY_KEYS},
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        sample_id: int | str | None = None,
        label: int | None = None,
        client_id: str | None = None,
    ) -> MultimodalSample:
        tensors: dict[str, torch.Tensor] = {}
        for mod in MODALITY_KEYS:
            missing_key = f"{mod}_missing"
            if missing_key in data and data[missing_key]:
                continue
            val = data.get(mod)
            if val is not None:
                if isinstance(val, torch.Tensor):
                    tensors[mod] = val
                elif isinstance(val, __import__("numpy").ndarray):
                    tensors[mod] = torch.from_numpy(val)
                else:
                    try:
                        tensors[mod] = torch.tensor(val)
                    except (TypeError, ValueError):
                        continue

        sid = sample_id if sample_id is not None else data.get("sample_id", -1)
        lbl = label if label is not None else data.get("label", 0)
        cid = client_id if client_id is not None else data.get("client_id", "server")
        meta = {
            k: v
            for k, v in data.items()
            if k
            not in (
                *MODALITY_KEYS,
                *(f"{m}_missing" for m in MODALITY_KEYS),
                "label",
                "sample_id",
                "client_id",
            )
        }

        return cls(
            sample_id=sid,
            label=int(lbl) if not isinstance(lbl, int) else lbl,
            client_id=str(cid),
            image=tensors.get("image"),
            text=tensors.get("text"),
            audio=tensors.get("audio"),
            sensor=tensors.get("sensor"),
            metadata=meta,
        )

    def __repr__(self) -> str:
        return (
            f"MultimodalSample(id={self.sample_id}, label={self.label}, "
            f"client={self.client_id}, modalities={self.available_modalities})"
        )
