from __future__ import annotations

from typing import Any

import numpy as np
import torch

from app.core.logging import logger
from app.data.modality import MODALITY_KEYS
from app.data.multimodal_dataset import MultimodalDataset


class DatasetStatistics:
    def __init__(self, dataset: MultimodalDataset):
        self._dataset = dataset
        self._compute()

    def _compute(self) -> None:
        self.num_samples = len(self._dataset)
        labels = self._dataset.labels
        self.num_classes = len(np.unique(labels))
        self.class_distribution = self._dataset.class_distribution
        self.modality_availability = self._dataset.modality_availability
        self.missing_modality_ratio = self._compute_missing_ratio()
        self.client_distribution: dict[str, int] = self._compute_client_distribution()
        self.avg_sequence_lengths: dict[str, float] = self._compute_avg_lengths()
        self.avg_image_sizes: dict[str, tuple[float, ...]] = (
            self._compute_avg_image_sizes()
        )
        self._log()

    def _compute_missing_ratio(self) -> float:
        if self.num_samples == 0:
            return 0.0
        present_mods = [
            m for m in MODALITY_KEYS if self.modality_availability.get(m, 0.0) > 0
        ]
        if not present_mods:
            return 0.0
        total_expected = self.num_samples * len(present_mods)
        available = sum(
            self.modality_availability.get(m, 0.0) * self.num_samples
            for m in present_mods
        )
        return 1.0 - (available / total_expected)

    def _compute_client_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for idx in range(self.num_samples):
            raw = self._dataset.get_sample(idx)
            cid = raw.get("data", {}).get("client_id", "server")
            if isinstance(cid, (np.generic,)):
                cid = str(cid)
            dist[str(cid)] = dist.get(str(cid), 0) + 1
        return dist

    def _compute_avg_lengths(self) -> dict[str, float]:
        lengths: dict[str, list[int]] = {m: [] for m in MODALITY_KEYS}
        for idx in range(min(self.num_samples, 1000)):
            sample = self._dataset[idx]
            for mod in MODALITY_KEYS:
                tensor = sample.get_tensor(mod)
                if tensor is not None and tensor.numel() > 0:
                    lengths[mod].append(tensor.size(0))
        result: dict[str, float] = {}
        for mod, vals in lengths.items():
            if vals:
                result[mod] = float(np.mean(vals))
            else:
                result[mod] = 0.0
        return result

    def _compute_avg_image_sizes(self) -> dict[str, tuple[float, ...]]:
        sizes: dict[str, list[tuple[int, ...]]] = {m: [] for m in MODALITY_KEYS}
        for idx in range(min(self.num_samples, 500)):
            sample = self._dataset[idx]
            for mod in ("image",):
                tensor = sample.get_tensor(mod)
                if tensor is not None and tensor.numel() > 0:
                    sizes[mod].append(tuple(tensor.shape))
        result: dict[str, tuple[float, ...]] = {}
        for mod, vals in sizes.items():
            if vals:
                avg = tuple(
                    float(np.mean([s[d] for s in vals])) for d in range(len(vals[0]))
                )
                result[mod] = avg
            else:
                result[mod] = ()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "class_distribution": self.class_distribution,
            "modality_availability": self.modality_availability,
            "missing_modality_ratio": self.missing_modality_ratio,
            "client_distribution": self.client_distribution,
            "avg_sequence_lengths": self.avg_sequence_lengths,
            "avg_image_sizes": self.avg_image_sizes,
        }

    def _log(self) -> None:
        logger.info(
            f"DatasetStatistics: {self.num_samples} samples, "
            f"{self.num_classes} classes, "
            f"missing_ratio={self.missing_modality_ratio:.3f}, "
            f"modalities={self.modality_availability}"
        )

    def __repr__(self) -> str:
        return (
            f"DatasetStatistics(samples={self.num_samples}, "
            f"classes={self.num_classes}, "
            f"missing_ratio={self.missing_modality_ratio:.3f})"
        )
