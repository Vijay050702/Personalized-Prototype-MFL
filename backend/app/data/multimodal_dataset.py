from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from app.core.config import settings
from app.core.logging import logger
from app.data.collate import collate_multimodal_samples
from app.data.modality import MODALITY_KEYS
from app.data.multimodal_sample import MultimodalSample
from app.data.validation import validate_sample
from app.datasets.base import DatasetLoadResult
from app.datasets.cache import DatasetCache


class MultimodalDataset(Dataset[MultimodalSample]):
    def __init__(
        self,
        data: list[dict[str, np.ndarray]] | None = None,
        labels: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        load_result: DatasetLoadResult | None = None,
        dataset_name: str | None = None,
        split: str = "train",
        client_id: str | None = None,
        client_indices: list[int] | None = None,
        transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        validate_samples: bool = True,
        num_classes: int | None = None,
        lazy_load: bool = False,
        cache: DatasetCache | None = None,
        cache_key: str | None = None,
    ):
        if load_result is not None:
            self._data = load_result.data
            self._labels = load_result.labels
            self._meta = load_result.metadata or {}
        elif data is not None and labels is not None:
            self._data = data
            self._labels = labels
            self._meta = metadata or {}
        else:
            raise ValueError("Either load_result or (data + labels) must be provided")

        self._dataset_name = dataset_name or self._meta.get("name", "unknown")
        self._split = split
        self._client_id = client_id
        self._transforms = transforms or {}
        self._validate_samples = validate_samples
        self._num_classes = num_classes
        self._lazy_load = lazy_load
        self._cache = cache
        self._cache_key = cache_key

        self._sample_cache: dict[int, MultimodalSample] = {}

        if client_indices is not None:
            self._data = [self._data[i] for i in client_indices]
            self._labels = self._labels[client_indices]
            logger.info(
                f"Filtered dataset to {len(self._data)} samples "
                f"for {self._dataset_name}/{split} client={client_id}"
            )

        if not lazy_load:
            self._preload_all()

        self._log_loading()

    def _preload_all(self) -> None:
        start = time.time()
        for idx in range(len(self._data)):
            self._build_sample(idx)
        elapsed = time.time() - start
        logger.debug(
            f"Preloaded {len(self._data)} samples for "
            f"{self._dataset_name}/{self._split} in {elapsed:.3f}s"
        )

    def _build_sample(self, idx: int) -> MultimodalSample:
        raw = self._data[idx]
        lbl = int(self._labels[idx])

        if self._cache_key and self._cache is not None:
            cached = self._cache.get(f"{self._cache_key}_sample_{idx}")
            if cached is not None:
                return cached

        sample = MultimodalSample.from_dict(
            raw,
            sample_id=idx,
            label=lbl,
            client_id=self._client_id or "server",
        )

        for mod in sample.available_modalities:
            tensor = sample.get_tensor(mod)
            if tensor is not None:
                transform = self._transforms.get(mod)
                if transform is not None:
                    tensor = transform(tensor)
                sample.set_tensor(mod, tensor)

        if self._validate_samples:
            validate_sample(sample, num_classes=self._num_classes)

        if self._cache_key and self._cache is not None:
            self._cache.set(f"{self._cache_key}_sample_{idx}", sample)

        return sample

    def __getitem__(self, idx: int) -> MultimodalSample:
        if self._lazy_load and idx in self._sample_cache:
            return self._sample_cache[idx]

        sample = self._build_sample(idx)

        if self._lazy_load:
            self._sample_cache[idx] = sample

        return sample

    def __len__(self) -> int:
        return len(self._data)

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def num_classes(self) -> int:
        if self._num_classes is not None:
            return self._num_classes
        return len(np.unique(self._labels))

    @property
    def class_distribution(self) -> dict[int, int]:
        unique, counts = np.unique(self._labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    @property
    def modality_availability(self) -> dict[str, float]:
        counts: dict[str, int] = {m: 0 for m in MODALITY_KEYS}
        total = len(self._data)
        if total == 0:
            return {m: 0.0 for m in MODALITY_KEYS}
        for raw in self._data:
            for mod in MODALITY_KEYS:
                missing_key = f"{mod}_missing"
                if missing_key in raw and raw[missing_key]:
                    continue
                if mod in raw:
                    val = raw[mod]
                    if hasattr(val, "__len__") and len(val) > 0:
                        counts[mod] += 1
                    elif not hasattr(val, "__len__"):
                        counts[mod] += 1
        return {m: counts[m] / total for m in MODALITY_KEYS}

    def to_load_result(self) -> DatasetLoadResult:
        return DatasetLoadResult(
            data=self._data,
            labels=self._labels,
            metadata={
                "name": self._dataset_name,
                "split": self._split,
                "modalities": [
                    m for m, r in self.modality_availability.items() if r > 0
                ],
                "num_samples": len(self),
                "num_classes": self.num_classes,
                "client_id": self._client_id,
            },
        )

    def get_sample(self, idx: int) -> dict[str, Any]:
        return {
            "data": self._data[idx],
            "label": int(self._labels[idx]),
        }

    def _log_loading(self) -> None:
        logger.info(
            f"MultimodalDataset: {self._dataset_name}/{self._split} "
            f"loaded {len(self)} samples, "
            f"{self.num_classes} classes, "
            f"modalities: {self.modality_availability}"
        )
