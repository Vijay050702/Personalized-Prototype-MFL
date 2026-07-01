from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split

from app.core.config import settings
from app.core.logging import logger
from app.data.collate import collate_multimodal_samples
from app.data.dataloader import MultimodalDataLoader
from app.data.modality import MODALITY_KEYS
from app.data.multimodal_dataset import MultimodalDataset
from app.data.multimodal_sample import MultimodalSample
from app.data.statistics import DatasetStatistics
from app.datasets import dataset_factory, dataset_registry
from app.datasets.base import DatasetLoadResult


class DataModule:
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        validate_samples: bool = True,
        lazy_load: bool = False,
        cache_enabled: bool = True,
        client_id: str | None = None,
        client_indices: list[int] | None = None,
        collate_fn: Callable | None = None,
    ):
        self._dataset_name = dataset_name
        self._batch_size = batch_size
        self._val_split = val_split
        self._test_split = test_split
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed
        self._transforms = transforms or {}
        self._validate_samples = validate_samples
        self._lazy_load = lazy_load
        self._cache_enabled = cache_enabled
        self._client_id = client_id
        self._client_indices = client_indices
        self._collate_fn = collate_fn or collate_multimodal_samples

        self._dataset: Dataset[MultimodalSample] | None = None
        self._train_dataset: Dataset[MultimodalSample] | None = None
        self._val_dataset: Dataset[MultimodalSample] | None = None
        self._test_dataset: Dataset[MultimodalSample] | None = None
        self._statistics: DatasetStatistics | None = None
        self._num_classes: int | None = None

    def prepare_data(self) -> None:
        start = time.time()
        ds_root = Path(settings.datasets_root)
        try:
            adapter = dataset_factory.create(self._dataset_name)
            load_result = adapter.load(ds_root, split="train")
        except Exception:
            logger.warning(
                f"Cannot load data for {self._dataset_name} from disk, "
                "using registered metadata"
            )
            load_result = self._build_from_registry()
        self._num_classes = self._resolve_num_classes(load_result)
        full_dataset = MultimodalDataset(
            load_result=load_result,
            dataset_name=self._dataset_name,
            split="train",
            client_id=self._client_id,
            client_indices=self._client_indices,
            transforms=self._transforms,
            validate_samples=self._validate_samples,
            num_classes=self._num_classes,
            lazy_load=self._lazy_load,
        )
        self._statistics = DatasetStatistics(full_dataset)
        self._dataset = full_dataset
        elapsed = time.time() - start
        logger.info(
            f"DataModule.prepare_data() for {self._dataset_name} "
            f"completed in {elapsed:.3f}s"
        )

    def _build_from_registry(self) -> DatasetLoadResult:
        meta = dataset_registry.get(self._dataset_name)
        return DatasetLoadResult(
            data=[],
            labels=np.array([], dtype=np.int64),
            metadata=meta,
        )

    def _resolve_num_classes(self, load_result: DatasetLoadResult) -> int | None:
        meta = load_result.metadata or {}
        nc = meta.get("num_classes", None)
        if nc is not None:
            return int(nc)
        try:
            reg = dataset_registry.get(self._dataset_name)
            return int(reg.get("num_classes", 0)) or None
        except Exception:
            return None

    def setup(self, stage: str | None = None) -> None:
        if self._dataset is None:
            self.prepare_data()

        full = self._dataset
        total = len(full)
        val_size = int(total * self._val_split)
        test_size = int(total * self._test_split)
        train_size = total - val_size - test_size

        if train_size <= 0:
            logger.warning(f"Train set too small ({train_size}), adjusting splits")
            train_size = max(1, total // 2)
            remaining = total - train_size
            val_size = remaining // 2
            test_size = remaining - val_size

        generator = torch.Generator().manual_seed(self._seed)
        subsets = random_split(
            full,
            [train_size, val_size, test_size],
            generator=generator,
        )

        transforms = self._transforms
        self._num_classes = self._num_classes or full.num_classes

        self._train_dataset = self._wrap_subset(subsets[0], "train", transforms)
        self._val_dataset = self._wrap_subset(subsets[1], "val", transforms)
        self._test_dataset = self._wrap_subset(subsets[2], "test", transforms)

        logger.info(
            f"DataModule.setup({stage}): "
            f"train={len(self._train_dataset)}, "
            f"val={len(self._val_dataset)}, "
            f"test={len(self._test_dataset)}"
        )

    def _wrap_subset(
        self,
        subset: Subset,
        split: str,
        transforms: dict[str, Callable] | None,
    ) -> MultimodalDataset:
        indices = subset.indices
        data_list = []
        labels_list = []
        for i in indices:
            raw = (
                self._dataset.get_sample(i)
                if hasattr(self._dataset, "get_sample")
                else None
            )
            data_list.append(
                self._dataset.dataset[i]
                if hasattr(self._dataset, "dataset")
                else self._dataset[i]
            )

        ds = MultimodalDataset(
            data=[d._tensors for d in data_list],
            labels=np.array([d.label for d in data_list], dtype=np.int64),
            metadata={
                "name": self._dataset_name,
                "split": split,
                "modalities": list(transforms.keys()) if transforms else MODALITY_KEYS,
            },
            dataset_name=self._dataset_name,
            split=split,
            client_id=self._client_id,
            transforms=transforms,
            validate_samples=self._validate_samples,
            num_classes=self._num_classes,
            lazy_load=self._lazy_load,
        )
        return ds

    def train_dataloader(self) -> MultimodalDataLoader:
        if self._train_dataset is None:
            self.setup("train")
        return MultimodalDataLoader(
            dataset=self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
            seed=self._seed,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> MultimodalDataLoader:
        if self._val_dataset is None:
            self.setup("val")
        return MultimodalDataLoader(
            dataset=self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
            seed=self._seed,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> MultimodalDataLoader:
        if self._test_dataset is None:
            self.setup("test")
        return MultimodalDataLoader(
            dataset=self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
            seed=self._seed,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self) -> MultimodalDataLoader:
        return self.test_dataloader()

    @property
    def statistics(self) -> DatasetStatistics | None:
        return self._statistics

    @property
    def num_classes(self) -> int:
        if self._num_classes is not None:
            return self._num_classes
        if self._statistics is not None:
            return self._statistics.num_classes
        return 0

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def log_summary(self) -> None:
        logger.info(
            f"DataModule: {self._dataset_name}, "
            f"batch_size={self._batch_size}, "
            f"workers={self._num_workers}, "
            f"val_split={self._val_split}, "
            f"test_split={self._test_split}, "
            f"num_classes={self.num_classes}"
        )
