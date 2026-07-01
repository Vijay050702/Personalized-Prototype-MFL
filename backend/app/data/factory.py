from __future__ import annotations

from typing import Any, Callable

import torch

from app.data.collate import collate_multimodal_samples
from app.data.dataloader import MultimodalDataLoader
from app.data.datamodule import DataModule
from app.data.multimodal_dataset import MultimodalDataset
from app.datasets.base import DatasetLoadResult


class DataFactory:
    def __init__(self) -> None:
        self._dataset_cache: dict[str, MultimodalDataset] = {}
        self._datamodule_cache: dict[str, DataModule] = {}

    def create_dataset(
        self,
        load_result: DatasetLoadResult,
        dataset_name: str = "dataset",
        split: str = "train",
        client_id: str | None = None,
        client_indices: list[int] | None = None,
        transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        validate_samples: bool = True,
        num_classes: int | None = None,
        lazy_load: bool = False,
        cache_key: str | None = None,
        use_cache: bool = True,
    ) -> MultimodalDataset:
        key = cache_key or f"{dataset_name}_{split}_{client_id or 'all'}"
        if use_cache and key in self._dataset_cache:
            return self._dataset_cache[key]

        from app.data.multimodal_dataset import MultimodalDataset

        dataset = MultimodalDataset(
            load_result=load_result,
            dataset_name=dataset_name,
            split=split,
            client_id=client_id,
            client_indices=client_indices,
            transforms=transforms,
            validate_samples=validate_samples,
            num_classes=num_classes,
            lazy_load=lazy_load,
        )
        if use_cache:
            self._dataset_cache[key] = dataset
        return dataset

    def create_dataloader(
        self,
        dataset: MultimodalDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
        collate_fn: Callable | None = None,
    ) -> MultimodalDataLoader:
        return MultimodalDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            seed=seed,
            collate_fn=collate_fn or collate_multimodal_samples,
        )

    def create_datamodule(
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
        use_cache: bool = True,
    ) -> DataModule:
        key = f"{dataset_name}_{client_id or 'all'}"
        if use_cache and key in self._datamodule_cache:
            return self._datamodule_cache[key]

        dm = DataModule(
            dataset_name=dataset_name,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            transforms=transforms,
            validate_samples=validate_samples,
            lazy_load=lazy_load,
            cache_enabled=cache_enabled,
            client_id=client_id,
            client_indices=client_indices,
        )
        if use_cache:
            self._datamodule_cache[key] = dm
        return dm

    def clear_cache(self) -> None:
        self._dataset_cache.clear()
        self._datamodule_cache.clear()

    def log_available(self) -> None:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"DataFactory: {len(self._dataset_cache)} datasets, "
            f"{len(self._datamodule_cache)} datamodules in cache"
        )


data_factory = DataFactory()
