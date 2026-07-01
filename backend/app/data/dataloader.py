from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler

from app.core.logging import logger
from app.data.collate import collate_multimodal_samples
from app.data.multimodal_sample import MultimodalSample


class MultimodalDataLoader:
    def __init__(
        self,
        dataset: Dataset[MultimodalSample],
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | None = None,
        seed: int | None = None,
        collate_fn: Callable | None = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        timeout: float = 0.0,
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._sampler = sampler
        self._seed = seed
        self._prefetch_factor = prefetch_factor
        self._persistent_workers = persistent_workers
        self._timeout = timeout

        if seed is not None:
            self._set_deterministic(seed)

        self._loader = self._build_loader(collate_fn)

    def _set_deterministic(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_loader(self, collate_fn: Callable | None) -> TorchDataLoader:
        generator = torch.Generator()
        if self._seed is not None:
            generator.manual_seed(self._seed)

        worker_init_fn = None
        if self._seed is not None:

            def _worker_init(wk_id: int) -> None:
                torch.manual_seed(self._seed + wk_id)

            worker_init_fn = _worker_init

        return TorchDataLoader(
            dataset=self._dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle if self._sampler is None else False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
            collate_fn=collate_fn or collate_multimodal_samples,
            worker_init_fn=worker_init_fn,
            generator=generator,
            prefetch_factor=(self._prefetch_factor if self._num_workers > 0 else None),
            persistent_workers=self._persistent_workers and self._num_workers > 0,
            timeout=self._timeout,
        )

    @property
    def loader(self) -> TorchDataLoader:
        return self._loader

    @property
    def dataset(self) -> Dataset[MultimodalSample]:
        return self._dataset

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def log_config(self) -> None:
        logger.info(
            f"MultimodalDataLoader: batch_size={self._batch_size}, "
            f"shuffle={self._shuffle}, workers={self._num_workers}, "
            f"pin_memory={self._pin_memory}, drop_last={self._drop_last}, "
            f"seed={self._seed}"
        )
