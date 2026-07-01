from __future__ import annotations

import numpy as np
import pytest
import torch

from app.data.collate import collate_multimodal_samples
from app.data.dataloader import MultimodalDataLoader
from app.data.multimodal_dataset import MultimodalDataset
from app.datasets.base import DatasetLoadResult


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    data = [
        {"sensor": rng.normal(0, 1, (128, 6)).astype(np.float32)} for _ in range(20)
    ]
    labels = rng.integers(0, 2, 20).astype(np.int64)
    lr = DatasetLoadResult(
        data=data,
        labels=labels,
        metadata={"name": "test", "modalities": ["sensor"], "num_classes": 2},
    )
    return MultimodalDataset(load_result=lr, dataset_name="test")


class TestMultimodalDataLoader:
    def test_create_dataloader(self, small_dataset):
        dl = MultimodalDataLoader(
            dataset=small_dataset,
            batch_size=4,
            shuffle=False,
        )
        assert len(dl) == 5  # 20 / 4

    def test_iteration_returns_batches(self, small_dataset):
        dl = MultimodalDataLoader(
            dataset=small_dataset,
            batch_size=4,
            shuffle=False,
        )
        batches = list(dl)
        assert len(batches) == 5
        for batch in batches:
            assert batch.batch_size == 4
            assert batch.labels is not None
            assert batch.labels.size(0) == 4

    def test_shuffle_deterministic(self, small_dataset):
        dl1 = MultimodalDataLoader(
            dataset=small_dataset,
            batch_size=4,
            shuffle=True,
            seed=42,
        )
        dl2 = MultimodalDataLoader(
            dataset=small_dataset,
            batch_size=4,
            shuffle=True,
            seed=42,
        )
        batches1 = [(b.labels.tolist(), b.sample_ids) for b in dl1]
        batches2 = [(b.labels.tolist(), b.sample_ids) for b in dl2]
        assert batches1 == batches2

    def test_drop_last(self, small_dataset):
        dl = MultimodalDataLoader(
            dataset=small_dataset,
            batch_size=6,
            shuffle=False,
            drop_last=True,
        )
        assert len(dl) == 3  # 20 / 6 = 3 (drops remainder)

    def test_log_config(self, small_dataset):
        dl = MultimodalDataLoader(
            dataset=small_dataset,
            batch_size=8,
            shuffle=False,
            seed=42,
        )
        dl.log_config()

    @property
    def loader(self, small_dataset):
        dl = MultimodalDataLoader(
            dataset=small_dataset,
            batch_size=4,
        )
        assert dl.loader is not None

    def test_batch_size_property(self, small_dataset):
        dl = MultimodalDataLoader(dataset=small_dataset, batch_size=8)
        assert dl.batch_size == 8

    def test_dataset_property(self, small_dataset):
        dl = MultimodalDataLoader(dataset=small_dataset, batch_size=4)
        assert dl.dataset is small_dataset
