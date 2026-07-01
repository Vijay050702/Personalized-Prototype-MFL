from __future__ import annotations

import numpy as np
import pytest
import torch

from app.data.factory import DataFactory, data_factory
from app.datasets.base import DatasetLoadResult


@pytest.fixture
def sample_load_result():
    rng = np.random.default_rng(42)
    data = [
        {"sensor": rng.normal(0, 1, (128, 6)).astype(np.float32)} for _ in range(10)
    ]
    labels = rng.integers(0, 2, 10).astype(np.int64)
    return DatasetLoadResult(
        data=data,
        labels=labels,
        metadata={"name": "factory_test", "modalities": ["sensor"], "num_classes": 2},
    )


class TestDataFactory:
    def test_create_dataset(self, sample_load_result):
        factory = DataFactory()
        ds = factory.create_dataset(
            load_result=sample_load_result,
            dataset_name="factory_test",
        )
        assert len(ds) == 10
        assert ds.num_classes == 2

    def test_create_dataset_caching(self, sample_load_result):
        factory = DataFactory()
        ds1 = factory.create_dataset(
            load_result=sample_load_result,
            dataset_name="factory_test",
            cache_key="test_key",
        )
        ds2 = factory.create_dataset(
            load_result=sample_load_result,
            dataset_name="factory_test",
            cache_key="test_key",
        )
        assert ds1 is ds2

    def test_create_dataloader(self, sample_load_result):
        factory = DataFactory()
        ds = factory.create_dataset(
            load_result=sample_load_result,
            dataset_name="factory_test",
        )
        dl = factory.create_dataloader(ds, batch_size=4)
        assert dl.batch_size == 4
        batches = list(dl)
        assert len(batches) > 0

    def test_create_datamodule(self):
        from pathlib import Path
        from app.core.config import settings
        from app.datasets.adapters.uci_har import UCIHARAdapter

        ds_root = Path(settings.datasets_root)
        adapter = UCIHARAdapter()
        adapter.generate_sample(ds_root, num_samples=30)

        factory = DataFactory()
        dm = factory.create_datamodule(
            dataset_name="uci_har",
            batch_size=4,
            val_split=0.1,
            test_split=0.1,
        )
        dm.prepare_data()
        assert dm.statistics is not None
        assert dm.num_classes > 0

    def test_create_datamodule_caching(self):
        factory = DataFactory()
        dm1 = factory.create_datamodule(dataset_name="uci_har", use_cache=True)
        dm2 = factory.create_datamodule(dataset_name="uci_har", use_cache=True)
        assert dm1 is dm2

    def test_clear_cache(self, sample_load_result):
        factory = DataFactory()
        factory.create_dataset(
            load_result=sample_load_result,
            dataset_name="test",
            cache_key="clear_test",
        )
        factory.clear_cache()
        factory.log_available()

    def test_singleton_instance(self):
        assert data_factory is not None
