from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.core.config import settings
from app.data.datamodule import DataModule
from app.data.modality import MODALITY_KEYS
from app.datasets.adapters.uci_har import UCIHARAdapter


@pytest.fixture(scope="module", autouse=True)
def generate_uci_har_sample():
    ds_root = Path(settings.datasets_root)
    adapter = UCIHARAdapter()
    adapter.generate_sample(ds_root, num_samples=50)
    yield


class TestDataModule:
    def test_prepare_data(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=8,
            val_split=0.1,
            test_split=0.1,
        )
        dm.prepare_data()
        assert dm.statistics is not None
        assert dm.num_classes > 0

    def test_setup_creates_splits(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=8,
            val_split=0.2,
            test_split=0.2,
        )
        dm.setup()
        train_ds = dm._train_dataset
        val_ds = dm._val_dataset
        test_ds = dm._test_dataset
        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total > 0

    def test_train_dataloader(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=4,
            val_split=0.1,
            test_split=0.1,
        )
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert batch.batch_size <= 4
        assert batch.labels is not None

    def test_val_dataloader(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=4,
            val_split=0.1,
            test_split=0.1,
        )
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        assert batch.labels is not None

    def test_test_dataloader(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=4,
            val_split=0.1,
            test_split=0.1,
        )
        dl = dm.test_dataloader()
        batch = next(iter(dl))
        assert batch.labels is not None

    def test_predict_dataloader(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=4,
            val_split=0.1,
            test_split=0.1,
        )
        dl = dm.predict_dataloader()
        batch = next(iter(dl))
        assert batch.labels is not None

    def test_statistics_property(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=4,
        )
        dm.prepare_data()
        stats = dm.statistics
        assert stats is not None
        assert stats.num_samples > 0
        assert stats.num_classes > 0

    def test_num_classes_property(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=4,
        )
        dm.prepare_data()
        assert dm.num_classes > 0

    def test_log_summary(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=4,
        )
        dm.prepare_data()
        dm.log_summary()

    def test_dataset_name_property(self):
        dm = DataModule(dataset_name="uci_har")
        assert dm.dataset_name == "uci_har"

    def test_batch_has_modality_mask(self):
        dm = DataModule(
            dataset_name="uci_har",
            batch_size=4,
        )
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert batch.modality_masks is not None
        assert batch.modality_masks.dtype == torch.bool
