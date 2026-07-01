from __future__ import annotations

from pathlib import Path

import pytest

from app.datasets import dataset_factory, dataset_registry
from app.datasets.dataset_factory import DatasetFactory
from app.datasets.dataset_registry import DatasetRegistry
from app.datasets.errors import DatasetAlreadyExistsError, DatasetNotFoundError
from app.datasets.metadata import MetadataGenerator


class TestDatasetFactory:
    def test_create_known_adapter(self):
        adapter = dataset_factory.create("uci_har")
        assert adapter.name == "uci_har"
        assert adapter.num_classes == 6

    def test_create_meld_adapter(self):
        adapter = dataset_factory.create("meld")
        assert adapter.name == "meld"

    def test_create_unknown_raises(self):
        with pytest.raises(DatasetNotFoundError):
            dataset_factory.create("completely_unknown")

    def test_list_available(self):
        available = dataset_factory.list_available()
        assert "uci_har" in available
        assert "meld" in available
        assert "ptb_xl" in available
        assert "hateful_memes" in available
        assert "generic" in available

    def test_exists(self):
        assert dataset_factory.exists("uci_har") is True
        assert dataset_factory.exists("nonexistent") is False

    def test_register_and_create_custom(self):
        from app.datasets.adapters.generic import GenericDatasetAdapter

        DatasetFactory.register("my_custom", GenericDatasetAdapter)
        adapter = DatasetFactory.create("my_custom")
        assert adapter.name == "generic"

    def test_singleton_instance(self):
        a1 = dataset_factory.create("uci_har")
        a2 = dataset_factory.create("uci_har")
        assert a1 is a2


class TestDatasetRegistry:
    def test_register_and_get(self):
        reg = DatasetRegistry()
        meta = {"download_status": "not_downloaded", "phase": "test"}
        reg.register("test_ds_1", meta)
        retrieved = reg.get("test_ds_1")
        assert retrieved["download_status"] == "not_downloaded"

    def test_register_duplicate_raises(self, tmp_path: Path):
        reg = DatasetRegistry()
        reg.register("unique_ds", {"key": "value"})
        with pytest.raises(DatasetAlreadyExistsError):
            reg.register("unique_ds", {"key": "value2"})

    def test_get_nonexistent_raises(self):
        reg = DatasetRegistry()
        with pytest.raises(DatasetNotFoundError):
            reg.get("does_not_exist")

    def test_exists(self):
        reg = DatasetRegistry()
        reg.register("exists_test", {"v": 1})
        assert reg.exists("exists_test") is True
        assert reg.exists("no_exists") is False

    def test_unregister(self):
        reg = DatasetRegistry()
        reg.register("to_unregister", {"v": 1})
        assert reg.exists("to_unregister")
        reg.unregister("to_unregister")
        assert not reg.exists("to_unregister")

    def test_update_status(self):
        reg = DatasetRegistry()
        reg.register("status_test", {"download_status": "not_downloaded"})
        reg.update_status("status_test", "download_status", "downloaded")
        meta = reg.get("status_test")
        assert meta["download_status"] == "downloaded"

    def test_list(self):
        reg = DatasetRegistry()
        reg.register("list_test_a", {"a": 1})
        reg.register("list_test_b", {"b": 2})
        items = reg.list()
        names = [i.get("_name") for i in items]
        assert "list_test_a" in names


class TestMetadataGenerator:
    def test_generate_metadata(self):
        gen = MetadataGenerator()
        meta = gen.generate(
            name="test_ds",
            classes=["a", "b"],
            modalities=["image"],
            input_shapes={"image": (3, 224, 224)},
            num_samples=1000,
        )
        assert meta["dataset_name"] == "test_ds"
        assert meta["num_classes"] == 2
        assert meta["num_samples"] == 1000

    def test_save_and_load(self, tmp_path: Path):
        gen = MetadataGenerator()
        meta = gen.generate(
            name="save_test",
            classes=["x"],
            modalities=["text"],
            input_shapes={"text": (128,)},
            num_samples=500,
        )
        gen.save(meta, tmp_path / "save_test")
        loaded = gen.load("save_test", tmp_path / "save_test" / "metadata.json")
        assert loaded is not None
        assert loaded["dataset_name"] == "save_test"
