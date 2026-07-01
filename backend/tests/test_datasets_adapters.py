from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.datasets.adapters.uci_har import UCIHARAdapter
from app.datasets.adapters.meld import MELDAdapter
from app.datasets.adapters.ptb_xl import PTBXLAdapter
from app.datasets.adapters.hateful_memes import HatefulMemesAdapter
from app.datasets.adapters.generic import GenericDatasetAdapter


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    return tmp_path / "datasets"


class TestDatasetAdapters:
    def test_uci_har_metadata(self):
        adapter = UCIHARAdapter()
        meta = adapter.get_metadata()
        assert meta["name"] == "uci_har"
        assert meta["num_classes"] == 6
        assert meta["modalities"] == ["sensor"]
        assert meta["num_samples"] == 10299

    def test_uci_har_generate_and_load(self, sample_dir: Path):
        adapter = UCIHARAdapter()
        adapter.generate_sample(sample_dir, num_samples=50)
        result = adapter.load(sample_dir, split="train")
        assert len(result) > 0
        assert result.metadata["name"] == "uci_har"
        assert "sensor" in result.data[0]

    def test_meld_metadata(self):
        adapter = MELDAdapter()
        meta = adapter.get_metadata()
        assert meta["name"] == "meld"
        assert meta["num_classes"] == 7
        assert "text" in meta["modalities"]

    def test_meld_generate_and_load(self, sample_dir: Path):
        adapter = MELDAdapter()
        adapter.generate_sample(sample_dir, num_samples=30)
        result = adapter.load(sample_dir, split="train")
        assert len(result) > 0
        assert "text" in result.data[0]
        assert "audio" in result.data[0]

    def test_ptbxl_metadata(self):
        adapter = PTBXLAdapter()
        meta = adapter.get_metadata()
        assert meta["name"] == "ptb_xl"
        assert meta["num_classes"] == 5

    def test_ptbxl_generate_and_load(self, sample_dir: Path):
        adapter = PTBXLAdapter()
        adapter.generate_sample(sample_dir, num_samples=30)
        result = adapter.load(sample_dir, split="train")
        assert len(result) > 0
        assert "sensor" in result.data[0]

    def test_hateful_memes_metadata(self):
        adapter = HatefulMemesAdapter()
        meta = adapter.get_metadata()
        assert meta["name"] == "hateful_memes"
        assert meta["num_classes"] == 2

    def test_hateful_memes_generate_and_load(self, sample_dir: Path):
        adapter = HatefulMemesAdapter()
        adapter.generate_sample(sample_dir, num_samples=30)
        result = adapter.load(sample_dir, split="train")
        assert len(result) > 0
        assert "image" in result.data[0]
        assert "text" in result.data[0]

    def test_generic_metadata(self):
        adapter = GenericDatasetAdapter(
            {"name": "custom", "modalities": ["image", "text"]}
        )
        meta = adapter.get_metadata()
        assert meta["name"] == "custom"
        assert meta["is_generic"] is True

    def test_generic_generate_and_load(self, sample_dir: Path):
        adapter = GenericDatasetAdapter(
            {"name": "custom_test", "modalities": ["image", "sensor"]}
        )
        adapter.generate_sample(sample_dir, num_samples=20)
        result = adapter.load(sample_dir, split="train")
        assert len(result) > 0
        assert "image" in result.data[0]
        assert "sensor" in result.data[0]

    def test_generic_raises_on_unknown(self):
        adapter = GenericDatasetAdapter(
            {"name": "unknown_dataset", "modalities": ["image"]}
        )
        with pytest.raises(FileNotFoundError):
            adapter.load(Path("/nonexistent"), split="train")


class TestAdapterValidation:
    def test_validate_nonexistent(self, tmp_path: Path):
        adapter = UCIHARAdapter()
        result = adapter.validate(tmp_path)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_generated_sample(self, sample_dir: Path):
        adapter = UCIHARAdapter()
        adapter.generate_sample(sample_dir, num_samples=50)
        result = adapter.validate(sample_dir)
        assert result["is_valid"] is True
