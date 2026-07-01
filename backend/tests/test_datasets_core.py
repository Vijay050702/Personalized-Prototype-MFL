from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.datasets.cache import DatasetCache
from app.datasets.downloader import DownloadManager, compute_checksum
from app.datasets.errors import DatasetValidationError
from app.datasets.loader import DatasetLoader
from app.datasets.missing_modality import MissingModalitySimulator
from app.datasets.base import DatasetLoadResult
from app.datasets.transforms import (
    build_vocabulary,
    normalize_mean_std,
    one_hot_encode,
    pad_sequences,
    resize,
    sliding_window,
    text_to_ids,
    to_spectrogram,
)
from app.datasets.validators import (
    validate_class_balance,
    validate_dataset,
    validate_dataset_files,
    validate_labels,
    validate_modalities,
)


class TestTransforms:
    def test_resize(self):
        data = np.random.rand(100, 100, 3)
        resized = resize(data, (224, 224))
        assert resized.shape == (224, 224, 3)

    def test_normalize_mean_std(self):
        data = np.random.rand(10, 10, 3).astype(np.float32)
        normalized = normalize_mean_std(data)
        assert normalized.shape == data.shape
        assert abs(normalized.mean()) < 1.0

    def test_pad_sequences(self):
        seqs = [np.array([1, 2, 3]), np.array([4, 5])]
        padded = pad_sequences(seqs, max_length=5)
        assert padded.shape == (2, 5)

    def test_one_hot_encode(self):
        labels = np.array([0, 1, 2, 3])
        oh = one_hot_encode(labels, 4)
        assert oh.shape == (4, 4)
        assert oh[0, 0] == 1.0

    def test_sliding_window(self):
        data = np.arange(100)
        windows = sliding_window(data, 10, 5)
        assert len(windows) == 19

    def test_to_spectrogram(self):
        audio = np.sin(np.linspace(0, 1000, 16000))
        spec = to_spectrogram(audio, n_fft=256, hop_length=128, n_mels=64)
        assert spec.shape[0] == 64

    def test_build_vocabulary(self):
        texts = ["hello world", "hello test world", "world of python"]
        vocab = build_vocabulary(texts, max_vocab_size=100)
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert "hello" in vocab
        assert "world" in vocab

    def test_text_to_ids(self):
        texts = ["hello world", "test"]
        vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3, "test": 4}
        ids = text_to_ids(texts, vocab, max_length=5)
        assert ids.shape == (2, 5)
        assert ids[0, 0] == 2


class TestValidators:
    def test_validate_modalities_valid(self):
        errors = validate_modalities(["image", "text"])
        assert len(errors) == 0

    def test_validate_modalities_invalid(self):
        errors = validate_modalities(["invalid_modality"])
        assert len(errors) > 0

    def test_validate_dataset_files_missing(self, tmp_path: Path):
        errors = validate_dataset_files(tmp_path, ["nonexistent.txt"])
        assert len(errors) > 0

    def test_validate_labels_consistency(self):
        errors = validate_labels([0, 1, 2, 3], expected_classes=4)
        assert len(errors) == 0

    def test_validate_labels_wrong_count(self):
        errors = validate_labels([0, 1], expected_classes=5)
        assert len(errors) > 0

    def test_validate_class_balance(self):
        labels = [0] * 90 + [1] * 10
        warnings = validate_class_balance(labels, imbalance_threshold=0.15)
        assert any("underrepresented" in w for w in warnings)

    def test_validate_dataset_nonexistent(self, tmp_path: Path):
        result = validate_dataset(
            "test", tmp_path / "nonexistent", modalities=["image"]
        )
        assert result["is_valid"] is False

    def test_validate_dataset_error_raised(self, tmp_path: Path):
        with pytest.raises(DatasetValidationError):
            from app.datasets.validators import validate_dataset_exists

            validate_dataset_exists(tmp_path / "nonexistent")


class TestCache:
    def test_cache_set_get(self, tmp_path: Path):
        cache = DatasetCache(cache_root=str(tmp_path))
        cache.set("test_key", {"data": [1, 2, 3]})
        result = cache.get("test_key")
        assert result == {"data": [1, 2, 3]}

    def test_cache_miss(self, tmp_path: Path):
        cache = DatasetCache(cache_root=str(tmp_path))
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_json(self, tmp_path: Path):
        cache = DatasetCache(cache_root=str(tmp_path))
        cache.set_json("json_test", {"name": "test", "value": 42})
        result = cache.get_json("json_test")
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_cache_exists(self, tmp_path: Path):
        cache = DatasetCache(cache_root=str(tmp_path))
        cache.set("exists_test", True)
        assert cache.exists("exists_test")
        assert not cache.exists("does_not_exist")

    def test_cache_invalidate(self, tmp_path: Path):
        cache = DatasetCache(cache_root=str(tmp_path))
        cache.set("to_remove", "data")
        assert cache.exists("to_remove")
        cache.invalidate("to_remove")
        assert not cache.exists("to_remove")

    def test_cache_clear(self, tmp_path: Path):
        cache = DatasetCache(cache_root=str(tmp_path))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert not cache.exists("a")
        assert not cache.exists("b")


class TestLoader:
    def test_load_from_memory(self):
        loader = DatasetLoader()
        data = [
            {"image": np.random.rand(3, 224, 224).astype(np.float32)} for _ in range(10)
        ]
        labels = np.random.randint(0, 2, 10)
        result = loader.load_from_memory(data, labels, ["image"], "test_ds")
        assert len(result) == 10
        assert result.metadata["num_classes"] == 2

    def test_load_result_getitem(self):
        data = [{"sensor": np.array([1.0, 2.0])}, {"sensor": np.array([3.0, 4.0])}]
        labels = np.array([0, 1])
        result = DatasetLoadResult(data, labels, {"name": "test"})
        d, l = result[1]
        assert l == 1
        assert np.array_equal(d["sensor"], [3.0, 4.0])


class TestMissingModalitySimulator:
    @pytest.fixture
    def sample_dataset(self) -> DatasetLoadResult:
        data = [
            {
                "image": np.random.rand(3, 32, 32).astype(np.float32),
                "text": np.array([1, 2, 3]),
            }
            for _ in range(20)
        ]
        labels = np.random.randint(0, 2, 20)
        return DatasetLoadResult(
            data, labels, {"name": "test", "modalities": ["image", "text"]}
        )

    def test_random_missing(self, sample_dataset: DatasetLoadResult):
        sim = MissingModalitySimulator()
        result = sim.apply_random(sample_dataset, missing_ratio=1.0, seed=42)
        missing_count = sum(1 for d in result.data if any("_missing" in k for k in d))
        assert missing_count > 0

    def test_modality_wise_missing(self, sample_dataset: DatasetLoadResult):
        sim = MissingModalitySimulator()
        result = sim.apply_modality_wise(
            sample_dataset, modalities_to_drop=["text"], missing_ratio=1.0, seed=42
        )
        for d in result.data:
            if "text_missing" in d:
                assert len(d["text"]) == 0

    def test_client_wise_missing(self, sample_dataset: DatasetLoadResult):
        sim = MissingModalitySimulator()
        for i, d in enumerate(sample_dataset.data):
            d["_client"] = "client_a" if i < 10 else "client_b"
        missing_map = {"client_a": ["image"], "client_b": ["text"]}
        result = sim.apply_client_wise(sample_dataset, missing_map=missing_map, seed=42)
        for i, d in enumerate(result.data):
            if i < 10:
                assert "text_missing" in d
                assert "image_missing" not in d
            else:
                assert "image_missing" in d
                assert "text_missing" not in d

    def test_apply_dispatcher(self, sample_dataset: DatasetLoadResult):
        sim = MissingModalitySimulator()
        r1 = sim.apply(sample_dataset, strategy="random", missing_ratio=0.5, seed=42)
        assert r1.metadata["missing_modality_strategy"] == "random"
        r2 = sim.apply(
            sample_dataset,
            strategy="modality_wise",
            missing_ratio=0.5,
            modalities=["image"],
            seed=42,
        )
        assert r2.metadata["missing_modality_strategy"] == "modality_wise"

    def test_apply_invalid_strategy(self, sample_dataset: DatasetLoadResult):
        sim = MissingModalitySimulator()
        with pytest.raises(ValueError):
            sim.apply(sample_dataset, strategy="invalid")

    def test_reproducible_seed(self, sample_dataset: DatasetLoadResult):
        sim = MissingModalitySimulator()
        r1 = sim.apply_random(sample_dataset, missing_ratio=0.5, seed=42)
        r2 = sim.apply_random(sample_dataset, missing_ratio=0.5, seed=42)
        for i in range(len(r1.data)):
            m1 = [k for k in r1.data[i] if k.endswith("_missing")]
            m2 = [k for k in r2.data[i] if k.endswith("_missing")]
            assert m1 == m2
