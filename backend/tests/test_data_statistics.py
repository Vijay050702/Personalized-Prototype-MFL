from __future__ import annotations

import numpy as np
import pytest

from app.data.multimodal_dataset import MultimodalDataset
from app.data.statistics import DatasetStatistics
from app.datasets.base import DatasetLoadResult


@pytest.fixture
def simple_dataset():
    rng = np.random.default_rng(42)
    data = [
        {
            "image": rng.normal(0, 1, (3, 32, 32)).astype(np.float32),
            "text": rng.integers(0, 100, 50).astype(np.int64),
        }
        for _ in range(20)
    ]
    labels = rng.integers(0, 4, 20).astype(np.int64)
    lr = DatasetLoadResult(
        data=data,
        labels=labels,
        metadata={
            "name": "stats_test",
            "modalities": ["image", "text"],
            "num_classes": 4,
        },
    )
    return MultimodalDataset(load_result=lr, dataset_name="stats_test")


@pytest.fixture
def dataset_with_missing():
    rng = np.random.default_rng(42)
    data = []
    for i in range(20):
        sample = {"image": rng.normal(0, 1, (3, 32, 32)).astype(np.float32)}
        if i < 10:
            sample["text"] = rng.integers(0, 100, 50).astype(np.int64)
        else:
            sample["text_missing"] = True
        data.append(sample)
    labels = rng.integers(0, 3, 20).astype(np.int64)
    lr = DatasetLoadResult(
        data=data,
        labels=labels,
        metadata={
            "name": "missing_test",
            "modalities": ["image", "text"],
            "num_classes": 3,
        },
    )
    return MultimodalDataset(load_result=lr, dataset_name="missing_test")


class TestDatasetStatistics:
    def test_basic_statistics(self, simple_dataset):
        stats = DatasetStatistics(simple_dataset)
        assert stats.num_samples == 20
        assert stats.num_classes == 4
        assert stats.class_distribution is not None
        assert sum(stats.class_distribution.values()) == 20

    def test_modality_availability(self, simple_dataset):
        stats = DatasetStatistics(simple_dataset)
        assert stats.modality_availability["image"] == 1.0
        assert stats.modality_availability["text"] == 1.0

    def test_missing_ratio_all_present(self, simple_dataset):
        stats = DatasetStatistics(simple_dataset)
        assert stats.missing_modality_ratio == 0.0

    def test_missing_ratio_with_missing(self, dataset_with_missing):
        stats = DatasetStatistics(dataset_with_missing)
        assert stats.missing_modality_ratio > 0.0

    def test_avg_sequence_lengths(self, simple_dataset):
        stats = DatasetStatistics(simple_dataset)
        assert "text" in stats.avg_sequence_lengths
        assert stats.avg_sequence_lengths["text"] > 0

    def test_to_dict(self, simple_dataset):
        stats = DatasetStatistics(simple_dataset)
        d = stats.to_dict()
        assert d["num_samples"] == 20
        assert d["num_classes"] == 4
        assert "modality_availability" in d
        assert "missing_modality_ratio" in d
        assert "class_distribution" in d

    def test_repr(self, simple_dataset):
        stats = DatasetStatistics(simple_dataset)
        r = repr(stats)
        assert "DatasetStatistics" in r
        assert "samples=20" in r

    def test_empty_dataset(self):
        lr = DatasetLoadResult(
            data=[],
            labels=np.array([], dtype=np.int64),
            metadata={"name": "empty", "modalities": [], "num_classes": 0},
        )
        ds = MultimodalDataset(load_result=lr, dataset_name="empty")
        stats = DatasetStatistics(ds)
        assert stats.num_samples == 0
        assert stats.num_classes == 0
