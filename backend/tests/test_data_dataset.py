from __future__ import annotations

import numpy as np
import pytest
import torch

from app.data.multimodal_dataset import MultimodalDataset
from app.data.multimodal_sample import MultimodalSample
from app.datasets.base import DatasetLoadResult


@pytest.fixture
def sample_load_result():
    rng = np.random.default_rng(42)
    data = [
        {
            "image": rng.normal(0, 1, (3, 32, 32)).astype(np.float32),
            "text": rng.integers(0, 100, 50).astype(np.int64),
        }
        for _ in range(10)
    ]
    labels = rng.integers(0, 3, 10).astype(np.int64)
    return DatasetLoadResult(
        data=data,
        labels=labels,
        metadata={"name": "test_ds", "modalities": ["image", "text"], "num_classes": 3},
    )


@pytest.fixture
def sample_load_result_missing():
    rng = np.random.default_rng(42)
    data = []
    for i in range(10):
        sample = {"image": rng.normal(0, 1, (3, 32, 32)).astype(np.float32)}
        if i % 2 == 0:
            sample["text"] = rng.integers(0, 100, 50).astype(np.int64)
        else:
            sample["text_missing"] = True
        data.append(sample)
    labels = rng.integers(0, 3, 10).astype(np.int64)
    return DatasetLoadResult(
        data=data,
        labels=labels,
        metadata={
            "name": "test_ds_missing",
            "modalities": ["image", "text"],
            "num_classes": 3,
        },
    )


class TestMultimodalDataset:
    def test_create_from_load_result(self, sample_load_result):
        ds = MultimodalDataset(load_result=sample_load_result, dataset_name="test")
        assert len(ds) == 10
        assert ds.num_classes == 3

    def test_getitem_returns_sample(self, sample_load_result):
        ds = MultimodalDataset(load_result=sample_load_result, dataset_name="test")
        sample = ds[0]
        assert isinstance(sample, MultimodalSample)
        assert sample.label in (0, 1, 2)
        assert sample.has_modality("image")

    def test_getitem_with_transform(self, sample_load_result):
        transform = {"image": lambda t: t * 2.0}
        ds = MultimodalDataset(
            load_result=sample_load_result,
            dataset_name="test",
            transforms=transform,
        )
        sample = ds[0]
        assert sample.has_modality("image")

    def test_client_indices_filter(self, sample_load_result):
        ds = MultimodalDataset(
            load_result=sample_load_result,
            dataset_name="test",
            client_indices=[0, 1, 2],
        )
        assert len(ds) == 3

    def test_client_id(self, sample_load_result):
        ds = MultimodalDataset(
            load_result=sample_load_result,
            dataset_name="test",
            client_id="client_0",
        )
        sample = ds[0]
        assert sample.client_id == "client_0"

    def test_class_distribution(self, sample_load_result):
        ds = MultimodalDataset(load_result=sample_load_result, dataset_name="test")
        dist = ds.class_distribution
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 10

    def test_modality_availability(self, sample_load_result):
        ds = MultimodalDataset(load_result=sample_load_result, dataset_name="test")
        avail = ds.modality_availability
        assert "image" in avail
        assert "text" in avail
        assert avail["image"] == 1.0

    def test_modality_availability_with_missing(self, sample_load_result_missing):
        ds = MultimodalDataset(
            load_result=sample_load_result_missing,
            dataset_name="test_missing",
        )
        avail = ds.modality_availability
        assert avail["image"] == 1.0
        assert 0.4 <= avail["text"] <= 0.6

    def test_labels_property(self, sample_load_result):
        ds = MultimodalDataset(load_result=sample_load_result, dataset_name="test")
        assert isinstance(ds.labels, np.ndarray)
        assert len(ds.labels) == 10

    def test_get_sample(self, sample_load_result):
        ds = MultimodalDataset(load_result=sample_load_result, dataset_name="test")
        raw = ds.get_sample(0)
        assert "data" in raw
        assert "label" in raw

    def test_to_load_result(self, sample_load_result):
        ds = MultimodalDataset(load_result=sample_load_result, dataset_name="test")
        lr = ds.to_load_result()
        assert isinstance(lr, DatasetLoadResult)
        assert len(lr) == 10

    def test_raises_without_data(self):
        with pytest.raises(ValueError):
            MultimodalDataset()

    def test_deterministic_indexing(self, sample_load_result):
        ds = MultimodalDataset(load_result=sample_load_result, dataset_name="test")
        s1 = ds[3]
        s2 = ds[3]
        assert s1.label == s2.label
        assert s1.sample_id == s2.sample_id

    def test_lazy_load(self, sample_load_result):
        ds = MultimodalDataset(
            load_result=sample_load_result,
            dataset_name="test",
            lazy_load=True,
        )
        assert len(ds) == 10
        sample = ds[5]
        assert isinstance(sample, MultimodalSample)
