from __future__ import annotations

import numpy as np
import pytest
import torch

from app.data.multimodal_sample import MultimodalSample


class TestMultimodalSample:
    def test_create_with_all_modalities(self):
        sample = MultimodalSample(
            sample_id=1,
            label=0,
            client_id="client_0",
            image=torch.randn(3, 32, 32),
            text=torch.randint(0, 100, (50,)),
            audio=torch.randn(80, 100),
            sensor=torch.randn(128, 6),
        )
        assert sample.sample_id == 1
        assert sample.label == 0
        assert sample.client_id == "client_0"
        assert sample.num_available() == 4

    def test_create_with_missing_modalities(self):
        sample = MultimodalSample(
            sample_id=2,
            label=1,
            image=torch.randn(3, 64, 64),
        )
        assert sample.has_modality("image") is True
        assert sample.has_modality("text") is False
        assert sample.has_modality("audio") is False
        assert sample.num_available() == 1

    def test_modality_mask(self):
        sample = MultimodalSample(
            sample_id=3,
            label=2,
            image=torch.randn(3, 32, 32),
            sensor=torch.randn(128, 6),
        )
        mask = sample.modality_mask
        assert mask["image"] is True
        assert mask["text"] is False
        assert mask["audio"] is False
        assert mask["sensor"] is True

    def test_modality_mask_tensor(self):
        sample = MultimodalSample(sample_id=4, label=0, image=torch.randn(3, 32, 32))
        tensor = sample.modality_mask_tensor
        assert tensor.dtype == torch.bool
        assert tensor[0].item() is True  # image
        assert tensor[1].item() is False  # text
        assert tensor[2].item() is False  # audio
        assert tensor[3].item() is False  # sensor

    def test_set_tensor(self):
        sample = MultimodalSample(sample_id=5, label=0)
        assert sample.has_modality("image") is False
        sample.set_tensor("image", torch.randn(3, 32, 32))
        assert sample.has_modality("image") is True
        sample.set_tensor("image", None)
        assert sample.has_modality("image") is False

    def test_get_tensor(self):
        img = torch.randn(3, 32, 32)
        sample = MultimodalSample(sample_id=1, label=0, image=img)
        assert sample.get_tensor("image") is img
        assert sample.get_tensor("text") is None

    def test_available_modalities(self):
        sample = MultimodalSample(
            sample_id=1,
            label=0,
            image=torch.randn(3, 32, 32),
            audio=torch.randn(80, 100),
        )
        avail = sample.available_modalities
        assert "image" in avail
        assert "audio" in avail
        assert "text" not in avail
        assert "sensor" not in avail

    def test_missing_modalities(self):
        sample = MultimodalSample(
            sample_id=1,
            label=0,
            image=torch.randn(3, 32, 32),
        )
        missing = sample.missing_modalities
        assert "text" in missing
        assert "audio" in missing
        assert "sensor" in missing
        assert "image" not in missing

    def test_repr(self):
        sample = MultimodalSample(sample_id=1, label=0, image=torch.randn(3, 32, 32))
        r = repr(sample)
        assert "MultimodalSample" in r
        assert "id=1" in r
        assert "label=0" in r

    def test_to_dict(self):
        sample = MultimodalSample(
            sample_id=1,
            label=0,
            client_id="c1",
            image=torch.randn(3, 32, 32),
            metadata={"source": "test"},
        )
        d = sample.to_dict()
        assert d["sample_id"] == 1
        assert d["label"] == 0
        assert d["client_id"] == "c1"
        assert d["image"] is not None
        assert "modality_mask" in d
        assert d["metadata"]["source"] == "test"

    def test_from_dict_with_tensors(self):
        raw = {
            "image": np.random.randn(3, 32, 32).astype(np.float32),
            "text": np.array([1, 2, 3]),
            "label": 1,
        }
        sample = MultimodalSample.from_dict(raw, sample_id=10)
        assert sample.sample_id == 10
        assert sample.label == 1
        assert sample.has_modality("image")
        assert sample.has_modality("text")

    def test_from_dict_with_missing_flag(self):
        raw = {
            "image": np.random.randn(3, 32, 32).astype(np.float32),
            "text_missing": True,
            "label": 0,
        }
        sample = MultimodalSample.from_dict(raw, sample_id=5)
        assert sample.has_modality("image") is True
        assert sample.has_modality("text") is False

    def test_from_dict_with_metadata(self):
        raw = {
            "sensor": np.random.randn(128, 6).astype(np.float32),
            "label": 2,
            "extra": "val",
        }
        sample = MultimodalSample.from_dict(raw, sample_id=3)
        assert sample.metadata.get("extra") == "val"
