from __future__ import annotations

import pytest
import torch

from app.data.multimodal_batch import MultimodalBatch


class TestMultimodalBatch:
    def test_empty_batch(self):
        batch = MultimodalBatch()
        assert batch.batch_size == 0
        assert len(batch) == 0
        assert batch.available_modalities == []

    def test_create_batch(self):
        batch = MultimodalBatch(
            images=torch.randn(4, 3, 32, 32),
            labels=torch.tensor([0, 1, 0, 1]),
            client_ids=["c1", "c2", "c1", "c2"],
            modality_masks=torch.ones(4, 4, dtype=torch.bool),
            sample_ids=[0, 1, 2, 3],
        )
        assert batch.batch_size == 4
        assert len(batch) == 4
        assert "image" in batch.available_modalities
        assert batch.labels is not None
        assert len(batch.client_ids) == 4

    def test_get_set_modality_tensor(self):
        batch = MultimodalBatch(
            images=torch.randn(2, 3, 32, 32),
            labels=torch.tensor([0, 1]),
        )
        assert batch.has_modality("image") is True
        assert batch.has_modality("text") is False

        batch.set_modality_tensor("text", torch.randint(0, 100, (2, 50)))
        assert batch.has_modality("text") is True
        assert batch.text is not None
        assert batch.text.size(0) == 2

    def test_to_dict(self):
        batch = MultimodalBatch(
            images=torch.randn(2, 3, 32, 32),
            labels=torch.tensor([0, 1]),
            metadata={"source": "test"},
        )
        d = batch.to_dict()
        assert d["batch_size"] == 2
        assert d["images"] is not None
        assert d["labels"] is not None
        assert d["metadata"]["source"] == "test"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="pin_memory requires CUDA"
    )
    def test_pin_memory(self):
        batch = MultimodalBatch(
            images=torch.randn(2, 3, 32, 32),
            labels=torch.tensor([0, 1]),
        )
        pinned = batch.pin_memory()
        assert pinned.images is not None
        assert pinned.images.is_pinned()

    def test_to_device(self):
        batch = MultimodalBatch(
            images=torch.randn(2, 3, 32, 32),
            labels=torch.tensor([0, 1]),
        )
        device = torch.device("cpu")
        moved = batch.to(device)
        assert moved.images is not None
        assert moved.images.device == device

    def test_repr(self):
        batch = MultimodalBatch(
            images=torch.randn(3, 3, 32, 32),
            labels=torch.tensor([0, 1, 2]),
        )
        r = repr(batch)
        assert "MultimodalBatch" in r
        assert "size=3" in r
