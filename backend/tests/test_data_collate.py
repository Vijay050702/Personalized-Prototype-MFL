from __future__ import annotations

import pytest
import torch

from app.data.collate import (
    collate_multimodal_samples,
    pad_sequence,
    stack_tensors,
)
from app.data.multimodal_sample import MultimodalSample


class TestPadSequence:
    def test_pad_sequence_varying_lengths(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
        padded = pad_sequence(seqs, padding_value=0)
        assert padded.shape == (3, 3)
        assert padded[0].tolist() == [1, 2, 3]
        assert padded[1].tolist() == [4, 5, 0]
        assert padded[2].tolist() == [6, 0, 0]

    def test_pad_sequence_same_length(self):
        seqs = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        padded = pad_sequence(seqs, padding_value=0)
        assert padded.shape == (2, 2)


class TestStackTensors:
    def test_stack_same_shape(self):
        tensors = [torch.randn(3, 32, 32) for _ in range(4)]
        stacked = stack_tensors(tensors)
        assert stacked.shape == (4, 3, 32, 32)

    def test_stack_varying_shapes(self):
        tensors = [torch.randn(3, 32), torch.randn(3, 64)]
        stacked = stack_tensors(tensors, padding_value=0.0)
        assert stacked.shape == (2, 3, 64)


class TestCollateMultimodal:
    def test_collate_empty(self):
        batch = collate_multimodal_samples([])
        assert batch.batch_size == 0

    def test_collate_single_modality(self):
        samples = [
            MultimodalSample(
                sample_id=i,
                label=i % 2,
                image=torch.randn(3, 32, 32),
            )
            for i in range(4)
        ]
        batch = collate_multimodal_samples(samples)
        assert batch.batch_size == 4
        assert batch.images is not None
        assert batch.images.shape == (4, 3, 32, 32)
        assert batch.labels is not None
        assert batch.labels.tolist() == [0, 1, 0, 1]

    def test_collate_mixed_modalities(self):
        samples = [
            MultimodalSample(
                sample_id=0,
                label=0,
                image=torch.randn(3, 32, 32),
                text=torch.tensor([1, 2, 3]),
            ),
            MultimodalSample(
                sample_id=1,
                label=1,
                image=torch.randn(3, 32, 32),
            ),
        ]
        batch = collate_multimodal_samples(samples)
        assert batch.batch_size == 2
        assert batch.images is not None
        assert batch.images.shape == (2, 3, 32, 32)
        assert batch.text is not None
        assert batch.modality_masks is not None
        assert batch.modality_masks[0, 0].item() is True  # sample 0 has image
        assert batch.modality_masks[1, 0].item() is True  # sample 1 has image
        assert batch.modality_masks[0, 1].item() is True  # sample 0 has text
        assert batch.modality_masks[1, 1].item() is False  # sample 1 no text

    def test_collate_variable_length_text(self):
        samples = [
            MultimodalSample(
                sample_id=i,
                label=0,
                text=torch.randint(0, 100, (10 + i,)),
            )
            for i in range(3)
        ]
        batch = collate_multimodal_samples(samples, pad_text=True)
        assert batch.text is not None
        assert batch.text.shape == (3, 12)

    def test_collate_modality_masks(self):
        samples = [
            MultimodalSample(sample_id=0, label=0, image=torch.randn(3, 32, 32)),
            MultimodalSample(sample_id=1, label=1, audio=torch.randn(80, 100)),
        ]
        batch = collate_multimodal_samples(samples)
        assert batch.modality_masks is not None
        assert batch.modality_masks[0, 0].item() is True  # image
        assert batch.modality_masks[0, 1].item() is False  # no text
        assert batch.modality_masks[1, 2].item() is True  # audio

    def test_collate_client_ids(self):
        samples = [
            MultimodalSample(
                sample_id=i, label=0, client_id=f"c{i}", image=torch.randn(3, 32, 32)
            )
            for i in range(3)
        ]
        batch = collate_multimodal_samples(samples)
        assert batch.client_ids == ["c0", "c1", "c2"]

    def test_collate_sample_ids(self):
        samples = [
            MultimodalSample(sample_id=i, label=0, image=torch.randn(3, 32, 32))
            for i in range(3)
        ]
        batch = collate_multimodal_samples(samples)
        assert batch.sample_ids == [0, 1, 2]

    def test_collate_batch_metadata(self):
        samples = [
            MultimodalSample(sample_id=0, label=0, image=torch.randn(3, 32, 32)),
            MultimodalSample(
                sample_id=1,
                label=0,
                image=torch.randn(3, 32, 32),
                text=torch.tensor([1, 2]),
            ),
        ]
        batch = collate_multimodal_samples(samples)
        assert batch.metadata["batch_size"] == 2
        assert "modality_counts" in batch.metadata
