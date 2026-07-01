from __future__ import annotations

import pytest
import torch

from app.data.multimodal_batch import MultimodalBatch
from app.data.multimodal_sample import MultimodalSample
from app.data.validation import (
    CorruptedSampleError,
    DataValidationError,
    EmptyTensorError,
    InvalidLabelError,
    InvalidModalityMaskError,
    ShapeMismatchError,
    validate_batch_consistency,
    validate_batch_tensors,
    validate_label,
    validate_modality_mask,
    validate_sample,
    validate_sample_tensor,
)


class TestValidateSampleTensor:
    def test_valid_tensor(self):
        t = torch.randn(3, 32, 32)
        validate_sample_tensor(t, "test")

    def test_empty_tensor(self):
        with pytest.raises(EmptyTensorError):
            validate_sample_tensor(torch.tensor([]), "empty")

    def test_nan_tensor(self):
        with pytest.raises(CorruptedSampleError):
            validate_sample_tensor(torch.tensor([float("nan")]), "nan")

    def test_inf_tensor(self):
        with pytest.raises(CorruptedSampleError):
            validate_sample_tensor(torch.tensor([float("inf")]), "inf")

    def test_non_tensor(self):
        with pytest.raises(DataValidationError):
            validate_sample_tensor([1, 2, 3], "list")


class TestValidateLabel:
    def test_valid_label(self):
        validate_label(0)
        validate_label(5)

    def test_negative_label(self):
        with pytest.raises(InvalidLabelError):
            validate_label(-1)

    def test_label_out_of_range(self):
        with pytest.raises(InvalidLabelError):
            validate_label(10, num_classes=5)


class TestValidateModalityMask:
    def test_valid_mask(self):
        mask = {"image": True, "text": False, "audio": False, "sensor": True}
        validate_modality_mask(mask)

    def test_invalid_type(self):
        with pytest.raises(InvalidModalityMaskError):
            validate_modality_mask("not_a_dict")

    def test_missing_key(self):
        with pytest.raises(InvalidModalityMaskError):
            validate_modality_mask({"image": True})

    def test_non_bool_value(self):
        with pytest.raises(InvalidModalityMaskError):
            validate_modality_mask(
                {"image": True, "text": "yes", "audio": False, "sensor": False}
            )

    def test_all_missing(self):
        with pytest.raises(InvalidModalityMaskError):
            validate_modality_mask(
                {"image": False, "text": False, "audio": False, "sensor": False}
            )


class TestValidateSample:
    def test_valid_sample(self):
        sample = MultimodalSample(
            sample_id=1,
            label=0,
            image=torch.randn(3, 32, 32),
        )
        validate_sample(sample)

    def test_no_modalities(self):
        sample = MultimodalSample(sample_id=1, label=0)
        with pytest.raises(CorruptedSampleError):
            validate_sample(sample)

    def test_with_label_range(self):
        sample = MultimodalSample(
            sample_id=1,
            label=3,
            image=torch.randn(3, 32, 32),
        )
        validate_sample(sample, num_classes=5)
        with pytest.raises(InvalidLabelError):
            validate_sample(sample, num_classes=2)


class TestValidateBatchTensors:
    def test_valid_tensors(self):
        tensors = {
            "image": torch.randn(4, 3, 32, 32),
            "text": torch.randint(0, 100, (4, 50)),
        }
        validate_batch_tensors(tensors, 4)

    def test_mismatched_batch_size(self):
        tensors = {"image": torch.randn(3, 3, 32, 32)}
        with pytest.raises(ShapeMismatchError):
            validate_batch_tensors(tensors, 4)

    def test_none_tensor_skipped(self):
        tensors: dict = {"image": None, "text": torch.randint(0, 100, (4, 50))}
        validate_batch_tensors(tensors, 4)


class TestValidateBatchConsistency:
    def test_valid_batch(self):
        batch = MultimodalBatch(
            images=torch.randn(3, 3, 32, 32),
            labels=torch.tensor([0, 1, 0]),
            client_ids=["c1", "c2", "c1"],
            modality_masks=torch.ones(3, 4, dtype=torch.bool),
            sample_ids=[0, 1, 2],
        )
        validate_batch_consistency(batch)

    def test_invalid_type(self):
        with pytest.raises(DataValidationError):
            validate_batch_consistency("not_a_batch")
