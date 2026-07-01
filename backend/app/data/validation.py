from __future__ import annotations

from typing import Any, Sequence

import torch

from app.data.modality import MODALITY_KEYS, modality_mask_from_dict
from app.data.multimodal_sample import MultimodalSample


class DataValidationError(Exception):
    pass


class EmptyTensorError(DataValidationError):
    pass


class InvalidLabelError(DataValidationError):
    pass


class InvalidModalityMaskError(DataValidationError):
    pass


class CorruptedSampleError(DataValidationError):
    pass


class ShapeMismatchError(DataValidationError):
    pass


class MissingMetadataError(DataValidationError):
    pass


def validate_sample_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    if not isinstance(tensor, torch.Tensor):
        raise DataValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.numel() == 0:
        raise EmptyTensorError(f"{name} is empty")
    if torch.isnan(tensor).any():
        raise CorruptedSampleError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise CorruptedSampleError(f"{name} contains Inf values")


def validate_label(label: int, num_classes: int | None = None) -> None:
    if not isinstance(label, (int, float)):
        raise InvalidLabelError(f"Label must be an int, got {type(label)}")
    label_int = int(label)
    if label_int < 0:
        raise InvalidLabelError(f"Label must be non-negative, got {label_int}")
    if num_classes is not None and label_int >= num_classes:
        raise InvalidLabelError(
            f"Label {label_int} out of range [0, {num_classes - 1}]"
        )


def validate_modality_mask(mask: dict[str, bool]) -> None:
    if not isinstance(mask, dict):
        raise InvalidModalityMaskError(
            f"Modality mask must be a dict, got {type(mask)}"
        )
    for mod in MODALITY_KEYS:
        if mod not in mask:
            raise InvalidModalityMaskError(f"Modality mask missing key '{mod}'")
        if not isinstance(mask[mod], bool):
            raise InvalidModalityMaskError(
                f"Modality mask value for '{mod}' must be bool, got {type(mask[mod])}"
            )
    if not any(mask.values()):
        raise InvalidModalityMaskError("All modalities are missing in mask")


def validate_sample(
    sample: MultimodalSample,
    num_classes: int | None = None,
) -> None:
    validate_label(sample.label, num_classes)
    if sample.num_available() == 0:
        raise CorruptedSampleError(
            f"Sample {sample.sample_id} has no available modalities"
        )
    validate_modality_mask(sample.modality_mask)
    for mod in sample.available_modalities:
        tensor = sample.get_tensor(mod)
        if tensor is not None:
            validate_sample_tensor(tensor, f"{mod} tensor")


def validate_batch_tensors(
    tensors: dict[str, torch.Tensor | None],
    expected_batch_size: int,
) -> None:
    for mod, tensor in tensors.items():
        if tensor is None:
            continue
        if tensor.size(0) != expected_batch_size:
            raise ShapeMismatchError(
                f"{mod} tensor batch size {tensor.size(0)} "
                f"does not match expected {expected_batch_size}"
            )


def validate_batch_consistency(batch: Any) -> None:
    from app.data.multimodal_batch import MultimodalBatch

    if not isinstance(batch, MultimodalBatch):
        raise DataValidationError(f"Expected MultimodalBatch, got {type(batch)}")
    bs = batch.batch_size
    if bs == 0:
        return
    if batch.client_ids and len(batch.client_ids) != bs:
        raise ShapeMismatchError(
            f"client_ids length {len(batch.client_ids)} != batch size {bs}"
        )
    if batch.sample_ids and len(batch.sample_ids) != bs:
        raise ShapeMismatchError(
            f"sample_ids length {len(batch.sample_ids)} != batch size {bs}"
        )
    if batch.modality_masks is not None and batch.modality_masks.size(0) != bs:
        raise ShapeMismatchError(
            f"modality_masks batch dim {batch.modality_masks.size(0)} != {bs}"
        )
    present_tensors = {
        mod: batch.get_modality_tensor(mod)
        for mod in MODALITY_KEYS
        if batch.has_modality(mod)
    }
    validate_batch_tensors(present_tensors, bs)
