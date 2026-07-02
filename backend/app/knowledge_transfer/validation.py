from __future__ import annotations

import torch


def validate_mapping_dimensions(
    source_dim: int,
    target_dim: int,
    source_name: str = "source",
    target_name: str = "target",
) -> None:
    if source_dim < 1:
        raise ValueError(f"{source_name} dimension must be >= 1, got {source_dim}")
    if target_dim < 1:
        raise ValueError(f"{target_name} dimension must be >= 1, got {target_dim}")


def validate_prototype_size(tensor: torch.Tensor, name: str = "tensor") -> None:
    if tensor.dim() != 1 and tensor.dim() != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {tensor.dim()}-D")
    if tensor.size(-1) == 0:
        raise ValueError(f"{name} has empty last dimension")


def validate_missing_modalities(
    available: set[str],
    missing: set[str],
    all_modalities: set[str],
) -> None:
    for m in available:
        if m not in all_modalities:
            raise ValueError(f"Available modality '{m}' not in known modalities")
    for m in missing:
        if m not in all_modalities:
            raise ValueError(f"Missing modality '{m}' not in known modalities")
    overlap = available & missing
    if overlap:
        raise ValueError(f"Modalities cannot be both available and missing: {overlap}")
    if not missing:
        raise ValueError("No missing modalities to synthesize")


def validate_no_nan(tensor: torch.Tensor, name: str = "tensor") -> None:
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")


def validate_shape_match(
    a: torch.Tensor,
    b: torch.Tensor,
    a_name: str = "a",
    b_name: str = "b",
) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a_name} {a.shape} vs {b_name} {b.shape}")
