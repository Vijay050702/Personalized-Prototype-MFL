from __future__ import annotations

from typing import Any

import torch


def validate_dimensions(
    tensor: torch.Tensor,
    expected_dim: int,
    name: str = "tensor",
) -> None:
    if tensor.dim() == 0:
        raise ValueError(f"{name} is a scalar, expected a 1-D tensor")
    if tensor.size(-1) != expected_dim:
        raise ValueError(
            f"{name} last dimension {tensor.size(-1)} does not match "
            f"expected dimension {expected_dim}"
        )


def validate_weights_sum_to_one(
    weights: dict[str, float],
    sources: list[str],
    name: str = "fusion_weights",
) -> None:
    for src in sources:
        if src not in weights:
            raise ValueError(f"Missing weight for source '{src}' in {name}")
    total = sum(weights.get(src, 0.0) for src in sources)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} sum to {total:.6f}, expected 1.0")


def validate_missing_modalities(
    available: set[str],
    missing: set[str],
    all_modalities: set[str],
) -> None:
    for mod in available:
        if mod not in all_modalities:
            raise ValueError(
                f"Available modality '{mod}' is not in the known set: {all_modalities}"
            )
    for mod in missing:
        if mod not in all_modalities:
            raise ValueError(
                f"Missing modality '{mod}' is not in the known set: {all_modalities}"
            )
    overlap = available & missing
    if overlap:
        raise ValueError(f"Modalities cannot be both available and missing: {overlap}")
    if not missing:
        raise ValueError("No missing modalities to validate")


def validate_confidence_range(
    confidence: float,
    name: str = "confidence",
) -> None:
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {confidence}")


def validate_duplicate_prototypes(
    prototypes: list[Any],
    prototype_key: str = "prototype",
) -> None:
    seen: set[tuple[str, int, str]] = set()
    for p in prototypes:
        client_id = getattr(p, "client_id", None)
        class_id = getattr(p, "class_id", None)
        modality = getattr(p, "modality", None)
        key = (str(client_id), int(class_id), str(modality))
        if key in seen:
            raise ValueError(
                f"Duplicate {prototype_key} for client={client_id}, "
                f"class={class_id}, modality={modality}"
            )
        seen.add(key)


def validate_shape_match(
    a: torch.Tensor,
    b: torch.Tensor,
    name_a: str = "a",
    name_b: str = "b",
) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {name_a}.shape={tuple(a.shape)} != "
            f"{name_b}.shape={tuple(b.shape)}"
        )


def validate_fusion_sources(
    sources: list[str],
) -> None:
    valid = {"local", "global", "cross_modal"}
    for src in sources:
        if src not in valid:
            raise ValueError(f"Unknown fusion source '{src}'. Valid: {valid}")
