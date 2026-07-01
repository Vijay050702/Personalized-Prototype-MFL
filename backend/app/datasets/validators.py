from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.datasets.errors import DatasetValidationError


def validate_dataset_exists(path: Path) -> None:
    if not path.exists():
        raise DatasetValidationError(f"Dataset path does not exist: {path}")


def validate_dataset_files(path: Path, expected_files: list[str]) -> list[str]:
    errors = []
    for f in expected_files:
        if not (path / f).exists():
            errors.append(f"Missing expected file: {f}")
    return errors


def validate_labels(
    labels: list[int], expected_classes: int | None = None
) -> list[str]:
    errors = []
    if not labels:
        errors.append("Label list is empty")
        return errors
    unique = set(labels)
    if expected_classes and len(unique) != expected_classes:
        errors.append(
            f"Expected {expected_classes} classes, found {len(unique)}: {sorted(unique)}"
        )
    return errors


def validate_class_balance(
    labels: list[int], imbalance_threshold: float = 0.1
) -> list[str]:
    warnings = []
    from collections import Counter

    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return warnings
    for cls, count in counts.most_common():
        ratio = count / total
        if ratio < imbalance_threshold:
            warnings.append(
                f"Class {cls} is underrepresented: {ratio:.2%} (threshold: {imbalance_threshold:.0%})"
            )
    return warnings


def validate_modalities(
    modalities: list[str], supported: set[str] | None = None
) -> list[str]:
    errors = []
    if supported is None:
        supported = {"image", "text", "audio", "sensor", "tabular", "multimodal"}
    for m in modalities:
        if m not in supported:
            errors.append(
                f"Unsupported modality: '{m}'. Supported: {sorted(supported)}"
            )
    return errors


def validate_dataset(
    name: str,
    path: Path,
    expected_files: list[str] | None = None,
    modalities: list[str] | None = None,
    expected_classes: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    try:
        validate_dataset_exists(path)
    except DatasetValidationError as e:
        errors.append(str(e))
        return {"is_valid": False, "errors": errors, "warnings": warnings, "info": info}

    if expected_files:
        errors.extend(validate_dataset_files(path, expected_files))

    if modalities:
        errors.extend(validate_modalities(modalities))

    info["path"] = str(path)
    info["name"] = name
    info["size_mb"] = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (
        1024 * 1024
    )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }
