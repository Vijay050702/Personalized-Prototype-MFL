from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.datasets.cache import DatasetCache


class MetadataGenerator:
    def __init__(self, cache: DatasetCache | None = None):
        self.cache = cache or DatasetCache()

    def generate(
        self,
        name: str,
        classes: list[str],
        modalities: list[str],
        input_shapes: dict[str, tuple[int, ...]],
        num_samples: int,
        client_count: int = 0,
        missing_modality_ratio: float = 0.0,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "dataset_name": name,
            "classes": classes,
            "num_classes": len(classes),
            "modalities": modalities,
            "input_shapes": {k: list(v) for k, v in input_shapes.items()},
            "num_samples": num_samples,
            "client_count": client_count,
            "missing_modality_ratio": missing_modality_ratio,
            "download_status": "not_downloaded",
            "preprocessing_status": "not_preprocessed",
            "partition_status": "not_partitioned",
            **extra,
        }

    def save(self, metadata: dict[str, Any], path: Path | None = None) -> Path:
        if path is None:
            path = Path(settings.datasets_root) / "processed" / metadata["dataset_name"]
        path.mkdir(parents=True, exist_ok=True)
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.cache.set_json(f"metadata_{metadata['dataset_name']}", metadata)
        return metadata_path

    def load(self, name: str, path: Path | None = None) -> dict[str, Any] | None:
        cached = self.cache.get_json(f"metadata_{name}")
        if cached:
            return cached
        if path is None:
            path = Path(settings.datasets_root) / "processed" / name / "metadata.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None


metadata_generator = MetadataGenerator()
