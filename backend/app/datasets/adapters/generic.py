from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import logger
from app.datasets.base import BaseDatasetAdapter, DatasetLoadResult
from app.datasets.validators import validate_dataset


class GenericDatasetAdapter(BaseDatasetAdapter):
    name = "generic"
    modalities = ["image", "text", "audio", "sensor"]
    classes = ["class_0", "class_1"]
    num_classes = 2
    input_shapes = {
        "image": (3, 224, 224),
        "text": (128,),
        "audio": (128, 128),
        "sensor": (128, 6),
    }

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self._config = config or {}
        self._name = self._config.get("name", "generic")
        self._modalities = self._config.get("modalities", ["image"])
        self._classes = self._config.get("classes", ["class_0", "class_1"])
        self._num_classes = len(self._classes)
        self._input_shapes = {
            m: tuple(self._config.get("input_shapes", {}).get(m, (3, 224, 224)))
            for m in self._modalities
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def modalities(self) -> list[str]:
        return self._modalities

    @property
    def classes(self) -> list[str]:
        return self._classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        return self._input_shapes

    def download(self, destination: Path, force: bool = False) -> Path:
        raw_dir = destination / "raw" / self.name
        if raw_dir.exists() and not force:
            return raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Generic dataset '{self.name}': no remote download. Use load() with your own data."
        )
        return raw_dir

    def load(self, path: Path, split: str = "train") -> DatasetLoadResult:
        custom_path = (
            Path(self._config.get("path", "")) if self._config.get("path") else None
        )
        search_paths = [
            p
            for p in [
                path / "processed" / self.name,
                path / "raw" / self.name,
                path / "sample" / self.name,
                custom_path,
            ]
            if p is not None
        ]

        for sp in search_paths:
            npy_path = sp / f"{split}_data.npy"
            label_path = sp / f"{split}_labels.npy"
            if npy_path.exists() and label_path.exists():
                data = np.load(npy_path, allow_pickle=True)
                labels = np.load(label_path, allow_pickle=True)
                return DatasetLoadResult(
                    data=list(data),
                    labels=labels,
                    metadata={
                        "name": self.name,
                        "modalities": self.modalities,
                        "num_samples": len(data),
                        "num_classes": self.num_classes,
                        "split": split,
                    },
                )

        raise FileNotFoundError(
            f"No data found for generic dataset '{self.name}'. "
            f"Place {{split}}_data.npy and {{split}}_labels.npy in one of: {search_paths}"
        )

    def validate(self, path: Path) -> dict[str, Any]:
        raw_dir = path / "raw" / self.name
        return validate_dataset(
            name=self.name, path=raw_dir, modalities=self.modalities
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modalities": self.modalities,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "input_shapes": {k: list(v) for k, v in self.input_shapes.items()},
            "num_samples": 0,
            "source": "custom",
            "is_generic": True,
        }

    def generate_sample(self, destination: Path, num_samples: int = 100) -> Path:
        sample_dir = destination / "sample" / self.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)

        for split in ("train", "test"):
            n = max(10, num_samples // 2)
            data_list = []
            for _ in range(n):
                sample = {}
                for mod in self.modalities:
                    shape = self.input_shapes.get(mod, (128,))
                    sample[mod] = rng.normal(0, 1, shape).astype(np.float32)
                data_list.append(sample)
            labels = rng.integers(0, self.num_classes, n)

            np.save(sample_dir / f"{split}_data.npy", np.array(data_list, dtype=object))
            np.save(sample_dir / f"{split}_labels.npy", labels)

        logger.info(f"Sample generic dataset '{self.name}' generated at {sample_dir}")
        return sample_dir


generic_adapter = GenericDatasetAdapter()
