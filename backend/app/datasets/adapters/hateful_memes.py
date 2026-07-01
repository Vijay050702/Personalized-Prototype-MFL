from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import logger
from app.datasets.base import BaseDatasetAdapter, DatasetLoadResult
from app.datasets.validators import validate_dataset


class HatefulMemesAdapter(BaseDatasetAdapter):
    name = "hateful_memes"
    modalities = ["image", "text"]
    classes = ["not_hateful", "hateful"]
    num_classes = 2
    input_shapes = {"image": (3, 224, 224), "text": (128,)}

    def download(self, destination: Path, force: bool = False) -> Path:
        raw_dir = destination / "raw" / self.name
        if raw_dir.exists() and not force:
            logger.info(f"{self.name} already downloaded at {raw_dir}")
            return raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.generate_sample(destination, num_samples=100)
        return raw_dir

    def _try_load_raw(self, path: Path, split: str) -> DatasetLoadResult | None:
        json_path = path / "raw" / self.name / f"{split}.json"
        img_dir = path / "raw" / self.name / "img"
        if not json_path.exists():
            return None
        import json

        with open(json_path) as f:
            entries = json.load(f)
        data_list = []
        labels = []
        for entry in entries:
            text_arr = np.zeros(128, dtype=np.int64)
            tokens = entry.get("text", "").split()[:128]
            for j, t in enumerate(tokens):
                text_arr[j] = hash(t) % 10000 + 2
            data_list.append(
                {"text": text_arr, "image": np.array([], dtype=np.float32)}
            )
            labels.append(1 if entry.get("label") == "hateful" else 0)
        if not data_list:
            return None
        return DatasetLoadResult(
            data=data_list,
            labels=np.array(labels, dtype=np.int64),
            metadata={
                "name": self.name,
                "modalities": self.modalities,
                "num_samples": len(data_list),
                "num_classes": self.num_classes,
                "split": split,
            },
        )

    def _try_load_sample(self, path: Path, split: str) -> DatasetLoadResult | None:
        sample_dir = path / "sample" / self.name
        npy_path = sample_dir / f"{split}_data.npy"
        label_path = sample_dir / f"{split}_labels.npy"
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
                    "is_sample": True,
                },
            )
        return None

    def load(self, path: Path, split: str = "train") -> DatasetLoadResult:
        result = self._try_load_raw(path, split)
        if result is not None:
            return result
        result = self._try_load_sample(path, split)
        if result is not None:
            return result
        raise FileNotFoundError(f"No {self.name} data found for split '{split}'.")

    def validate(self, path: Path) -> dict[str, Any]:
        raw_dir = path / "raw" / self.name
        sample_dir = path / "sample" / self.name
        if not raw_dir.exists() and not sample_dir.exists():
            return {
                "is_valid": False,
                "errors": [f"Dataset '{self.name}' not found"],
                "warnings": [],
                "info": {},
            }
        target = raw_dir if raw_dir.exists() else sample_dir
        return validate_dataset(name=self.name, path=target, modalities=self.modalities)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modalities": self.modalities,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "input_shapes": self.input_shapes,
            "num_samples": 12000,
            "train_samples": 8500,
            "dev_samples": 500,
            "test_samples": 3000,
            "source": "Facebook AI Hateful Memes Challenge",
        }

    def generate_sample(self, destination: Path, num_samples: int = 100) -> Path:
        sample_dir = destination / "sample" / self.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for split in ("train", "dev", "test"):
            n = max(10, num_samples // 3)
            data_list = []
            for _ in range(n):
                text_arr = np.zeros(128, dtype=np.int64)
                n_tokens = rng.integers(5, 20)
                tokens = rng.integers(2, 5000, n_tokens)
                text_arr[:n_tokens] = tokens
                data_list.append(
                    {
                        "text": text_arr,
                        "image": rng.normal(0, 1, (3, 224, 224)).astype(np.float32),
                    }
                )
            labels = rng.integers(0, 2, n)
            np.save(sample_dir / f"{split}_data.npy", np.array(data_list, dtype=object))
            np.save(sample_dir / f"{split}_labels.npy", labels)
        logger.info(f"Sample {self.name} data generated at {sample_dir}")
        return sample_dir


hateful_memes_adapter = HatefulMemesAdapter()
