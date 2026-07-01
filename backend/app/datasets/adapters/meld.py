from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import logger
from app.datasets.base import BaseDatasetAdapter, DatasetLoadResult
from app.datasets.validators import validate_dataset

MELD_CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class MELDAdapter(BaseDatasetAdapter):
    name = "meld"
    modalities = ["text", "audio"]
    classes = MELD_CLASSES
    num_classes = 7
    input_shapes = {"text": (128,), "audio": (128, 128)}

    def download(self, destination: Path, force: bool = False) -> Path:
        raw_dir = destination / "raw" / self.name
        if raw_dir.exists() and not force:
            logger.info(f"{self.name} already downloaded at {raw_dir}")
            return raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"MELD requires Kaggle download. Generating sample data.")
        self.generate_sample(destination, num_samples=100)
        return raw_dir

    def _try_load_processed(self, path: Path, split: str) -> DatasetLoadResult | None:
        npy_path = path / "processed" / self.name / f"{split}_data.npy"
        label_path = path / "processed" / self.name / f"{split}_labels.npy"
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
        return None

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
        result = self._try_load_processed(path, split)
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
            "num_samples": 13000,
            "train_samples": 9989,
            "test_samples": 2610,
            "source": "MELD - Multimodal EmotionLines Dataset (Kaggle)",
            "requires_authentication": True,
        }

    def generate_sample(self, destination: Path, num_samples: int = 100) -> Path:
        sample_dir = destination / "sample" / self.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for split in ("train", "test", "dev"):
            n = max(10, num_samples // 3)
            data_list = [
                {
                    "text": np.zeros(128, dtype=np.int64),
                    "audio": rng.normal(0, 1, (128, 128)).astype(np.float32),
                }
                for _ in range(n)
            ]
            for d in data_list:
                d["text"][:10] = rng.integers(2, 1000, 10)
                rng.shuffle(d["text"][:10])
            labels = rng.integers(0, self.num_classes, n)
            np.save(sample_dir / f"{split}_data.npy", np.array(data_list, dtype=object))
            np.save(sample_dir / f"{split}_labels.npy", labels)
        logger.info(f"Sample {self.name} data generated at {sample_dir}")
        return sample_dir


meld_adapter = MELDAdapter()
