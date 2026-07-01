from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import logger
from app.datasets.base import BaseDatasetAdapter, DatasetLoadResult
from app.datasets.downloader import download_manager
from app.datasets.validators import validate_dataset

UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
UCI_HAR_FILES = [
    "train/X_train.txt",
    "train/y_train.txt",
    "train/subject_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
    "test/subject_test.txt",
    "activity_labels.txt",
    "features.txt",
]


class UCIHARAdapter(BaseDatasetAdapter):
    name = "uci_har"
    modalities = ["sensor"]
    classes = [
        "walking",
        "walking_upstairs",
        "walking_downstairs",
        "sitting",
        "standing",
        "laying",
    ]
    num_classes = 6
    input_shapes = {"sensor": (561,)}

    def download(self, destination: Path, force: bool = False) -> Path:
        raw_dir = destination / "raw" / self.name
        if raw_dir.exists() and not force:
            logger.info(f"{self.name} already downloaded at {raw_dir}")
            return raw_dir

        archive_path = raw_dir.parent / f"{self.name}.zip"
        dl_path = download_manager.download(UCI_HAR_URL, archive_path, force=force)
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_manager.extract(dl_path, raw_dir)

        uci_dir = raw_dir / "UCI HAR Dataset"
        if uci_dir.exists():
            for item in uci_dir.iterdir():
                dest = raw_dir / item.name
                if not dest.exists():
                    item.rename(dest)

        return raw_dir

    def _try_load_raw(self, path: Path, split: str) -> DatasetLoadResult | None:
        raw_dir = path / "raw" / self.name
        if split == "train":
            x_path = raw_dir / "train" / "X_train.txt"
            y_path = raw_dir / "train" / "y_train.txt"
        else:
            x_path = raw_dir / "test" / "X_test.txt"
            y_path = raw_dir / "test" / "y_test.txt"

        if not x_path.exists() or not y_path.exists():
            return None

        x = np.loadtxt(x_path)
        y = np.loadtxt(y_path, dtype=np.int64) - 1
        data = [{"sensor": x[i].astype(np.float32)} for i in range(len(x))]
        return DatasetLoadResult(
            data=data,
            labels=y,
            metadata={
                "name": self.name,
                "modalities": self.modalities,
                "num_samples": len(x),
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

        txt_x_path = sample_dir / split / f"X_{split}.txt"
        txt_y_path = sample_dir / split / f"y_{split}.txt"
        if not txt_x_path.exists():
            txt_x_path = (
                sample_dir
                / split
                / (f"X_train.txt" if split == "train" else "X_test.txt")
            )
            txt_y_path = (
                sample_dir
                / split
                / (f"y_train.txt" if split == "train" else "y_test.txt")
            )

        if txt_x_path.exists() and txt_y_path.exists():
            x = np.loadtxt(txt_x_path)
            y = np.loadtxt(txt_y_path, dtype=np.int64) - 1
            data = [{"sensor": x[i].astype(np.float32)} for i in range(len(x))]
            return DatasetLoadResult(
                data=data,
                labels=y,
                metadata={
                    "name": self.name,
                    "modalities": self.modalities,
                    "num_samples": len(x),
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
        raise FileNotFoundError(
            f"No {self.name} data found for split '{split}' in {path / 'raw' / self.name} or {path / 'sample' / self.name}"
        )

    def validate(self, path: Path) -> dict[str, Any]:
        raw_dir = path / "raw" / self.name
        sample_dir = path / "sample" / self.name
        if not raw_dir.exists() and not sample_dir.exists():
            return {
                "is_valid": False,
                "errors": [
                    f"Dataset '{self.name}' not found at {raw_dir} or {sample_dir}"
                ],
                "warnings": [],
                "info": {},
            }
        target_dir = raw_dir if raw_dir.exists() else sample_dir
        return validate_dataset(
            name=self.name,
            path=target_dir,
            expected_files=None if target_dir != raw_dir else UCI_HAR_FILES,
            modalities=self.modalities,
            expected_classes=self.num_classes,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modalities": self.modalities,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "input_shapes": self.input_shapes,
            "num_samples": 10299,
            "train_samples": 7352,
            "test_samples": 2947,
            "features": 561,
            "subjects": 30,
            "sampling_rate": 50,
            "source": "UCI Machine Learning Repository",
        }

    def generate_sample(self, destination: Path, num_samples: int = 100) -> Path:
        sample_dir = destination / "sample" / self.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)

        for split in ("train", "test"):
            n = num_samples // 2 if split == "train" else num_samples // 4
            x = rng.normal(0, 1, (n, 561)).astype(np.float32)
            y = rng.integers(0, self.num_classes, n)
            split_dir = sample_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                split_dir / "X_train.txt"
                if split == "train"
                else split_dir / "X_test.txt",
                x,
            )
            np.savetxt(
                split_dir / "y_train.txt"
                if split == "train"
                else split_dir / "y_test.txt",
                y + 1,
                fmt="%d",
            )

        with open(sample_dir / "activity_labels.txt", "w") as f:
            for i, cls in enumerate(self.classes, 1):
                f.write(f"{i} {cls}\n")
        with open(sample_dir / "features.txt", "w") as f:
            for i in range(1, 562):
                f.write(f"f{i}\n")
        logger.info(f"Sample {self.name} data generated at {sample_dir}")
        return sample_dir


uci_har_adapter = UCIHARAdapter()
