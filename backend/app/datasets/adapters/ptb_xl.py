from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import logger
from app.datasets.base import BaseDatasetAdapter, DatasetLoadResult
from app.datasets.validators import validate_dataset

PTBXL_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
PTBXL_FILES = ["ptbxl_database.csv", "scp_statements.csv"]


class PTBXLAdapter(BaseDatasetAdapter):
    name = "ptb_xl"
    modalities = ["sensor"]
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    num_classes = 5
    input_shapes = {"sensor": (12, 5000)}

    def download(self, destination: Path, force: bool = False) -> Path:
        raw_dir = destination / "raw" / self.name
        if raw_dir.exists() and not force:
            logger.info(f"{self.name} already downloaded at {raw_dir}")
            return raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PTB-XL requires PhysioNet access. Generating sample data.")
        self.generate_sample(destination, num_samples=100)
        return raw_dir

    def _try_load_raw(self, path: Path, split: str) -> DatasetLoadResult | None:
        raw_dir = path / "raw" / self.name
        csv_path = raw_dir / "ptbxl_database.csv"
        records_dir = raw_dir / "records100"
        if not csv_path.exists() or not records_dir.exists():
            return None
        import pandas as pd

        df = pd.read_csv(csv_path)
        if split == "train":
            df = df[df["strat_fold"] < 9]
        elif split == "val":
            df = df[df["strat_fold"] == 9]
        else:
            df = df[df["strat_fold"] == 10]
        data_list = []
        labels = []
        for _, row in df.iterrows():
            ecg_path = records_dir / f"{row['filename_hr']}.npy"
            if ecg_path.exists():
                data_list.append({"sensor": np.load(ecg_path).astype(np.float32)})
                labels.append(int(row["diagnostic_class_idx"]))
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
        target = raw_dir if raw_dir.exists() else (path / "sample" / self.name)
        return validate_dataset(
            name=self.name,
            path=target,
            expected_files=PTBXL_FILES if target == raw_dir else None,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modalities": self.modalities,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "input_shapes": self.input_shapes,
            "num_samples": 21837,
            "leads": 12,
            "sampling_rate": 500,
            "duration_seconds": 10,
            "source": "PhysioNet PTB-XL",
        }

    def generate_sample(self, destination: Path, num_samples: int = 100) -> Path:
        sample_dir = destination / "sample" / self.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for split in ("train", "val", "test"):
            n = max(10, num_samples // 3)
            data_list = [
                {"sensor": rng.normal(0, 1, (12, 5000)).astype(np.float32)}
                for _ in range(n)
            ]
            labels = rng.integers(0, self.num_classes, n)
            np.save(sample_dir / f"{split}_data.npy", np.array(data_list, dtype=object))
            np.save(sample_dir / f"{split}_labels.npy", labels)
        logger.info(f"Sample {self.name} data generated at {sample_dir}")
        return sample_dir


ptbxl_adapter = PTBXLAdapter()
