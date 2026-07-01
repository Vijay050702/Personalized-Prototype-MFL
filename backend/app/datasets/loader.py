from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.datasets.base import DatasetLoadResult


class DatasetLoader:
    def load_from_memory(
        self,
        data: list[dict[str, np.ndarray]],
        labels: list[int] | np.ndarray,
        modalities: list[str],
        name: str = "custom",
    ) -> DatasetLoadResult:
        labels_arr = (
            np.array(labels, dtype=np.int64)
            if not isinstance(labels, np.ndarray)
            else labels
        )
        return DatasetLoadResult(
            data=data,
            labels=labels_arr,
            metadata={
                "name": name,
                "modalities": modalities,
                "num_samples": len(data),
                "num_classes": len(np.unique(labels_arr)),
            },
        )

    def load_from_files(
        self,
        path: Path,
        modalities: list[str],
        label_file: str = "labels.npy",
        **kwargs: Any,
    ) -> DatasetLoadResult:
        data_path = path / "data.npy"
        labels_path = path / label_file

        if not data_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Data files not found in {path}")

        data_arrays = np.load(data_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)

        if isinstance(data_arrays, np.ndarray) and data_arrays.ndim > 0:
            data_list = []
            for item in data_arrays:
                if isinstance(item, dict):
                    data_list.append(item)
                elif isinstance(item, np.ndarray):
                    data_list.append({modalities[0]: item})
                else:
                    data_list.append({modalities[0]: np.array(item)})
        else:
            data_list = [{modalities[0]: data_arrays}]

        return DatasetLoadResult(
            data=data_list,
            labels=np.array(labels, dtype=np.int64),
            metadata={
                "name": path.name,
                "modalities": modalities,
                "num_samples": len(data_list),
                "path": str(path),
            },
        )


dataset_loader = DatasetLoader()
