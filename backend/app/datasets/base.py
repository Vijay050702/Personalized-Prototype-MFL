from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class DatasetLoadResult:
    def __init__(
        self,
        data: list[dict[str, np.ndarray]],
        labels: np.ndarray,
        metadata: dict[str, Any],
        client_id: str | None = None,
    ):
        self.data = data
        self.labels = labels
        self.metadata = metadata
        self.client_id = client_id

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[dict[str, np.ndarray], int]:
        return self.data[idx], int(self.labels[idx])


class BaseDatasetAdapter(ABC):
    name: str = ""
    modalities: list[str] = []
    classes: list[str] = []
    num_classes: int = 0
    input_shapes: dict[str, tuple[int, ...]] = {}

    @abstractmethod
    def download(self, destination: Path, force: bool = False) -> Path:
        pass

    @abstractmethod
    def load(self, path: Path, split: str = "train") -> DatasetLoadResult:
        pass

    @abstractmethod
    def validate(self, path: Path) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate_sample(self, destination: Path, num_samples: int = 100) -> Path:
        pass


class BasePartitioner(ABC):
    @abstractmethod
    def partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> dict[int, list[int]]:
        pass


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, samples: list[dict[str, np.ndarray]]) -> None:
        pass

    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        pass
