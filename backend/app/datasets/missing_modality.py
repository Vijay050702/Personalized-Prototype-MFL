from __future__ import annotations

import copy
from typing import Any

import numpy as np

from app.datasets.base import DatasetLoadResult
from app.datasets.errors import InvalidModalityError


class MissingModalitySimulator:
    def apply_random(
        self,
        dataset: DatasetLoadResult,
        missing_ratio: float = 0.3,
        seed: int = 42,
    ) -> DatasetLoadResult:
        rng = np.random.default_rng(seed)
        result = copy.deepcopy(dataset)
        modalities = list(result.metadata.get("modalities", []))
        if not modalities:
            return result

        for i in range(len(result.data)):
            if rng.random() < missing_ratio:
                mod = modalities[rng.integers(len(modalities))]
                result.data[i][mod] = np.array([])
                result.data[i][f"{mod}_missing"] = True

        result.metadata["missing_modality_ratio"] = missing_ratio
        result.metadata["missing_modality_strategy"] = "random"
        return result

    def apply_modality_wise(
        self,
        dataset: DatasetLoadResult,
        modalities_to_drop: list[str] | None = None,
        missing_ratio: float = 0.5,
        seed: int = 42,
    ) -> DatasetLoadResult:
        rng = np.random.default_rng(seed)
        result = copy.deepcopy(dataset)
        available = list(result.metadata.get("modalities", []))
        if modalities_to_drop:
            for m in modalities_to_drop:
                if m not in available:
                    raise InvalidModalityError(
                        f"Modality '{m}' not in dataset: {available}"
                    )
            target_mods = modalities_to_drop
        else:
            target_mods = available

        for i in range(len(result.data)):
            if rng.random() < missing_ratio:
                mod = target_mods[rng.integers(len(target_mods))]
                result.data[i][mod] = np.array([])
                result.data[i][f"{mod}_missing"] = True

        result.metadata["missing_modality_ratio"] = missing_ratio
        result.metadata["missing_modality_strategy"] = "modality_wise"
        result.metadata["missing_modalities"] = target_mods
        return result

    def apply_client_wise(
        self,
        dataset: DatasetLoadResult,
        missing_map: dict[str, list[str]] | None = None,
        seed: int = 42,
    ) -> DatasetLoadResult:
        rng = np.random.default_rng(seed)
        result = copy.deepcopy(dataset)
        available = list(result.metadata.get("modalities", []))

        if missing_map is None:
            rng.shuffle(available)
            mid = len(available) // 2
            client_a_mods = available[:mid]
            client_b_mods = available[mid:]
            missing_map = {"client_a": client_a_mods, "client_b": client_b_mods}

        for i in range(len(result.data)):
            client_id = result.data[i].get("_client", "client_a")
            client_mods = missing_map.get(client_id, available)
            for mod in available:
                if mod not in client_mods:
                    result.data[i][mod] = np.array([])
                    result.data[i][f"{mod}_missing"] = True

        result.metadata["missing_modality_strategy"] = "client_wise"
        result.metadata["missing_modality_map"] = missing_map
        return result

    def apply(
        self,
        dataset: DatasetLoadResult,
        strategy: str = "random",
        missing_ratio: float = 0.3,
        modalities: list[str] | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> DatasetLoadResult:
        if strategy == "random":
            return self.apply_random(dataset, missing_ratio, seed)
        elif strategy == "modality_wise":
            return self.apply_modality_wise(dataset, modalities, missing_ratio, seed)
        elif strategy == "client_wise":
            return self.apply_client_wise(dataset, kwargs.get("missing_map"), seed)
        else:
            raise ValueError(f"Unknown missing modality strategy: {strategy}")
