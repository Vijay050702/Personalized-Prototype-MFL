from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

import torch

from app.federated.models import (
    AggregatedPrototype,
    ClientPrototypePackage,
)


class AggregationStatistics:
    def __init__(self):
        self._package_counts: Counter[str] = Counter()
        self._class_counts: Counter[int] = Counter()
        self._modality_counts: Counter[str] = Counter()
        self._client_counts: Counter[str] = Counter()
        self._round_durations: list[float] = []
        self._participations: dict[int, int] = {}
        self._weight_distributions: dict[int, list[float]] = {}
        self._drift_history: list[dict[str, float]] = []
        self._latency_history: list[float] = []
        self._missing_modality_log: list[dict[str, Any]] = []
        self._start_time: float = time.time()

    def record_package(self, package: ClientPrototypePackage) -> None:
        self._package_counts[package.modality] += 1
        self._class_counts[package.class_id] += 1
        self._modality_counts[package.modality] += 1
        self._client_counts[package.client_id] += 1

    def record_packages(self, packages: list[ClientPrototypePackage]) -> None:
        for pkg in packages:
            self.record_package(pkg)

    def record_round_completion(
        self,
        round_id: int,
        duration: float,
        num_participants: int,
    ) -> None:
        self._round_durations.append(duration)
        self._participations[round_id] = num_participants
        self._latency_history.append(duration)

    def record_weight_distribution(self, round_id: int, weights: list[float]) -> None:
        self._weight_distributions[round_id] = weights

    def record_drift(
        self,
        old_prototypes: list[AggregatedPrototype],
        new_prototypes: list[AggregatedPrototype],
    ) -> dict[str, float]:
        old_map: dict[str, AggregatedPrototype] = {}
        for p in old_prototypes:
            old_map[f"{p.modality}_c{p.class_id}"] = p
        new_map: dict[str, AggregatedPrototype] = {}
        for p in new_prototypes:
            new_map[f"{p.modality}_c{p.class_id}"] = p

        drifts: dict[str, float] = {}
        for key, new_p in new_map.items():
            if key in old_map:
                old_p = old_map[key]
                old_t = old_p.to_tensor()
                new_t = new_p.to_tensor()
                drift = float(
                    torch.nn.functional.pairwise_distance(
                        old_t.unsqueeze(0), new_t.unsqueeze(0)
                    ).squeeze()
                )
                drifts[key] = drift

        self._drift_history.append(drifts)
        return drifts

    def record_missing_modalities(
        self,
        client_id: str,
        expected: set[str],
        actual: set[str],
    ) -> None:
        missing = sorted(expected - actual)
        if missing:
            self._missing_modality_log.append(
                {
                    "client_id": client_id,
                    "missing_modalities": missing,
                    "timestamp": time.time(),
                }
            )

    @property
    def total_packages_received(self) -> int:
        return sum(self._package_counts.values())

    @property
    def unique_clients(self) -> int:
        return len(self._client_counts)

    @property
    def unique_classes(self) -> list[int]:
        return sorted(self._class_counts.keys())

    @property
    def unique_modalities(self) -> list[str]:
        return sorted(self._modality_counts.keys())

    @property
    def avg_round_duration(self) -> float:
        if not self._round_durations:
            return 0.0
        return sum(self._round_durations) / len(self._round_durations)

    @property
    def total_rounds(self) -> int:
        return len(self._round_durations)

    def average_drift(self) -> float:
        if not self._drift_history:
            return 0.0
        all_drifts = []
        for entry in self._drift_history:
            all_drifts.extend(entry.values())
        if not all_drifts:
            return 0.0
        return sum(all_drifts) / len(all_drifts)

    def weight_statistics(self, round_id: int) -> dict[str, float] | None:
        weights = self._weight_distributions.get(round_id)
        if not weights:
            return None
        t = torch.tensor(weights)
        return {
            "mean": float(t.mean()),
            "std": float(t.std()) if len(weights) > 1 else 0.0,
            "min": float(t.min()),
            "max": float(t.max()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_packages_received": self.total_packages_received,
            "unique_clients": self.unique_clients,
            "unique_classes": self.unique_classes,
            "unique_modalities": self.unique_modalities,
            "total_rounds": self.total_rounds,
            "avg_round_duration": self.avg_round_duration,
            "average_drift": self.average_drift(),
            "uptime_seconds": time.time() - self._start_time,
            "missing_modality_events": len(self._missing_modality_log),
        }

    def reset(self) -> None:
        self._package_counts.clear()
        self._class_counts.clear()
        self._modality_counts.clear()
        self._client_counts.clear()
        self._round_durations.clear()
        self._participations.clear()
        self._weight_distributions.clear()
        self._drift_history.clear()
        self._latency_history.clear()
        self._missing_modality_log.clear()
        self._start_time = time.time()
