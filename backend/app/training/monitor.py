from __future__ import annotations

import time
from typing import Any

import torch


class ResourceMonitor:
    def __init__(self) -> None:
        self._measurements: list[dict[str, Any]] = []
        self._start_time: float = time.time()

    def record(
        self,
        round_id: int,
        training_time: float = 0.0,
        communication_time: float = 0.0,
        aggregation_time: float = 0.0,
        knowledge_transfer_time: float = 0.0,
        personalization_time: float = 0.0,
        evaluation_time: float = 0.0,
        prototype_count: int = 0,
        client_count: int = 0,
        payload_size_bytes: int = 0,
    ) -> None:
        measurement: dict[str, Any] = {
            "round_id": round_id,
            "timestamp": time.time(),
            "training_time": training_time,
            "communication_time": communication_time,
            "aggregation_time": aggregation_time,
            "knowledge_transfer_time": knowledge_transfer_time,
            "personalization_time": personalization_time,
            "evaluation_time": evaluation_time,
            "prototype_count": prototype_count,
            "client_count": client_count,
            "payload_size_bytes": payload_size_bytes,
        }

        gpu_memory = self._get_gpu_memory()
        if gpu_memory is not None:
            measurement["gpu_memory_allocated_mb"] = gpu_memory["allocated_mb"]
            measurement["gpu_memory_cached_mb"] = gpu_memory["cached_mb"]

        self._measurements.append(measurement)

    def _get_gpu_memory(self) -> dict[str, float] | None:
        if not torch.cuda.is_available():
            return None
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "cached_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        }

    @property
    def total_elapsed(self) -> float:
        return time.time() - self._start_time

    @property
    def round_count(self) -> int:
        return len(self._measurements)

    def average_prototype_count(self) -> float:
        if not self._measurements:
            return 0.0
        return sum(m["prototype_count"] for m in self._measurements) / len(
            self._measurements
        )

    def average_client_count(self) -> float:
        if not self._measurements:
            return 0.0
        return sum(m["client_count"] for m in self._measurements) / len(
            self._measurements
        )

    def total_training_time(self) -> float:
        return sum(m["training_time"] for m in self._measurements)

    def total_communication_time(self) -> float:
        return sum(m["communication_time"] for m in self._measurements)

    def total_aggregation_time(self) -> float:
        return sum(m["aggregation_time"] for m in self._measurements)

    def total_personalization_time(self) -> float:
        return sum(m["personalization_time"] for m in self._measurements)

    def total_evaluation_time(self) -> float:
        return sum(m["evaluation_time"] for m in self._measurements)

    def total_payload_bytes(self) -> int:
        return sum(m["payload_size_bytes"] for m in self._measurements)

    def average_payload_bytes(self) -> float:
        if not self._measurements:
            return 0.0
        return self.total_payload_bytes() / len(self._measurements)

    def statistics(self) -> dict[str, Any]:
        return {
            "total_elapsed": self.total_elapsed,
            "round_count": self.round_count,
            "average_prototype_count": self.average_prototype_count(),
            "average_client_count": self.average_client_count(),
            "total_training_time": self.total_training_time(),
            "total_communication_time": self.total_communication_time(),
            "total_aggregation_time": self.total_aggregation_time(),
            "total_personalization_time": self.total_personalization_time(),
            "total_evaluation_time": self.total_evaluation_time(),
            "total_payload_bytes": self.total_payload_bytes(),
            "average_payload_bytes": self.average_payload_bytes(),
        }

    def get_measurements(self, round_id: int) -> dict[str, Any] | None:
        for m in self._measurements:
            if m["round_id"] == round_id:
                return m
        return None

    def clear(self) -> None:
        self._measurements.clear()
        self._start_time = time.time()
