from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ClientProfile:
    client_id: str
    available_modalities: set[str] = field(default_factory=set)
    missing_modalities: set[str] = field(default_factory=set)
    training_steps: int = 0
    prototype_history: list[dict[str, Any]] = field(default_factory=list)
    confidence_history: list[dict[str, Any]] = field(default_factory=list)
    prototype_drift: float = 0.0
    drift_history: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_modalities(
        self,
        available: set[str],
        all_modalities: set[str],
    ) -> None:
        self.available_modalities = available.copy()
        self.missing_modalities = all_modalities - available

    def record_training_step(self) -> None:
        self.training_steps += 1

    def record_prototype(
        self,
        class_id: int,
        modality: str,
        confidence: float,
    ) -> None:
        self.prototype_history.append(
            {
                "class_id": class_id,
                "modality": modality,
                "confidence": confidence,
                "timestamp": time.time(),
            }
        )

    def record_confidence(
        self,
        modality: str,
        class_id: int,
        confidence: float,
    ) -> None:
        self.confidence_history.append(
            {
                "modality": modality,
                "class_id": class_id,
                "confidence": confidence,
                "timestamp": time.time(),
            }
        )

    def update_drift(self, drift: float) -> None:
        self.prototype_drift = drift
        self.drift_history.append(drift)

    @property
    def average_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        return sum(entry["confidence"] for entry in self.confidence_history) / len(
            self.confidence_history
        )

    @property
    def confidence_trend(self) -> float:
        if len(self.confidence_history) < 2:
            return 0.0
        recent = self.confidence_history[-5:]
        if len(recent) < 2:
            return 0.0
        first = recent[0]["confidence"]
        last = recent[-1]["confidence"]
        return last - first

    @property
    def average_drift(self) -> float:
        if not self.drift_history:
            return 0.0
        return sum(self.drift_history) / len(self.drift_history)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "available_modalities": sorted(self.available_modalities),
            "missing_modalities": sorted(self.missing_modalities),
            "training_steps": self.training_steps,
            "prototype_drift": self.prototype_drift,
            "average_confidence": self.average_confidence,
            "confidence_trend": self.confidence_trend,
            "average_drift": self.average_drift,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
