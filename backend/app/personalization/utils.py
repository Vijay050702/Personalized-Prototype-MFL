from __future__ import annotations

import time
from typing import Any

import torch

from app.core.logging import logger


class PersonalizationLogger:
    def __init__(self, name: str = "personalization"):
        self._name = name
        self._events: list[dict[str, Any]] = []

    def log_fusion(
        self,
        client_id: str,
        class_id: int,
        modality: str,
        fusion_strategy: str,
        weights: dict[str, float],
        confidence: float,
    ) -> None:
        self._events.append(
            {
                "type": "fusion",
                "client_id": client_id,
                "class_id": class_id,
                "modality": modality,
                "fusion_strategy": fusion_strategy,
                "weights": weights,
                "confidence": confidence,
                "timestamp": time.time(),
            }
        )
        logger.debug(
            f"Fused {modality}/c{class_id} for {client_id} "
            f"[{fusion_strategy}] weights={weights} conf={confidence:.3f}"
        )

    def log_selection(
        self,
        client_id: str,
        class_id: int,
        modality: str,
        source: str,
        confidence: float,
    ) -> None:
        self._events.append(
            {
                "type": "selection",
                "client_id": client_id,
                "class_id": class_id,
                "modality": modality,
                "source": source,
                "confidence": confidence,
                "timestamp": time.time(),
            }
        )
        logger.debug(f"Selected {source} for {modality}/c{class_id} ({client_id})")

    def log_adaptation(
        self,
        client_id: str,
        strategy: str,
        drift: float,
        step: int,
    ) -> None:
        self._events.append(
            {
                "type": "adaptation",
                "client_id": client_id,
                "strategy": strategy,
                "drift": drift,
                "step": step,
                "timestamp": time.time(),
            }
        )
        logger.debug(f"Adaptation [{strategy}] for {client_id}: drift={drift:.4f}")

    def log_loss(
        self,
        loss_name: str,
        value: float,
        step: int = 0,
    ) -> None:
        self._events.append(
            {
                "type": "loss",
                "loss_name": loss_name,
                "value": value,
                "step": step,
                "timestamp": time.time(),
            }
        )
        logger.debug(f"Loss [{loss_name}] = {value:.6f} (step {step})")

    def log_confidence(
        self,
        client_id: str,
        modality: str,
        class_id: int,
        confidence: float,
    ) -> None:
        self._events.append(
            {
                "type": "confidence",
                "client_id": client_id,
                "modality": modality,
                "class_id": class_id,
                "confidence": confidence,
                "timestamp": time.time(),
            }
        )

    def get_history(
        self,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if event_type is not None:
            return [e for e in self._events if e["type"] == event_type]
        return self._events.copy()

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for e in self._events:
            t = e["type"]
            counts[t] = counts.get(t, 0) + 1
        return {
            "total_events": len(self._events),
            **counts,
        }

    def reset(self) -> None:
        self._events.clear()


def compute_prototype_drift(
    current: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    return torch.norm(current - previous, p=2)


def l2_normalize(
    tensor: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    norm = tensor.norm(p=2, dim=-1, keepdim=True)
    return tensor / (norm + eps)
