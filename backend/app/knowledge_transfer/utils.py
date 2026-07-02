from __future__ import annotations

import time
from typing import Any

import torch

from app.core.logging import logger


class TransferLogger:
    def __init__(self, name: str = "knowledge_transfer"):
        self._name = name
        self._events: list[dict[str, Any]] = []

    def log_translation(
        self,
        source_modality: str,
        target_modality: str,
        class_id: int,
        confidence: float,
        duration: float,
    ) -> None:
        entry = {
            "event": "translation",
            "source_modality": source_modality,
            "target_modality": target_modality,
            "class_id": class_id,
            "confidence": confidence,
            "duration": duration,
            "timestamp": time.time(),
        }
        self._events.append(entry)
        logger.debug(
            f"Translated {source_modality} -> {target_modality} "
            f"(class={class_id}, conf={confidence:.3f}, {duration:.4f}s)"
        )

    def log_loss(
        self,
        loss_name: str,
        value: float,
        step: int = 0,
    ) -> None:
        entry = {
            "event": "loss",
            "loss_name": loss_name,
            "value": value,
            "step": step,
            "timestamp": time.time(),
        }
        self._events.append(entry)
        logger.debug(f"Loss [{loss_name}]: {value:.6f} (step={step})")

    def log_confidence(self, modality: str, class_id: int, confidence: float) -> None:
        entry = {
            "event": "confidence",
            "modality": modality,
            "class_id": class_id,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        self._events.append(entry)

    def log_alignment(
        self,
        source_modality: str,
        target_modality: str,
        alignment_score: float,
    ) -> None:
        entry = {
            "event": "alignment",
            "source_modality": source_modality,
            "target_modality": target_modality,
            "alignment_score": alignment_score,
            "timestamp": time.time(),
        }
        self._events.append(entry)
        logger.info(
            f"Alignment {source_modality} <-> {target_modality}: {alignment_score:.4f}"
        )

    def get_history(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.get("event") == event_type]

    def summary(self) -> dict[str, Any]:
        translations = self.get_history("translation")
        losses = self.get_history("loss")
        confidences = self.get_history("confidence")
        alignments = self.get_history("alignment")
        return {
            "total_events": len(self._events),
            "translations": len(translations),
            "losses": len(losses),
            "confidences": len(confidences),
            "alignments": len(alignments),
        }

    def reset(self) -> None:
        self._events.clear()


def compute_embedding_norm(embedding: torch.Tensor) -> torch.Tensor:
    return embedding.norm(p=2, dim=-1, keepdim=True)


def l2_normalize(embedding: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = compute_embedding_norm(embedding)
    return embedding / (norm + eps)
