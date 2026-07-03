from __future__ import annotations

from typing import Any

import torch

from app.personalization.metrics import PersonalizationMetrics
from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.profile import ClientProfile
from app.training.logger import TrainingLogger


class Evaluator:
    def __init__(
        self,
        personalization_metrics: PersonalizationMetrics | None = None,
        logger_instance: TrainingLogger | None = None,
    ) -> None:
        self._personalization_metrics = (
            personalization_metrics or PersonalizationMetrics()
        )
        self._logger = logger_instance or TrainingLogger()
        self._eval_history: list[dict[str, Any]] = []

    def evaluate(
        self,
        loss: float = 0.0,
        accuracy: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
    ) -> dict[str, float]:
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def evaluate_classification(
        self,
        all_outputs: list[torch.Tensor],
        all_targets: list[torch.Tensor],
    ) -> dict[str, float]:
        if not all_outputs or not all_targets:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }
        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        preds = outputs.argmax(dim=1)

        accuracy = (preds == targets).float().mean().item()

        num_classes = outputs.size(1)
        precisions: list[float] = []
        recalls: list[float] = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fp = ((preds == c) & (targets != c)).sum().float()
            fn = ((preds != c) & (targets == c)).sum().float()
            prec = (tp / (tp + fp + 1e-8)).item()
            rec = (tp / (tp + fn + 1e-8)).item()
            precisions.append(prec)
            recalls.append(rec)

        precision = sum(precisions) / max(len(precisions), 1)
        recall = sum(recalls) / max(len(recalls), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def evaluate_prototypes(
        self,
        prototypes: list[Any],
    ) -> dict[str, float]:
        if not prototypes:
            return {
                "prototype_count": 0.0,
                "avg_confidence": 0.0,
            }
        confidences = [p.confidence for p in prototypes if hasattr(p, "confidence")]
        avg_conf = sum(confidences) / max(len(confidences), 1)
        return {
            "prototype_count": float(len(prototypes)),
            "avg_confidence": avg_conf,
        }

    def evaluate_personalization(
        self,
        personalized_prototypes: list[PersonalizedPrototype],
        profiles: dict[str, ClientProfile] | None = None,
    ) -> dict[str, float]:
        if not personalized_prototypes:
            return {
                "personalization_gain": 0.0,
                "prototype_drift": 0.0,
                "alignment_score": 0.0,
                "fusion_quality": 0.0,
                "client_diversity": 0.0,
                "confidence_trend": 0.0,
                "prototype_stability": 0.0,
            }
        return self._personalization_metrics.compute_all(
            personalized_prototypes,
            profiles=profiles,
        )

    def compute_communication_statistics(
        self,
        total_bytes_sent: int = 0,
        total_messages: int = 0,
        avg_latency: float = 0.0,
        compression_ratio: float = 1.0,
    ) -> dict[str, float]:
        return {
            "total_bytes_sent": float(total_bytes_sent),
            "total_messages": float(total_messages),
            "avg_latency": avg_latency,
            "compression_ratio": compression_ratio,
        }

    def evaluate_all(
        self,
        round_id: int,
        loss: float = 0.0,
        accuracy: float = 0.0,
        personalized_prototypes: list[PersonalizedPrototype] | None = None,
        profiles: dict[str, ClientProfile] | None = None,
        prototypes: list[Any] | None = None,
        comm_stats: dict[str, float] | None = None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = self.evaluate(loss=loss, accuracy=accuracy)

        if personalized_prototypes:
            p_metrics = self.evaluate_personalization(
                personalized_prototypes, profiles=profiles
            )
            metrics.update(p_metrics)

        if prototypes:
            proto_metrics = self.evaluate_prototypes(prototypes)
            metrics.update(proto_metrics)

        if comm_stats:
            metrics.update(comm_stats)

        self._eval_history.append({"round_id": round_id, "metrics": dict(metrics)})

        self._logger.log_evaluation(round_id=round_id, metrics=metrics)

        return metrics

    @property
    def eval_history(self) -> list[dict[str, Any]]:
        return list(self._eval_history)

    def get_round_metrics(self, round_id: int) -> dict[str, float] | None:
        for entry in self._eval_history:
            if entry["round_id"] == round_id:
                return entry["metrics"]
        return None

    def clear(self) -> None:
        self._eval_history.clear()
