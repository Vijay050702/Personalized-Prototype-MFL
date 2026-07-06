from __future__ import annotations

from typing import Any

import numpy as np
import torch


class VisualizationDataGenerator:
    @staticmethod
    def training_loss(
        round_metrics: list[dict[str, Any]],
    ) -> list[dict[str, float]]:
        return [
            {
                "round": int(m.get("round_id", i)),
                "loss": float(m.get("loss", 0.0)),
            }
            for i, m in enumerate(round_metrics)
        ]

    @staticmethod
    def accuracy(
        round_metrics: list[dict[str, Any]],
    ) -> list[dict[str, float]]:
        return [
            {
                "round": int(m.get("round_id", i)),
                "accuracy": float(m.get("accuracy", 0.0)),
            }
            for i, m in enumerate(round_metrics)
        ]

    @staticmethod
    def communication_rounds(
        round_metrics: list[dict[str, Any]],
    ) -> list[dict[str, float]]:
        return [
            {
                "round": int(m.get("round_id", i)),
                "communication_cost": float(m.get("communication_cost", 0.0)),
                "bytes_transferred": float(m.get("bytes_transferred", 0.0)),
            }
            for i, m in enumerate(round_metrics)
        ]

    @staticmethod
    def prototype_drift(
        history: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        return [
            {
                "round": i,
                "drift": float(entry.get("drift", 0.0)),
            }
            for i, entry in enumerate(history)
        ]

    @staticmethod
    def prototype_evolution(
        prototype_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        evolved: list[dict[str, Any]] = []
        for entry in prototype_history:
            point: dict[str, Any] = {
                "round": entry.get("round", 0),
                "class_id": entry.get("class_id", 0),
                "modality": entry.get("modality", "unknown"),
                "confidence": float(entry.get("confidence", 0.0)),
            }
            embedding = entry.get("embedding")
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            if isinstance(embedding, np.ndarray):
                if embedding.ndim == 1 and embedding.size >= 2:
                    point["x"] = float(embedding[0])
                    point["y"] = float(embedding[1])
                point["embedding_dim"] = embedding.size
            elif isinstance(embedding, list):
                if len(embedding) >= 2:
                    point["x"] = float(embedding[0])
                    point["y"] = float(embedding[1])
                point["embedding_dim"] = len(embedding)
            evolved.append(point)
        return evolved

    @staticmethod
    def client_participation(
        client_histories: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        for client_id, history in client_histories.items():
            for entry in history:
                data.append(
                    {
                        "client_id": client_id,
                        "round": entry.get("round", 0),
                        "active": 1,
                    }
                )
        return data

    @staticmethod
    def knowledge_transfer_quality(
        transfer_metrics: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        return [
            {
                "round": int(m.get("round_id", i)),
                "alignment_score": float(m.get("alignment_score", 0.0)),
                "transfer_accuracy": float(m.get("transfer_accuracy", 0.0)),
                "cross_modal_similarity": float(m.get("cross_modal_similarity", 0.0)),
            }
            for i, m in enumerate(transfer_metrics)
        ]

    @staticmethod
    def personalization_gain(
        personalization_metrics: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        return [
            {
                "round": int(m.get("round_id", i)),
                "personalization_gain": float(m.get("personalization_gain", 0.0)),
                "client_adaptation_score": float(m.get("client_adaptation_score", 0.0)),
                "prototype_fusion_quality": float(
                    m.get("prototype_fusion_quality", 0.0)
                ),
            }
            for i, m in enumerate(personalization_metrics)
        ]

    @staticmethod
    def all_metrics(
        round_metrics: list[dict[str, Any]],
        prototype_history: list[dict[str, float]] | None = None,
        transfer_metrics: list[dict[str, float]] | None = None,
        personalization_metrics: list[dict[str, float]] | None = None,
        client_histories: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "training_loss": VisualizationDataGenerator.training_loss(round_metrics),
            "accuracy": VisualizationDataGenerator.accuracy(round_metrics),
            "communication_rounds": VisualizationDataGenerator.communication_rounds(
                round_metrics
            ),
        }
        if prototype_history:
            data["prototype_drift"] = VisualizationDataGenerator.prototype_drift(
                prototype_history
            )
        if transfer_metrics:
            data["knowledge_transfer_quality"] = (
                VisualizationDataGenerator.knowledge_transfer_quality(transfer_metrics)
            )
        if personalization_metrics:
            data["personalization_gain"] = (
                VisualizationDataGenerator.personalization_gain(personalization_metrics)
            )
        if client_histories:
            data["client_participation"] = (
                VisualizationDataGenerator.client_participation(client_histories)
            )
        return data
