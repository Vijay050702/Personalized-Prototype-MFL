from __future__ import annotations

from typing import Any

import torch

from app.prototypes.prototype import Prototype
from app.prototypes.similarity import SimilarityEngine


class VisualizationSupport:
    def __init__(self, similarity_engine: SimilarityEngine | None = None):
        self._similarity = similarity_engine or SimilarityEngine(metric="cosine")

    def embedding_data(
        self, embeddings: torch.Tensor, labels: list[int] | None = None
    ) -> dict[str, Any]:
        data = {
            "embeddings": embeddings.detach().cpu().tolist(),
            "dim": embeddings.size(1),
            "count": embeddings.size(0),
        }
        if labels:
            data["labels"] = labels
        return data

    def prototype_trajectories(
        self,
        history: list[dict[str, Any]],
        embedding_dim: int,
    ) -> dict[str, Any]:
        trajectory: list[list[float]] = []
        timestamps: list[float] = []
        for h in history:
            if "embedding" in h:
                trajectory.append(h["embedding"])
                timestamps.append(h.get("timestamp", 0.0))
        return {
            "trajectory": trajectory,
            "timestamps": timestamps,
            "steps": len(trajectory),
            "dim": embedding_dim,
        }

    def similarity_heatmap(self, prototypes: list[Prototype]) -> dict[str, Any]:
        matrix = self._similarity.prototype_similarity_matrix(prototypes)
        return {
            "matrix": matrix.detach().cpu().tolist(),
            "size": len(prototypes),
            "prototype_ids": [p.prototype_id for p in prototypes],
            "class_ids": [p.class_id for p in prototypes],
        }

    def cluster_plot_data(
        self,
        embeddings: torch.Tensor,
        assignments: list[int],
        centroids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        data = {
            "embeddings": embeddings.detach().cpu().tolist(),
            "assignments": assignments,
            "n_clusters": len(set(assignments)) - (1 if -1 in assignments else 0),
        }
        if centroids is not None:
            data["centroids"] = centroids.detach().cpu().tolist()
        return data

    def pairwise_distances(self, prototypes: list[Prototype]) -> dict[str, Any]:
        matrix = self._similarity.prototype_distance_matrix(prototypes)
        return {
            "distance_matrix": matrix.detach().cpu().tolist(),
            "size": len(prototypes),
            "prototype_ids": [p.prototype_id for p in prototypes],
        }

    def prototype_summary(self, prototype: Prototype) -> dict[str, Any]:
        return prototype.to_dict()
