from __future__ import annotations

from typing import Any

import torch

from app.prototypes.prototype import Prototype


class SimilarityEngine:
    def __init__(self, metric: str = "cosine"):
        self._metric = metric

    @property
    def metric(self) -> str:
        return self._metric

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._metric == "cosine":
            return torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).squeeze(0)
        elif self._metric == "euclidean":
            return 1.0 / (
                1.0
                + torch.nn.functional.pairwise_distance(a.unsqueeze(0), b.unsqueeze(0))
            ).squeeze(0)
        elif self._metric == "manhattan":
            dist = torch.abs(a - b).sum()
            return 1.0 / (1.0 + dist)
        elif self._metric == "dot":
            return (a * b).sum()
        else:
            raise ValueError(f"Unknown similarity metric: {self._metric}")

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._metric == "cosine":
            return 1.0 - torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).squeeze(0)
        elif self._metric == "euclidean":
            return torch.nn.functional.pairwise_distance(
                a.unsqueeze(0), b.unsqueeze(0)
            ).squeeze(0)
        elif self._metric == "manhattan":
            return torch.abs(a - b).sum()
        elif self._metric == "dot":
            return 1.0 / (1.0 + (a * b).sum())
        else:
            raise ValueError(f"Unknown similarity metric: {self._metric}")

    def batch_similarity(
        self,
        embedding: torch.Tensor,
        prototypes: list[Prototype],
    ) -> torch.Tensor:
        if not prototypes:
            return torch.tensor([])
        proto_embs = torch.stack([p.embedding for p in prototypes], dim=0)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        if self._metric == "cosine":
            sims = torch.nn.functional.cosine_similarity(embedding, proto_embs, dim=1)
        elif self._metric == "euclidean":
            dists = torch.cdist(embedding, proto_embs, p=2.0).squeeze(0)
            sims = 1.0 / (1.0 + dists)
        elif self._metric == "dot":
            sims = (embedding * proto_embs).sum(dim=1)
        else:
            raise ValueError(f"Unknown similarity metric: {self._metric}")
        return sims

    def pairwise_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._metric == "cosine":
            norms = embeddings.norm(p=2, dim=1, keepdim=True)
            normalized = embeddings / (norms + 1e-8)
            return normalized @ normalized.T
        elif self._metric == "euclidean":
            dists = torch.cdist(embeddings, embeddings, p=2.0)
            return 1.0 / (1.0 + dists)
        elif self._metric == "dot":
            return embeddings @ embeddings.T
        else:
            raise ValueError(f"Unknown similarity metric: {self._metric}")

    def prototype_similarity_matrix(self, prototypes: list[Prototype]) -> torch.Tensor:
        embeddings = torch.stack([p.embedding for p in prototypes], dim=0)
        return self.pairwise_similarity_matrix(embeddings)

    def prototype_distance_matrix(self, prototypes: list[Prototype]) -> torch.Tensor:
        embeddings = torch.stack([p.embedding for p in prototypes], dim=0)
        dists = torch.cdist(embeddings, embeddings, p=2.0)
        return dists
