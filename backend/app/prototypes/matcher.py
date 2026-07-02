from __future__ import annotations

from typing import Any

import torch

from app.prototypes.prototype import Prototype
from app.prototypes.repository import PrototypeRepository
from app.prototypes.similarity import SimilarityEngine


class PrototypeMatcher:
    def __init__(
        self,
        repository: PrototypeRepository,
        similarity_engine: SimilarityEngine | None = None,
        metric: str = "cosine",
    ):
        self._repository = repository
        self._similarity = similarity_engine or SimilarityEngine(metric=metric)
        self._metric = metric

    def match(
        self,
        embedding: torch.Tensor,
        top_k: int = 1,
        class_filter: list[int] | None = None,
        modality_filter: str | None = None,
    ) -> list[tuple[Prototype, float]]:
        candidates = self._repository.list()
        if class_filter is not None:
            candidates = [p for p in candidates if p.class_id in class_filter]
        if modality_filter is not None:
            candidates = [p for p in candidates if p.modality == modality_filter]

        if not candidates:
            return []

        similarities = self._similarity.batch_similarity(embedding, candidates)
        scored = list(zip(candidates, similarities.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def batch_match(
        self,
        embeddings: torch.Tensor,
        top_k: int = 1,
        class_filter: list[int] | None = None,
        modality_filter: str | None = None,
    ) -> list[list[tuple[Prototype, float]]]:
        results = []
        for i in range(embeddings.size(0)):
            result = self.match(
                embeddings[i],
                top_k=top_k,
                class_filter=class_filter,
                modality_filter=modality_filter,
            )
            results.append(result)
        return results

    def nearest_prototype(
        self,
        embedding: torch.Tensor,
        class_filter: list[int] | None = None,
    ) -> tuple[Prototype, float] | None:
        matches = self.match(embedding, top_k=1, class_filter=class_filter)
        return matches[0] if matches else None

    def rank(
        self,
        embedding: torch.Tensor,
        class_filter: list[int] | None = None,
    ) -> list[tuple[Prototype, float]]:
        return self.match(
            embedding, top_k=len(self._repository.list()), class_filter=class_filter
        )

    def match_to_class(
        self,
        embedding: torch.Tensor,
        top_k: int = 1,
    ) -> list[tuple[int, float]]:
        matches = self.match(embedding, top_k=top_k)
        class_scores: dict[int, float] = {}
        for proto, score in matches:
            cid = proto.class_id
            if cid not in class_scores or score > class_scores[cid]:
                class_scores[cid] = score
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_classes[:top_k]

    @property
    def repository(self) -> PrototypeRepository:
        return self._repository

    @property
    def metric(self) -> str:
        return self._metric
