from __future__ import annotations

from typing import Any

import torch.nn as nn

from app.core.logging import logger
from app.prototypes.clustering import PrototypeClustering
from app.prototypes.confidence import ConfidenceEstimator
from app.prototypes.generator import PrototypeGenerator
from app.prototypes.matcher import PrototypeMatcher
from app.prototypes.memory import PrototypeMemory
from app.prototypes.metrics import PrototypeMetrics
from app.prototypes.prototype import Prototype
from app.prototypes.repository import PrototypeRepository
from app.prototypes.similarity import SimilarityEngine
from app.prototypes.updater import PrototypeUpdater
from app.prototypes.losses import (
    CenterLoss,
    PrototypeCompactnessLoss,
    PrototypeConsistencyLoss,
    PrototypeDiversityLoss,
    PrototypeSeparationLoss,
)


class PrototypeFactory:
    _registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, component: Any) -> None:
        cls._registry[name] = component
        logger.debug(f"Registered prototype component: {name}")

    @classmethod
    def get(cls, name: str) -> Any:
        if name not in cls._registry:
            raise ValueError(f"Unknown prototype component: {name}")
        return cls._registry[name]

    @staticmethod
    def create_prototype(
        embedding: torch.Tensor,
        class_id: int,
        modality: str = "shared",
        **kwargs: Any,
    ) -> Prototype:
        return Prototype(
            embedding=embedding,
            class_id=class_id,
            modality=modality,
            **kwargs,
        )

    @staticmethod
    def create_repository() -> PrototypeRepository:
        return PrototypeRepository()

    @staticmethod
    def create_generator(strategy: str = "centroid") -> PrototypeGenerator:
        return PrototypeGenerator(strategy=strategy)

    @staticmethod
    def create_memory(max_global: int = 1000, max_local: int = 100) -> PrototypeMemory:
        return PrototypeMemory(max_global=max_global, max_local=max_local)

    @staticmethod
    def create_updater(strategy: str = "ema") -> PrototypeUpdater:
        return PrototypeUpdater(strategy=strategy)

    @staticmethod
    def create_similarity(metric: str = "cosine") -> SimilarityEngine:
        return SimilarityEngine(metric=metric)

    @staticmethod
    def create_confidence(
        metric: str = "cosine",
        base_threshold: float = 0.5,
    ) -> ConfidenceEstimator:
        sim = SimilarityEngine(metric=metric)
        return ConfidenceEstimator(
            similarity_engine=sim,
            base_threshold=base_threshold,
        )

    @staticmethod
    def create_clustering(strategy: str = "kmeans") -> PrototypeClustering:
        return PrototypeClustering(strategy=strategy)

    @staticmethod
    def create_matcher(
        repository: PrototypeRepository,
        metric: str = "cosine",
    ) -> PrototypeMatcher:
        sim = SimilarityEngine(metric=metric)
        return PrototypeMatcher(
            repository=repository,
            similarity_engine=sim,
            metric=metric,
        )

    @staticmethod
    def create_metrics(prototypes: list[Prototype]) -> PrototypeMetrics:
        return PrototypeMetrics(prototypes)

    @staticmethod
    def create_loss(name: str, **kwargs: Any) -> nn.Module:
        loss_map = {
            "compactness": PrototypeCompactnessLoss,
            "separation": PrototypeSeparationLoss,
            "center": CenterLoss,
            "consistency": PrototypeConsistencyLoss,
            "diversity": PrototypeDiversityLoss,
        }
        if name not in loss_map:
            raise ValueError(
                f"Unknown loss: {name}. Available: {list(loss_map.keys())}"
            )
        return loss_map[name](**kwargs)

    @staticmethod
    def create_matcher_with_defaults(
        metric: str = "cosine",
    ) -> PrototypeMatcher:
        repo = PrototypeRepository()
        return PrototypeFactory.create_matcher(repo, metric=metric)

    @staticmethod
    def default_system(metric: str = "cosine") -> dict[str, Any]:
        repo = PrototypeRepository()
        sim = SimilarityEngine(metric=metric)
        matcher = PrototypeMatcher(
            repository=repo, similarity_engine=sim, metric=metric
        )
        confidence = ConfidenceEstimator(similarity_engine=sim)
        return {
            "repository": repo,
            "similarity": sim,
            "matcher": matcher,
            "confidence": confidence,
            "generator": PrototypeGenerator(strategy="centroid"),
            "updater": PrototypeUpdater(strategy="ema"),
            "memory": PrototypeMemory(),
        }
