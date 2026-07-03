from __future__ import annotations

from typing import Any

import torch

from app.knowledge_transfer.similarity import Similarity
from app.personalization.adaptive_gate import AdaptiveGate
from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.profile import ClientProfile
from app.personalization.utils import PersonalizationLogger
from app.personalization.validation import (
    validate_confidence_range,
    validate_dimensions,
    validate_fusion_sources,
)
from app.personalization.weighting import WeightCalculator


class FusionEngine:
    def __init__(
        self,
        strategy: str = "weighted_sum",
        weight_calculator: WeightCalculator | None = None,
        adaptive_gate: AdaptiveGate | None = None,
        logger_instance: PersonalizationLogger | None = None,
    ):
        if strategy not in {
            "weighted_sum",
            "adaptive",
            "confidence_weighted",
            "learnable",
        }:
            raise ValueError(
                f"Unknown fusion strategy '{strategy}'. "
                f"Choose from: weighted_sum, adaptive, confidence_weighted, learnable"
            )
        self._strategy = strategy
        self._weight_calculator = weight_calculator or WeightCalculator(
            strategy="fixed"
        )
        self._adaptive_gate = adaptive_gate
        self._logger = logger_instance or PersonalizationLogger()

    def fuse(
        self,
        personalized_prototype: PersonalizedPrototype,
        client_profile: ClientProfile | None = None,
    ) -> PersonalizedPrototype:
        sources = personalized_prototype.available_sources()
        if not sources:
            raise ValueError("No prototype sources available for fusion")

        validate_fusion_sources(sources)
        source_embeddings = self._get_embeddings(personalized_prototype, sources)

        for name, emb in source_embeddings.items():
            validate_dimensions(emb, personalized_prototype.embedding_dim, name)

        weights = self._compute_weights(
            sources=sources,
            personalized_prototype=personalized_prototype,
            source_embeddings=source_embeddings,
            client_profile=client_profile,
        )

        fused = self._apply_weights(source_embeddings, weights)

        confidence = self._compute_fusion_confidence(personalized_prototype, weights)
        validate_confidence_range(confidence)

        personalized_prototype.personalized_prototype = fused.detach().cpu().tolist()
        personalized_prototype.fusion_weights = weights
        personalized_prototype.confidence = confidence

        self._logger.log_fusion(
            client_id=personalized_prototype.client_id,
            class_id=personalized_prototype.class_id,
            modality=personalized_prototype.modality,
            fusion_strategy=self._strategy,
            weights=weights,
            confidence=confidence,
        )

        return personalized_prototype

    def _get_embeddings(
        self,
        pp: PersonalizedPrototype,
        sources: list[str],
    ) -> dict[str, torch.Tensor]:
        embeddings: dict[str, torch.Tensor] = {}
        if "local" in sources and pp.local_prototype is not None:
            embeddings["local"] = torch.tensor(pp.local_prototype, dtype=torch.float32)
        if "global" in sources and pp.global_prototype is not None:
            embeddings["global"] = torch.tensor(
                pp.global_prototype, dtype=torch.float32
            )
        if "cross_modal" in sources and pp.cross_modal_prototype is not None:
            embeddings["cross_modal"] = torch.tensor(
                pp.cross_modal_prototype, dtype=torch.float32
            )
        return embeddings

    def _compute_weights(
        self,
        sources: list[str],
        personalized_prototype: PersonalizedPrototype,
        source_embeddings: dict[str, torch.Tensor],
        client_profile: ClientProfile | None,
    ) -> dict[str, float]:
        if self._strategy == "adaptive" and self._adaptive_gate is not None:
            global_conf = personalized_prototype.confidence
            profile_features: list[float] = []
            if client_profile is not None:
                profile_features = [
                    client_profile.average_confidence,
                    client_profile.prototype_drift,
                    client_profile.confidence_trend,
                ]
            return self._adaptive_gate.compute_weights(
                source_embeddings=source_embeddings,
                global_confidence=global_conf,
                profile_features=profile_features if profile_features else None,
            )

        if self._strategy == "confidence_weighted":
            confidences: dict[str, float] = {}
            for s in sources:
                if s == "local":
                    confidences[s] = 0.9
                elif s == "global":
                    confidences[s] = 0.8
                else:
                    confidences[s] = 0.7
            return self._weight_calculator.compute(
                sources=sources,
                confidences=confidences,
            )

        return self._weight_calculator.compute(
            sources=sources,
            confidences=None,
            embeddings=source_embeddings,
        )

    def _apply_weights(
        self,
        embeddings: dict[str, torch.Tensor],
        weights: dict[str, float],
    ) -> torch.Tensor:
        weighted_sum: torch.Tensor | None = None
        for name, emb in embeddings.items():
            w = weights.get(name, 0.0)
            if weighted_sum is None:
                weighted_sum = w * emb
            else:
                weighted_sum = weighted_sum + w * emb
        if weighted_sum is None:
            raise ValueError("No embeddings to fuse")
        return weighted_sum

    def _compute_fusion_confidence(
        self,
        pp: PersonalizedPrototype,
        weights: dict[str, float],
    ) -> float:
        weighted_conf = 0.0
        total_weight = 0.0
        source_conf_map: dict[str, float] = {}

        if pp.local_prototype is not None:
            source_conf_map["local"] = 0.9
        if pp.global_prototype is not None:
            source_conf_map["global"] = 0.8
        if pp.cross_modal_prototype is not None:
            source_conf_map["cross_modal"] = 0.7

        for src, w in weights.items():
            base_conf = source_conf_map.get(src, 0.5)
            weighted_conf += w * base_conf
            total_weight += w

        if total_weight == 0:
            return 0.0
        return min(1.0, weighted_conf / total_weight)

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def weight_calculator(self) -> WeightCalculator:
        return self._weight_calculator

    def to_config(self) -> dict[str, Any]:
        return {
            "strategy": self._strategy,
            "weight_calculator": self._weight_calculator.to_config(),
            "adaptive_gate": self._adaptive_gate.to_config()
            if self._adaptive_gate
            else None,
        }
