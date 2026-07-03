from __future__ import annotations

from typing import Any

import torch

from app.federated.models import AggregatedPrototype
from app.knowledge_transfer.inference import InferenceOutput
from app.prototypes.prototype import Prototype
from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.profile import ClientProfile
from app.personalization.utils import PersonalizationLogger
from app.personalization.validation import validate_confidence_range


class PrototypeSelector:
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        logger_instance: PersonalizationLogger | None = None,
    ):
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
            )
        self._threshold = confidence_threshold
        self._logger = logger_instance or PersonalizationLogger()

    def select_best_local(
        self,
        prototypes: list[Prototype],
        class_id: int,
        modality: str,
    ) -> Prototype | None:
        candidates = [
            p
            for p in prototypes
            if p.class_id == class_id
            and p.modality == modality
            and p.confidence >= self._threshold
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.confidence)

    def select_best_global(
        self,
        prototypes: list[AggregatedPrototype],
        class_id: int,
        modality: str,
    ) -> AggregatedPrototype | None:
        candidates = [
            p
            for p in prototypes
            if p.class_id == class_id
            and p.modality == modality
            and p.confidence >= self._threshold
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.confidence)

    def select_best_transferred(
        self,
        outputs: list[InferenceOutput],
        class_id: int,
        modality: str,
    ) -> InferenceOutput | None:
        candidates = [
            o
            for o in outputs
            if o.class_id == class_id
            and o.modality == modality
            and o.confidence >= self._threshold
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda o: o.confidence)

    def select_sources(
        self,
        local_prototypes: list[Prototype],
        global_prototypes: list[AggregatedPrototype],
        transferred: list[InferenceOutput],
        class_id: int,
        modality: str,
        client_profile: ClientProfile | None = None,
    ) -> PersonalizedPrototype:
        local = self.select_best_local(local_prototypes, class_id, modality)
        global_p = self.select_best_global(global_prototypes, class_id, modality)
        cross = self.select_best_transferred(transferred, class_id, modality)

        client_id = client_profile.client_id if client_profile else "unknown"

        result = PersonalizedPrototype(
            client_id=client_id,
            class_id=class_id,
            modality=modality,
        )

        if local is not None:
            result.local_prototype = local.embedding.detach().cpu().tolist()
            result.confidence = max(result.confidence, local.confidence)

        if global_p is not None:
            result.global_prototype = global_p.prototype_vector
            result.confidence = max(result.confidence, global_p.confidence)

        if cross is not None:
            result.cross_modal_prototype = cross.prototype_vector
            result.confidence = max(result.confidence, cross.confidence)

        if local is not None:
            result.embedding_dim = local.embedding.size(-1)
        elif global_p is not None:
            result.embedding_dim = global_p.embedding_dim
        elif cross is not None:
            result.embedding_dim = cross.embedding_dim

        return result

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    def to_config(self) -> dict[str, Any]:
        return {"confidence_threshold": self._threshold}
