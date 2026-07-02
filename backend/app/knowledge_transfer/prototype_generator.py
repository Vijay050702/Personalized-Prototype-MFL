from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from app.core.logging import logger
from app.federated.models import AggregatedPrototype
from app.knowledge_transfer.cross_modal_mapper import CrossModalMapper
from app.knowledge_transfer.modality_graph import ModalityGraph
from app.knowledge_transfer.utils import TransferLogger
from app.knowledge_transfer.validation import (
    validate_missing_modalities,
    validate_no_nan,
    validate_prototype_size,
)


@dataclass
class SynthesisResult:
    modality: str
    class_id: int
    prototype_vector: list[float]
    embedding_dim: int
    confidence: float
    source_modality: str
    path: list[str]


class PrototypeGenerator:
    def __init__(
        self,
        mapper: CrossModalMapper,
        graph: ModalityGraph,
        logger_instance: TransferLogger | None = None,
    ):
        self._mapper = mapper
        self._graph = graph
        self._logger = logger_instance or TransferLogger()

    def synthesize(
        self,
        source_prototype: AggregatedPrototype,
        target_modality: str,
    ) -> SynthesisResult:
        source_modality = source_prototype.modality
        if source_modality == target_modality:
            raise ValueError(
                f"Source and target modality are the same: {source_modality}"
            )

        embedding_t = torch.tensor(
            source_prototype.prototype_vector, dtype=torch.float32
        )
        validate_prototype_size(embedding_t)
        validate_no_nan(embedding_t, "source_prototype")

        path = self._graph.find_path(source_modality, target_modality)
        if path is None:
            raise ValueError(
                f"No path from {source_modality} to {target_modality} in modality graph"
            )

        result = self._mapper.translate(source_modality, target_modality, embedding_t)
        validate_no_nan(result, "synthesized_embedding")

        synthesized = result.detach().cpu().tolist()
        if isinstance(synthesized, float):
            synthesized = [synthesized]

        confidence = self._estimate_confidence(source_prototype.confidence, path)

        self._logger.log_translation(
            source_modality=source_modality,
            target_modality=target_modality,
            class_id=source_prototype.class_id,
            confidence=confidence,
            duration=0.0,
        )

        return SynthesisResult(
            modality=target_modality,
            class_id=source_prototype.class_id,
            prototype_vector=synthesized,
            embedding_dim=result.size(-1) if result.dim() > 0 else 1,
            confidence=confidence,
            source_modality=source_modality,
            path=path,
        )

    def batch_synthesize(
        self,
        prototypes: list[AggregatedPrototype],
        target_modality: str,
    ) -> list[SynthesisResult]:
        if not prototypes:
            return []

        grouped: dict[str, list[AggregatedPrototype]] = {}
        for p in prototypes:
            grouped.setdefault(p.modality, []).append(p)

        results: list[SynthesisResult] = []
        for source_mod, mod_protos in grouped.items():
            if source_mod == target_modality:
                continue
            for proto in mod_protos:
                results.append(self.synthesize(proto, target_modality))
        return results

    def synthesize_missing_modalities(
        self,
        available_prototypes: list[AggregatedPrototype],
        all_modalities: set[str],
    ) -> list[SynthesisResult]:
        available_mods = {p.modality for p in available_prototypes}
        missing = all_modalities - available_mods
        validate_missing_modalities(available_mods, missing, all_modalities)

        results: list[SynthesisResult] = []
        for target_mod in sorted(missing):
            candidates = [
                p
                for p in available_prototypes
                if self._graph.find_path(p.modality, target_mod) is not None
            ]
            if not candidates:
                logger.warning(f"No path to {target_mod} from any available modality")
                continue

            best = max(candidates, key=lambda p: p.confidence)
            results.append(self.synthesize(best, target_mod))
        return results

    @staticmethod
    def _estimate_confidence(
        source_confidence: float,
        path: list[str],
    ) -> float:
        decay = 0.9 ** (len(path) - 1)
        return source_confidence * decay

    @property
    def mapper(self) -> CrossModalMapper:
        return self._mapper

    @property
    def graph(self) -> ModalityGraph:
        return self._graph

    @property
    def logger(self) -> TransferLogger:
        return self._logger
