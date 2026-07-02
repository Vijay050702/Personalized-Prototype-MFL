from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from app.core.logging import logger
from app.federated.models import AggregatedPrototype
from app.knowledge_transfer.cross_modal_mapper import CrossModalMapper
from app.knowledge_transfer.modality_graph import ModalityGraph
from app.knowledge_transfer.prototype_generator import (
    PrototypeGenerator,
    SynthesisResult,
)
from app.knowledge_transfer.utils import TransferLogger
from app.knowledge_transfer.validation import validate_missing_modalities


@dataclass
class InferenceOutput:
    modality: str
    class_id: int
    prototype_vector: list[float]
    embedding_dim: int
    confidence: float
    source_modality: str
    path: list[str]


class InferenceEngine:
    def __init__(
        self,
        mapper: CrossModalMapper,
        graph: ModalityGraph,
        generator: PrototypeGenerator,
        logger_instance: TransferLogger | None = None,
    ):
        self._mapper = mapper
        self._graph = graph
        self._generator = generator
        self._logger = logger_instance or TransferLogger()

    def infer_missing_modalities(
        self,
        available_prototypes: list[AggregatedPrototype],
        target_modalities: set[str],
        all_known_modalities: set[str] | None = None,
    ) -> list[InferenceOutput]:
        available_mods = {p.modality for p in available_prototypes}
        missing = target_modalities - available_mods

        if not missing:
            return []

        known = all_known_modalities or (available_mods | target_modalities)
        validate_missing_modalities(available_mods, missing, known)

        results: list[InferenceOutput] = []
        for target_mod in sorted(missing):
            path_sources: list[tuple[AggregatedPrototype, list[str]]] = []
            for proto in available_prototypes:
                path = self._graph.find_path(proto.modality, target_mod)
                if path is not None:
                    path_sources.append((proto, path))

            if not path_sources:
                logger.warning(f"No path to {target_mod} from any available modality")
                continue

            path_sources.sort(
                key=lambda x: (
                    len(x[1]),
                    -x[0].confidence,
                )
            )
            best_proto, best_path = path_sources[0]

            embedding_t = torch.tensor(best_proto.prototype_vector, dtype=torch.float32)

            try:
                result = self._mapper.translate(
                    best_proto.modality, target_mod, embedding_t
                )
            except ValueError as e:
                logger.warning(
                    f"Translation failed {best_proto.modality} -> {target_mod}: {e}"
                )
                continue

            result_vector = result.detach().cpu().tolist()
            if isinstance(result_vector, float):
                result_vector = [result_vector]

            confidence = best_proto.confidence * (0.9 ** (len(best_path) - 1))

            self._logger.log_translation(
                source_modality=best_proto.modality,
                target_modality=target_mod,
                class_id=best_proto.class_id,
                confidence=confidence,
                duration=0.0,
            )

            results.append(
                InferenceOutput(
                    modality=target_mod,
                    class_id=best_proto.class_id,
                    prototype_vector=result_vector,
                    embedding_dim=result.size(-1) if result.dim() > 0 else 1,
                    confidence=confidence,
                    source_modality=best_proto.modality,
                    path=best_path,
                )
            )

        return results

    def infer_single(
        self,
        prototype: AggregatedPrototype,
        target_modality: str,
    ) -> InferenceOutput:
        synthesis = self._generator.synthesize(prototype, target_modality)
        return InferenceOutput(
            modality=synthesis.modality,
            class_id=synthesis.class_id,
            prototype_vector=synthesis.prototype_vector,
            embedding_dim=synthesis.embedding_dim,
            confidence=synthesis.confidence,
            source_modality=synthesis.source_modality,
            path=synthesis.path,
        )

    def batch_infer(
        self,
        prototypes: list[AggregatedPrototype],
        target_modality: str,
    ) -> list[InferenceOutput]:
        results: list[InferenceOutput] = []
        for proto in prototypes:
            if proto.modality == target_modality:
                continue
            try:
                output = self.infer_single(proto, target_modality)
                results.append(output)
            except ValueError as e:
                logger.warning(f"Skipping {proto.class_id}/{proto.modality}: {e}")
                continue
        return results

    @property
    def mapper(self) -> CrossModalMapper:
        return self._mapper

    @property
    def graph(self) -> ModalityGraph:
        return self._graph

    @property
    def generator(self) -> PrototypeGenerator:
        return self._generator
