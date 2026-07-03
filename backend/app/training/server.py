from __future__ import annotations

from typing import Any

import torch

from app.core.logging import logger
from app.federated.aggregator import FederatedAggregator
from app.federated.models import AggregatedPrototype
from app.federated.repository import FederatedRepository
from app.knowledge_transfer.inference import InferenceEngine, InferenceOutput
from app.knowledge_transfer.prototype_generator import SynthesisResult
from app.personalization.personalized_memory import PersonalizedMemory
from app.training.logger import TrainingLogger
from app.training.state import ServerState
from app.training.utils import Timer


class Server:
    def __init__(
        self,
        server_id: str = "server",
        federated_aggregator: FederatedAggregator | None = None,
        federated_repository: FederatedRepository | None = None,
        inference_engine: InferenceEngine | None = None,
        personalized_memory: PersonalizedMemory | None = None,
        logger_instance: TrainingLogger | None = None,
    ) -> None:
        self._server_id = server_id
        self._aggregator = federated_aggregator
        self._repository = federated_repository or FederatedRepository()
        self._inference_engine = inference_engine
        self._personalized_memory = personalized_memory or PersonalizedMemory()
        self._logger = logger_instance or TrainingLogger()
        self._state = ServerState()
        self._timer = Timer()
        self._client_results: dict[str, list[Any]] = {}
        self._aggregated_prototypes: dict[tuple[int, str], AggregatedPrototype] = {}
        self._synthesized_prototypes: list[SynthesisResult] = []
        self._inferred_outputs: list[InferenceOutput] = []

    @property
    def server_id(self) -> str:
        return self._server_id

    @property
    def state(self) -> ServerState:
        return self._state

    @property
    def repository(self) -> FederatedRepository:
        return self._repository

    @property
    def aggregated_prototypes(self) -> dict[tuple[int, str], AggregatedPrototype]:
        return dict(self._aggregated_prototypes)

    def collect_client_results(
        self,
        client_results: dict[str, list[Any]],
    ) -> int:
        self._client_results = client_results
        total_packages = sum(len(pkgs) for pkgs in client_results.values())
        self._logger.log_aggregation(
            round_id=self._state.current_round,
            num_clients=len(client_results),
            num_prototypes=total_packages,
        )
        return total_packages

    def aggregate_prototypes(
        self,
        round_id: int,
    ) -> dict[tuple[int, str], AggregatedPrototype]:
        if self._aggregator is None:
            raise ValueError("FederatedAggregator not configured")

        self._timer.start()
        client_packages: dict[str, list[dict[str, Any]]] = {}
        for client_id, packages in self._client_results.items():
            serialized = []
            for pkg in packages:
                if hasattr(pkg, "model_dump"):
                    serialized.append(pkg.model_dump())
                elif isinstance(pkg, dict):
                    serialized.append(pkg)
                else:
                    serialized.append(
                        {
                            "client_id": client_id,
                            "round_id": round_id,
                            "modality": pkg.modality,
                            "class_id": pkg.class_id,
                            "prototype_vector": (
                                pkg.prototype_vector
                                if hasattr(pkg, "prototype_vector")
                                else pkg.embedding.detach().cpu().tolist()
                            ),
                            "sample_count": pkg.sample_count,
                            "embedding_dim": pkg.embedding_dim,
                            "confidence": pkg.confidence,
                        }
                    )
            client_packages[client_id] = serialized

        results = self._aggregator.run_round(client_packages)
        self._aggregated_prototypes = results

        for key, proto in results.items():
            self._repository.store_global_prototype(proto)

        agg_time = self._timer.stop()
        self._state.current_round = round_id
        self._state.rounds_completed += 1
        self._state.global_prototype_count = self._repository.global_count()

        self._logger.log_aggregation(
            round_id=round_id,
            num_clients=len(client_packages),
            num_prototypes=len(results),
        )

        return results

    def run_knowledge_transfer(
        self,
        round_id: int,
        all_modalities: set[str] | None = None,
    ) -> tuple[list[SynthesisResult], list[InferenceOutput]]:
        self._timer.start()

        global_protos = self._repository.list_global_prototypes()
        self._synthesized_prototypes = []
        self._inferred_outputs = []

        if self._inference_engine is not None and global_protos:
            mods = all_modalities or {p.modality for p in global_protos}
            inferred = self._inference_engine.infer_missing_modalities(
                available_prototypes=global_protos,
                target_modalities=mods,
                all_known_modalities=mods,
            )
            self._inferred_outputs = inferred

        kt_time = self._timer.stop()

        self._logger.log_knowledge_transfer(
            round_id=round_id,
            num_synthesized=len(self._synthesized_prototypes)
            + len(self._inferred_outputs),
            modalities=list(all_modalities) if all_modalities else None,
        )

        return self._synthesized_prototypes, self._inferred_outputs

    def broadcast_global_prototypes(
        self,
    ) -> list[AggregatedPrototype]:
        return self._repository.list_global_prototypes()

    def broadcast_synthesized_prototypes(
        self,
    ) -> list[InferenceOutput]:
        return list(self._inferred_outputs)

    def get_aggregated_for_modality(
        self,
        modality: str,
        class_id: int,
    ) -> AggregatedPrototype | None:
        return self._repository.get_global_prototype(modality, class_id)

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "server_id": self._server_id,
            "state": self._state.to_dict(),
            "repository_state": self._repository.export_state(),
            "aggregated_count": len(self._aggregated_prototypes),
            "synthesized_count": len(self._synthesized_prototypes),
            "inferred_count": len(self._inferred_outputs),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "repository_state" in state_dict:
            self._repository.import_state(state_dict["repository_state"])
        if "state" in state_dict:
            s = state_dict["state"]
            self._state.current_round = s.get("current_round", 0)
            self._state.rounds_completed = s.get("rounds_completed", 0)
            self._state.global_prototype_count = s.get("global_prototype_count", 0)
