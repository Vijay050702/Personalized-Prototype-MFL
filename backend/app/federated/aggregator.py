from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.federated.adaptive_weighting import AdaptiveWeightCalculator
from app.federated.aggregation import PrototypeAggregator
from app.federated.communication import CommunicationHandler
from app.federated.completeness import CompletenessScorer
from app.federated.divergence import DivergenceCalculator
from app.federated.models import (
    AggregatedPrototype,
    AggregationRound,
    ClientPrototypePackage,
    DivergenceReport,
)
from app.federated.repository import FederatedRepository
from app.federated.scheduler import RoundScheduler
from app.federated.serialization import PrototypeSerializer
from app.federated.statistics import AggregationStatistics


class FederatedAggregator:
    def __init__(
        self,
        repository: FederatedRepository,
        scheduler: RoundScheduler,
        divergence_calculator: DivergenceCalculator,
        completeness_scorer: CompletenessScorer,
        weight_calculator: AdaptiveWeightCalculator,
        statistics: AggregationStatistics,
        serializer: PrototypeSerializer,
        communication_handler: CommunicationHandler,
        aggregator: PrototypeAggregator,
    ):
        self._repository = repository
        self._scheduler = scheduler
        self._divergence = divergence_calculator
        self._completeness = completeness_scorer
        self._weighting = weight_calculator
        self._statistics = statistics
        self._serializer = serializer
        self._communication = communication_handler
        self._aggregator = aggregator

    @property
    def repository(self) -> FederatedRepository:
        return self._repository

    @property
    def scheduler(self) -> RoundScheduler:
        return self._scheduler

    @property
    def statistics(self) -> AggregationStatistics:
        return self._statistics

    @property
    def completeness_scorer(self) -> CompletenessScorer:
        return self._completeness

    @property
    def divergence_calculator(self) -> DivergenceCalculator:
        return self._divergence

    @property
    def weight_calculator(self) -> AdaptiveWeightCalculator:
        return self._weighting

    @property
    def global_prototypes(self) -> list[AggregatedPrototype]:
        return self._repository.list_global_prototypes()

    def start_round(self, client_ids: list[str]) -> AggregationRound:
        round_obj = self._scheduler.new_round(client_ids)
        self._repository.store_round(round_obj)
        logger.info(
            f"Started aggregation round {round_obj.round_id} "
            f"with {len(client_ids)} clients"
        )
        return round_obj

    def receive_packages(
        self, packages_data: list[dict[str, Any]]
    ) -> list[ClientPrototypePackage]:
        packages = self._communication.receive_batch(packages_data)
        for pkg in packages:
            self._statistics.record_package(pkg)
        return packages

    def receive_packages_from_client(
        self, client_id: str, packages_data: list[dict[str, Any]]
    ) -> list[ClientPrototypePackage]:
        packages = self.receive_packages(packages_data)
        self._scheduler.client_arrived(client_id)
        return packages

    def compute_completeness(
        self, packages: list[ClientPrototypePackage]
    ) -> dict[str, float]:
        client_packages: dict[str, list[ClientPrototypePackage]] = {}
        for pkg in packages:
            if pkg.client_id not in client_packages:
                client_packages[pkg.client_id] = []
            client_packages[pkg.client_id].append(pkg)
        return self._completeness.client_completeness_scores(client_packages)

    def compute_divergences(
        self, packages: list[ClientPrototypePackage]
    ) -> dict[str, float]:
        divergences: dict[str, float] = {}
        for modality in {p.modality for p in packages}:
            for class_id in {p.class_id for p in packages}:
                class_modality_pkgs = [
                    p
                    for p in packages
                    if p.modality == modality and p.class_id == class_id
                ]
                if len(class_modality_pkgs) < 2:
                    for p in class_modality_pkgs:
                        divergences[p.client_id] = 0.0
                    continue

                global_proto = self._repository.get_global_prototype(modality, class_id)
                if global_proto is not None:
                    reference_pkg = ClientPrototypePackage(
                        client_id="global",
                        round_id=0,
                        modality=modality,
                        class_id=class_id,
                        prototype_vector=global_proto.prototype_vector,
                        sample_count=global_proto.sample_count,
                        embedding_dim=global_proto.embedding_dim,
                    )
                    for p in class_modality_pkgs:
                        d = self._divergence.compute(p, reference_pkg)
                        divergences[p.client_id] = d
                else:
                    scores = self._divergence.compute_pairwise(class_modality_pkgs)
                    client_avg: dict[str, list[float]] = {}
                    for c1, c2, d in scores:
                        client_avg.setdefault(c1, []).append(d)
                        client_avg.setdefault(c2, []).append(d)
                    for p in class_modality_pkgs:
                        if p.client_id in client_avg:
                            vals = client_avg[p.client_id]
                            divergences[p.client_id] = (
                                sum(vals) / len(vals) if vals else 0.0
                            )
                        else:
                            divergences[p.client_id] = 0.0
        return divergences

    def compute_weights(
        self,
        packages: list[ClientPrototypePackage],
        completeness_scores: dict[str, float],
        divergences: dict[str, float],
    ) -> list[float]:
        return self._weighting.compute_normalized_weights(
            packages, completeness_scores, divergences
        )

    def aggregate(
        self,
        packages: list[ClientPrototypePackage],
        weights: list[float],
    ) -> dict[tuple[int, str], AggregatedPrototype]:
        self._statistics.record_weight_distribution(
            self._scheduler.current_round_id, weights
        )
        return self._aggregator.per_class_aggregation(packages, weights)

    def store_aggregated(
        self,
        results: dict[tuple[int, str], AggregatedPrototype],
        round_id: int,
    ) -> None:
        for (class_id, modality), proto in results.items():
            proto.round_id = round_id
            self._repository.store_global_prototype(proto)

    def finalize_round(self) -> AggregationRound | None:
        if not self._scheduler.can_finalize():
            logger.warning("Round cannot be finalized yet")
            return None

        round_obj = self._scheduler.finalize_round()
        self._repository.store_round(round_obj)

        self._statistics.record_round_completion(
            round_id=round_obj.round_id,
            duration=round_obj.duration,
            num_participants=len(round_obj.participating_clients),
        )

        new_protos = self._repository.list_global_prototypes()
        old_protos = [
            self._repository.get_previous_prototype(p.modality, p.class_id) or p
            for p in new_protos
        ]
        self._statistics.record_drift(old_protos, new_protos)

        logger.info(
            f"Finalized round {round_obj.round_id} "
            f"({len(round_obj.participating_clients)} clients, "
            f"{round_obj.duration:.2f}s)"
        )
        return round_obj

    def run_round(
        self,
        client_packages: dict[str, list[dict[str, Any]]],
    ) -> dict[tuple[int, str], AggregatedPrototype]:
        client_ids = list(client_packages.keys())
        self.start_round(client_ids)

        all_packages: list[ClientPrototypePackage] = []
        for client_id, pkg_data in client_packages.items():
            received = self.receive_packages_from_client(client_id, pkg_data)
            all_packages.extend(received)

        completeness_scores = self.compute_completeness(all_packages)
        divergences = self.compute_divergences(all_packages)
        weights = self.compute_weights(all_packages, completeness_scores, divergences)

        results = self.aggregate(all_packages, weights)

        round_id = self._scheduler.current_round_id
        self.store_aggregated(results, round_id)

        self._repository.store_client_packages(round_id, all_packages)

        self.finalize_round()

        return results

    def get_divergence_reports(
        self, packages: list[ClientPrototypePackage]
    ) -> list[DivergenceReport]:
        reports: list[DivergenceReport] = []
        for modality in {p.modality for p in packages}:
            for class_id in {p.class_id for p in packages}:
                ref = self._repository.get_global_prototype(modality, class_id)
                if ref is None:
                    continue
                reference_pkg = ClientPrototypePackage(
                    client_id="global",
                    round_id=0,
                    modality=modality,
                    class_id=class_id,
                    prototype_vector=ref.prototype_vector,
                    sample_count=ref.sample_count,
                    embedding_dim=ref.embedding_dim,
                )
                for p in packages:
                    if p.modality == modality and p.class_id == class_id:
                        d = self._divergence.compute(p, reference_pkg)
                        reports.append(
                            DivergenceReport(
                                client_id=p.client_id,
                                modality=modality,
                                class_id=class_id,
                                divergence_score=d,
                                divergence_metric=self._divergence.metric,
                            )
                        )
        return reports

    def get_statistics_summary(self) -> dict[str, Any]:
        return self._statistics.to_dict()

    def create_snapshot(self) -> int:
        return self._repository.create_snapshot()

    def restore_snapshot(self, version: int) -> None:
        self._repository.restore_snapshot(version)

    def export_state(self) -> dict[str, Any]:
        return self._repository.export_state()

    def import_state(self, state: dict[str, Any]) -> None:
        self._repository.import_state(state)

    def clear(self) -> None:
        self._repository.clear()
        self._statistics.reset()
        self._communication.clear_history()
        logger.info("Cleared federated aggregator state")
