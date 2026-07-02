from __future__ import annotations

from typing import Any

from app.federated.adaptive_weighting import AdaptiveWeightCalculator
from app.federated.aggregation import PrototypeAggregator
from app.federated.aggregator import FederatedAggregator
from app.federated.communication import CommunicationHandler
from app.federated.completeness import CompletenessScorer
from app.federated.divergence import DivergenceCalculator
from app.federated.registry import FederatedRegistry
from app.federated.repository import FederatedRepository
from app.federated.scheduler import RoundScheduler
from app.federated.serialization import PrototypeSerializer
from app.federated.statistics import AggregationStatistics


class FederatedFactory:
    @staticmethod
    def create_default() -> FederatedAggregator:
        registry = FederatedRegistry()
        repository = registry.create_repository()
        serializer = registry.create_serializer()
        communication = CommunicationHandler(serializer=serializer)
        scheduler = registry.create_scheduler()
        divergence = registry.create_divergence_calculator(metric="cosine")
        completeness = registry.create_completeness_scorer()
        weighting = registry.create_weight_calculator()
        aggregator = registry.create_aggregator()
        statistics = registry.create_statistics()

        return FederatedAggregator(
            repository=repository,
            scheduler=scheduler,
            divergence_calculator=divergence,
            completeness_scorer=completeness,
            weight_calculator=weighting,
            statistics=statistics,
            serializer=serializer,
            communication_handler=communication,
            aggregator=aggregator,
        )

    @staticmethod
    def create_with_config(config: dict[str, Any]) -> FederatedAggregator:
        registry = FederatedRegistry()

        scheduler_config = config.get("scheduler", {})
        scheduler = registry.create_scheduler(
            timeout_seconds=scheduler_config.get("timeout_seconds", 300.0),
            min_clients=scheduler_config.get("min_clients", 1),
            allow_partial=scheduler_config.get("allow_partial", True),
            max_late_clients=scheduler_config.get("max_late_clients", 0),
        )

        weighting_config = config.get("weighting", {})
        divergence = registry.create_divergence_calculator(
            metric=weighting_config.get("divergence_metric", "cosine")
        )
        completeness = registry.create_completeness_scorer(
            expected_modalities=config.get("expected_modalities")
        )
        weighting = registry.create_weight_calculator(
            temperature=weighting_config.get("temperature", 1.0),
            divergence_metric=weighting_config.get("divergence_metric", "cosine"),
            divergence_weight=weighting_config.get("divergence_weight", 0.5),
            sample_weight=weighting_config.get("sample_weight", 0.3),
            completeness_weight=weighting_config.get("completeness_weight", 0.2),
        )

        aggregation_config = config.get("aggregation", {})
        aggregator = registry.create_aggregator(
            epsilon=aggregation_config.get("epsilon", 1e-8)
        )

        repository = registry.create_repository()
        serializer = registry.create_serializer()
        communication = CommunicationHandler(serializer=serializer)
        statistics = registry.create_statistics()

        return FederatedAggregator(
            repository=repository,
            scheduler=scheduler,
            divergence_calculator=divergence,
            completeness_scorer=completeness,
            weight_calculator=weighting,
            statistics=statistics,
            serializer=serializer,
            communication_handler=communication,
            aggregator=aggregator,
        )

    @staticmethod
    def create_custom(
        repository: FederatedRepository | None = None,
        scheduler: RoundScheduler | None = None,
        divergence_calculator: DivergenceCalculator | None = None,
        completeness_scorer: CompletenessScorer | None = None,
        weight_calculator: AdaptiveWeightCalculator | None = None,
        statistics: AggregationStatistics | None = None,
        serializer: PrototypeSerializer | None = None,
        communication_handler: CommunicationHandler | None = None,
        aggregator: PrototypeAggregator | None = None,
    ) -> FederatedAggregator:
        registry = FederatedRegistry()
        return FederatedAggregator(
            repository=repository or registry.create_repository(),
            scheduler=scheduler or registry.create_scheduler(),
            divergence_calculator=divergence_calculator
            or registry.create_divergence_calculator(),
            completeness_scorer=completeness_scorer
            or registry.create_completeness_scorer(),
            weight_calculator=weight_calculator or registry.create_weight_calculator(),
            statistics=statistics or registry.create_statistics(),
            serializer=serializer or registry.create_serializer(),
            communication_handler=communication_handler or CommunicationHandler(),
            aggregator=aggregator or registry.create_aggregator(),
        )
