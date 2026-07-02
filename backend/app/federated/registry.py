from __future__ import annotations

from typing import Any, Callable

import torch

from app.federated.adaptive_weighting import AdaptiveWeightCalculator
from app.federated.aggregation import PrototypeAggregator
from app.federated.completeness import CompletenessScorer
from app.federated.communication import CommunicationHandler
from app.federated.divergence import DivergenceCalculator
from app.federated.repository import FederatedRepository
from app.federated.scheduler import RoundScheduler
from app.federated.serialization import PrototypeSerializer
from app.federated.statistics import AggregationStatistics


class FederatedRegistry:
    def __init__(self):
        self._divergence_metrics: dict[str, str] = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "manhattan": "manhattan",
        }
        self._weight_strategies: dict[str, type[AdaptiveWeightCalculator]] = {
            "adaptive": AdaptiveWeightCalculator,
        }
        self._aggregation_methods: dict[str, type[PrototypeAggregator]] = {
            "weighted": PrototypeAggregator,
        }
        self._component_factories: dict[str, Callable[..., Any]] = {}

    def register_divergence_metric(self, name: str, metric: str) -> None:
        if name in self._divergence_metrics:
            raise ValueError(f"Divergence metric '{name}' is already registered")
        self._divergence_metrics[name] = metric

    def get_divergence_metric(self, name: str) -> str:
        if name not in self._divergence_metrics:
            raise ValueError(
                f"Unknown divergence metric '{name}'. "
                f"Available: {self.list_divergence_metrics()}"
            )
        return self._divergence_metrics[name]

    def list_divergence_metrics(self) -> list[str]:
        return sorted(self._divergence_metrics.keys())

    def unregister_divergence_metric(self, name: str) -> None:
        self._divergence_metrics.pop(name, None)

    def register_weight_strategy(
        self, name: str, strategy_cls: type[AdaptiveWeightCalculator]
    ) -> None:
        if name in self._weight_strategies:
            raise ValueError(f"Weight strategy '{name}' is already registered")
        self._weight_strategies[name] = strategy_cls

    def get_weight_strategy(self, name: str) -> type[AdaptiveWeightCalculator]:
        if name not in self._weight_strategies:
            raise ValueError(
                f"Unknown weight strategy '{name}'. "
                f"Available: {self.list_weight_strategies()}"
            )
        return self._weight_strategies[name]

    def list_weight_strategies(self) -> list[str]:
        return sorted(self._weight_strategies.keys())

    def unregister_weight_strategy(self, name: str) -> None:
        self._weight_strategies.pop(name, None)

    def register_aggregation_method(
        self, name: str, method_cls: type[PrototypeAggregator]
    ) -> None:
        if name in self._aggregation_methods:
            raise ValueError(f"Aggregation method '{name}' is already registered")
        self._aggregation_methods[name] = method_cls

    def get_aggregation_method(self, name: str) -> type[PrototypeAggregator]:
        if name not in self._aggregation_methods:
            raise ValueError(
                f"Unknown aggregation method '{name}'. "
                f"Available: {self.list_aggregation_methods()}"
            )
        return self._aggregation_methods[name]

    def list_aggregation_methods(self) -> list[str]:
        return sorted(self._aggregation_methods.keys())

    def unregister_aggregation_method(self, name: str) -> None:
        self._aggregation_methods.pop(name, None)

    def register_component(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._component_factories:
            raise ValueError(f"Component '{name}' is already registered")
        self._component_factories[name] = factory

    def get_component(self, name: str, **kwargs: Any) -> Any:
        if name not in self._component_factories:
            raise ValueError(
                f"Unknown component '{name}'. Available: {self.list_components()}"
            )
        return self._component_factories[name](**kwargs)

    def list_components(self) -> list[str]:
        return sorted(self._component_factories.keys())

    def unregister_component(self, name: str) -> None:
        self._component_factories.pop(name, None)

    def create_divergence_calculator(
        self, metric: str = "cosine"
    ) -> DivergenceCalculator:
        resolved = self.get_divergence_metric(metric)
        return DivergenceCalculator(metric=resolved)

    def create_repository(self) -> FederatedRepository:
        return FederatedRepository()

    def create_serializer(self) -> PrototypeSerializer:
        return PrototypeSerializer()

    def create_statistics(self) -> AggregationStatistics:
        return AggregationStatistics()

    def create_scheduler(
        self,
        timeout_seconds: float = 300.0,
        min_clients: int = 1,
        allow_partial: bool = True,
        max_late_clients: int = 0,
    ) -> RoundScheduler:
        return RoundScheduler(
            timeout_seconds=timeout_seconds,
            min_clients=min_clients,
            allow_partial=allow_partial,
            max_late_clients=max_late_clients,
        )

    def create_completeness_scorer(
        self, expected_modalities: list[str] | None = None
    ) -> CompletenessScorer:
        return CompletenessScorer(expected_modalities=expected_modalities)

    def create_weight_calculator(
        self,
        temperature: float = 1.0,
        divergence_metric: str = "cosine",
        divergence_weight: float = 0.5,
        sample_weight: float = 0.3,
        completeness_weight: float = 0.2,
    ) -> AdaptiveWeightCalculator:
        dc = self.create_divergence_calculator(metric=divergence_metric)
        return AdaptiveWeightCalculator(
            temperature=temperature,
            divergence_weight=divergence_weight,
            sample_weight=sample_weight,
            completeness_weight=completeness_weight,
            divergence_calculator=dc,
        )

    def create_aggregator(self, epsilon: float = 1e-8) -> PrototypeAggregator:
        return PrototypeAggregator(epsilon=epsilon)

    def create_communication_handler(
        self,
    ) -> CommunicationHandler:
        return CommunicationHandler(serializer=self.create_serializer())

    def to_config(self) -> dict[str, Any]:
        return {
            "divergence_metrics": self.list_divergence_metrics(),
            "weight_strategies": self.list_weight_strategies(),
            "aggregation_methods": self.list_aggregation_methods(),
            "custom_components": self.list_components(),
        }
