from __future__ import annotations

from typing import Any, Callable

from app.core.logging import logger


class MetricRegistry:
    _metrics: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, fn: Callable) -> None:
        cls._metrics[name] = fn
        logger.debug(f"Registered metric: {name}")

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._metrics:
            raise ValueError(
                f"Unknown metric: {name}. Available: {list(cls._metrics.keys())}"
            )
        return cls._metrics[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._metrics.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._metrics.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        cls._metrics.clear()

    @classmethod
    def contains(cls, name: str) -> bool:
        return name in cls._metrics


class BaselineRegistry:
    _baselines: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, baseline_cls: type) -> None:
        cls._baselines[name] = baseline_cls
        logger.debug(f"Registered baseline: {name}")

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._baselines:
            raise ValueError(
                f"Unknown baseline: {name}. Available: {list(cls._baselines.keys())}"
            )
        return cls._baselines[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._baselines.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._baselines.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        cls._baselines.clear()

    @classmethod
    def contains(cls, name: str) -> bool:
        return name in cls._baselines


class AblationRegistry:
    _ablations: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    @classmethod
    def register(
        cls, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        cls._ablations[name] = fn
        logger.debug(f"Registered ablation: {name}")

    @classmethod
    def get(cls, name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        if name not in cls._ablations:
            raise ValueError(
                f"Unknown ablation: {name}. Available: {list(cls._ablations.keys())}"
            )
        return cls._ablations[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._ablations.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._ablations.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        cls._ablations.clear()

    @classmethod
    def contains(cls, name: str) -> bool:
        return name in cls._ablations


class ExperimentRegistry:
    _experiments: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(cls, experiment_id: str, config: dict[str, Any]) -> None:
        cls._experiments[experiment_id] = config
        logger.debug(f"Registered experiment: {experiment_id}")

    @classmethod
    def get(cls, experiment_id: str) -> dict[str, Any]:
        if experiment_id not in cls._experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
        return cls._experiments[experiment_id]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._experiments.keys())

    @classmethod
    def unregister(cls, experiment_id: str) -> None:
        cls._experiments.pop(experiment_id, None)

    @classmethod
    def clear(cls) -> None:
        cls._experiments.clear()

    @classmethod
    def contains(cls, experiment_id: str) -> bool:
        return experiment_id in cls._experiments

    @classmethod
    def count(cls) -> int:
        return len(cls._experiments)
