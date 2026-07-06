from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.evaluation.baselines import BaselineFactory
from app.evaluation.evaluator import EvaluationEngine
from app.evaluation.experiment_runner import ExperimentRunner
from app.evaluation.metrics import MetricFactory
from app.evaluation.registry import (
    AblationRegistry,
    BaselineRegistry,
    ExperimentRegistry,
    MetricRegistry,
)


class EvaluationFactory:
    @staticmethod
    def create_engine(config: dict[str, Any] | None = None) -> EvaluationEngine:
        cfg = config or {}
        engine = EvaluationEngine(config=cfg)
        logger.debug("EvaluationEngine created via factory")
        return engine

    @staticmethod
    def create_runner(config: dict[str, Any] | None = None) -> ExperimentRunner:
        cfg = config or {}
        runner = ExperimentRunner(config=cfg)
        logger.debug("ExperimentRunner created via factory")
        return runner

    @staticmethod
    def create_metric(name: str, **kwargs: Any) -> Any:
        return MetricFactory.create(name, **kwargs)

    @staticmethod
    def create_baseline(name: str, config: dict[str, Any] | None = None) -> Any:
        return BaselineFactory.create(name, config or {})

    @staticmethod
    def from_config(config: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "metrics" in config:
            result["metrics"] = [
                EvaluationFactory.create_metric(m) if isinstance(m, str) else m
                for m in config["metrics"]
            ]
        if "baselines" in config:
            result["baselines"] = [
                EvaluationFactory.create_baseline(b) if isinstance(b, str) else b
                for b in config["baselines"]
            ]
        result["engine"] = EvaluationFactory.create_engine(config)
        return result

    @staticmethod
    def list_available_metrics() -> list[str]:
        return MetricRegistry.list()

    @staticmethod
    def list_available_baselines() -> list[str]:
        return BaselineRegistry.list()

    @staticmethod
    def list_available_ablations() -> list[str]:
        return AblationRegistry.list()

    @staticmethod
    def list_available_experiments() -> list[str]:
        return ExperimentRegistry.list()
