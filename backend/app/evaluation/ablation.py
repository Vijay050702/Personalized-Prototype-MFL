from __future__ import annotations

import copy
from typing import Any

from app.core.logging import logger
from app.evaluation.registry import AblationRegistry
from app.evaluation.statistical_analysis import StatisticalAnalysis


class AblationStudy:
    def __init__(self, base_config: dict[str, Any]) -> None:
        self._base_config = copy.deepcopy(base_config)
        self._results: dict[str, dict[str, float]] = {}
        self._configs: dict[str, dict[str, Any]] = {}

    def _modify_config(self, config: dict[str, Any], ablation: str) -> dict[str, Any]:
        modified = copy.deepcopy(config)
        if ablation == "without_prototypes":
            modified.setdefault("ablation", {})["disable_prototypes"] = True
        elif ablation == "without_aggregation":
            modified.setdefault("ablation", {})["disable_aggregation"] = True
        elif ablation == "without_knowledge_transfer":
            modified.setdefault("ablation", {})["disable_knowledge_transfer"] = True
        elif ablation == "without_personalization":
            modified.setdefault("ablation", {})["disable_personalization"] = True
        elif ablation == "without_adaptive_weighting":
            modified.setdefault("ablation", {})["disable_adaptive_weighting"] = True
        elif ablation == "without_prototype_memory":
            modified.setdefault("ablation", {})["disable_prototype_memory"] = True
        return modified

    def run(
        self,
        runner: Any,
        ablations: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        if ablations is None:
            ablations = self._get_default_ablations()
        self._results["full"] = self._run_single(runner, self._base_config)
        self._configs["full"] = copy.deepcopy(self._base_config)
        logger.info(f"Ablation 'full': {self._results['full']}")

        for ablation in ablations:
            config = self._modify_config(self._base_config, ablation)
            self._configs[ablation] = config
            try:
                self._results[ablation] = self._run_single(runner, config)
                logger.info(f"Ablation '{ablation}': {self._results[ablation]}")
            except Exception as e:
                logger.error(f"Ablation '{ablation}' failed: {e}")
                self._results[ablation] = {"error": 1.0}
        return self._results

    def _run_single(self, runner: Any, config: dict[str, Any]) -> dict[str, float]:
        if hasattr(runner, "run_single"):
            return runner.run_single(config)
        if hasattr(runner, "run"):
            result = runner.run(config)
            if isinstance(result, dict):
                return {
                    k: float(v)
                    for k, v in result.items()
                    if isinstance(v, (int, float))
                }
        return {"completed": 1.0}

    def _get_default_ablations(self) -> list[str]:
        return [
            "without_prototypes",
            "without_aggregation",
            "without_knowledge_transfer",
            "without_personalization",
            "without_adaptive_weighting",
            "without_prototype_memory",
        ]

    def comparison_table(self) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = {}
        all_metrics: set[str] = set()
        for ablation_name, metrics in self._results.items():
            all_metrics.update(metrics.keys())
        sorted_metrics = sorted(all_metrics)

        for ablation_name, metrics in self._results.items():
            row: dict[str, Any] = {}
            for metric in sorted_metrics:
                if metric == "error":
                    continue
                full_val = self._results.get("full", {}).get(metric, 0.0)
                ablation_val = metrics.get(metric, 0.0)
                row[metric] = ablation_val
                if isinstance(ablation_val, (int, float)) and isinstance(
                    full_val, (int, float)
                ):
                    if full_val != 0:
                        row[f"{metric}_degradation"] = float(full_val - ablation_val)
                        row[f"{metric}_relative"] = float(
                            (full_val - ablation_val) / abs(full_val)
                        )
                    else:
                        row[f"{metric}_degradation"] = 0.0
                        row[f"{metric}_relative"] = 0.0
            table[ablation_name] = row
        return table

    @property
    def results(self) -> dict[str, dict[str, float]]:
        return dict(self._results)

    def summary(self) -> dict[str, Any]:
        full = self._results.get("full", {})
        comparison = self.comparison_table()
        worst_degradation: dict[str, tuple[str, float]] = {}
        for metric in full:
            if not isinstance(full[metric], (int, float)):
                continue
            worst_ablation = ""
            worst_val = 0.0
            for ablation_name, row in comparison.items():
                if ablation_name == "full":
                    continue
                deg = row.get(f"{metric}_degradation", 0.0)
                if isinstance(deg, (int, float)) and deg > worst_val:
                    worst_val = deg
                    worst_ablation = ablation_name
            if worst_ablation:
                worst_degradation[metric] = (worst_ablation, worst_val)
        return {
            "num_ablations": len(self._results),
            "ablations_run": list(self._results.keys()),
            "full_metrics": full,
            "worst_degradation": worst_degradation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": self._results,
            "comparison": self.comparison_table(),
            "summary": self.summary(),
        }


def without_prototypes(config: dict[str, Any]) -> dict[str, Any]:
    modified = copy.deepcopy(config)
    modified.setdefault("ablation", {})["disable_prototypes"] = True
    return modified


def without_aggregation(config: dict[str, Any]) -> dict[str, Any]:
    modified = copy.deepcopy(config)
    modified.setdefault("ablation", {})["disable_aggregation"] = True
    return modified


def without_knowledge_transfer(config: dict[str, Any]) -> dict[str, Any]:
    modified = copy.deepcopy(config)
    modified.setdefault("ablation", {})["disable_knowledge_transfer"] = True
    return modified


def without_personalization(config: dict[str, Any]) -> dict[str, Any]:
    modified = copy.deepcopy(config)
    modified.setdefault("ablation", {})["disable_personalization"] = True
    return modified


def without_adaptive_weighting(config: dict[str, Any]) -> dict[str, Any]:
    modified = copy.deepcopy(config)
    modified.setdefault("ablation", {})["disable_adaptive_weighting"] = True
    return modified


def without_prototype_memory(config: dict[str, Any]) -> dict[str, Any]:
    modified = copy.deepcopy(config)
    modified.setdefault("ablation", {})["disable_prototype_memory"] = True
    return modified


AblationRegistry.register("without_prototypes", without_prototypes)
AblationRegistry.register("without_aggregation", without_aggregation)
AblationRegistry.register("without_knowledge_transfer", without_knowledge_transfer)
AblationRegistry.register("without_personalization", without_personalization)
AblationRegistry.register("without_adaptive_weighting", without_adaptive_weighting)
AblationRegistry.register("without_prototype_memory", without_prototype_memory)
