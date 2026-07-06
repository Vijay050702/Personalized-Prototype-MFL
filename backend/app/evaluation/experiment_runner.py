from __future__ import annotations

import copy
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.logging import logger
from app.evaluation.ablation import AblationStudy
from app.evaluation.baselines import Baseline, BaselineFactory
from app.evaluation.evaluator import EvaluationEngine
from app.evaluation.metrics import ClassificationMetrics
from app.evaluation.registry import ExperimentRegistry
from app.evaluation.statistical_analysis import StatisticalAnalysis
from app.training.experiment import Experiment as TrainingExperiment
from app.training.trainer import Trainer
from app.training.utils import compute_accuracy


class ExperimentRunner:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        output_dir: str = "experiments",
    ) -> None:
        self._config = config or {}
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._results: dict[str, Any] = {}
        self._engine = EvaluationEngine(config=self._config)

    @property
    def engine(self) -> EvaluationEngine:
        return self._engine

    @property
    def results(self) -> dict[str, Any]:
        return dict(self._results)

    def run_single(
        self,
        experiment_config: dict[str, Any],
        experiment_id: str | None = None,
    ) -> dict[str, float]:
        exp_id = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"
        logger.info(f"Running single experiment: {exp_id}")

        exp = TrainingExperiment(experiment_id=exp_id, config=experiment_config)
        exp.initialize()

        total_rounds = experiment_config.get("rounds", 10)
        eval_frequency = experiment_config.get("evaluation_frequency", 1)
        round_metrics: list[dict[str, float]] = []

        for round_id in range(1, total_rounds + 1):
            result = exp.coordinator._round_manager.run_round(round_id=round_id)
            metrics = {
                "round_id": float(round_id),
                **result.get("metrics", {}),
            }
            round_metrics.append(metrics)

            if round_id % eval_frequency == 0:
                logger.info(
                    f"Round {round_id}/{total_rounds}: acc={metrics.get('accuracy', 0.0):.4f}"
                )

        final_metrics = round_metrics[-1] if round_metrics else {}
        final_metrics["num_rounds"] = float(total_rounds)
        final_metrics["experiment_id"] = exp_id

        ExperimentRegistry.register(exp_id, experiment_config)
        self._results[exp_id] = {
            "config": experiment_config,
            "round_metrics": round_metrics,
            "final_metrics": final_metrics,
        }
        return final_metrics

    def run_batch(
        self,
        experiments: list[dict[str, Any]],
        parallel: bool = False,
    ) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for i, exp_config in enumerate(experiments):
            exp_id = exp_config.get("experiment_id", f"batch_{i}")
            try:
                final = self.run_single(exp_config, experiment_id=exp_id)
                results[exp_id] = final
            except Exception as e:
                logger.error(f"Experiment {exp_id} failed: {e}")
                results[exp_id] = {"error": 1.0, "message": str(e)}
        return results

    def run_with_multiple_datasets(
        self,
        base_config: dict[str, Any],
        datasets: list[str],
    ) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for dataset_name in datasets:
            config = copy.deepcopy(base_config)
            config.setdefault("dataset", {})["name"] = dataset_name
            exp_id = f"dataset_{dataset_name}"
            try:
                final = self.run_single(config, experiment_id=exp_id)
                results[exp_id] = final
            except Exception as e:
                logger.error(f"Dataset {dataset_name} failed: {e}")
                results[exp_id] = {"error": 1.0}
        return results

    def run_with_multiple_seeds(
        self,
        base_config: dict[str, Any],
        seeds: list[int],
    ) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for seed in seeds:
            torch.manual_seed(seed)
            config = copy.deepcopy(base_config)
            config["seed"] = seed
            exp_id = f"seed_{seed}"
            try:
                final = self.run_single(config, experiment_id=exp_id)
                results[exp_id] = final
            except Exception as e:
                logger.error(f"Seed {seed} failed: {e}")
                results[exp_id] = {"error": 1.0}
        return results

    def run_ablation(
        self,
        base_config: dict[str, Any],
        ablations: list[str] | None = None,
    ) -> dict[str, Any]:
        study = AblationStudy(base_config)
        study.run(self, ablations=ablations)
        return study.to_dict()

    def run_baselines(
        self,
        baselines: list[str],
        base_config: dict[str, Any],
        num_rounds: int = 10,
        server_model: nn.Module | None = None,
        dataloaders: list[DataLoader] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        results: dict[str, list[dict[str, Any]]] = {}
        for name in baselines:
            try:
                baseline = BaselineFactory.create(name, base_config)
                results[name] = self._run_baseline(
                    baseline, num_rounds, server_model, dataloaders
                )
            except Exception as e:
                logger.error(f"Baseline {name} failed: {e}")
                results[name] = [{"error": 1.0, "message": str(e)}]
        return results

    def _run_baseline(
        self,
        baseline: Baseline,
        num_rounds: int,
        server_model: nn.Module | None,
        dataloaders: list[DataLoader] | None,
    ) -> list[dict[str, Any]]:
        if server_model is None:
            server_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        if dataloaders is None:
            data = torch.randn(50, 10)
            labels = torch.randint(0, 5, (50,))
            dataset = torch.utils.data.TensorDataset(data, labels)
            dataloaders = [DataLoader(dataset, batch_size=10)]

        clients = [{"lr": 0.01} for _ in range(len(dataloaders))]
        for r in range(1, num_rounds + 1):
            try:
                baseline.train_round(r, clients, server_model, dataloaders)
            except Exception as e:
                logger.error(f"Baseline round {r} failed: {e}")
        return baseline.round_metrics

    def run(
        self,
        config: dict[str, Any],
    ) -> dict[str, float]:
        return self.run_single(config)

    def resume_experiment(
        self,
        experiment_id: str,
        checkpoint_path: str | None = None,
    ) -> dict[str, float]:
        if experiment_id not in ExperimentRegistry.list():
            config = self._config
        else:
            config = ExperimentRegistry.get(experiment_id)
        exp = TrainingExperiment(experiment_id=experiment_id, config=config)
        state = exp.resume(checkpoint_path=checkpoint_path)
        if state is None:
            return {"completed": 0.0}
        return {"completed": 1.0, "final_round": float(state.current_round)}

    def save_results(
        self,
        path: str | None = None,
    ) -> str:
        save_path = path or str(
            self._output_dir / f"results_{uuid.uuid4().hex[:8]}.json"
        )
        with open(save_path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)
        logger.info(f"Results saved to {save_path}")
        return save_path

    def load_results(self, path: str) -> dict[str, Any]:
        with open(path) as f:
            data = json.load(f)
        self._results.update(data)
        return data

    def get_best_experiment(self, metric: str = "accuracy") -> tuple[str, float]:
        best_id = ""
        best_val = float("-inf")
        for exp_id, result in self._results.items():
            final = result.get("final_metrics", {})
            val = final.get(metric, float("-inf"))
            if isinstance(val, (int, float)) and val > best_val:
                best_val = val
                best_id = exp_id
        return best_id, best_val

    def summarize_experiments(self) -> dict[str, Any]:
        if not self._results:
            return {"num_experiments": 0}
        summaries: dict[str, Any] = {
            "num_experiments": len(self._results),
            "experiment_ids": list(self._results.keys()),
        }
        metric_values: dict[str, list[float]] = {}
        for exp_id, result in self._results.items():
            final = result.get("final_metrics", {})
            for k, v in final.items():
                if isinstance(v, (int, float)):
                    metric_values.setdefault(k, []).append(float(v))
        for metric, values in metric_values.items():
            if values:
                summaries[f"{metric}_mean"] = float(sum(values) / len(values))
                summaries[f"{metric}_std"] = (
                    float(
                        (
                            sum((v - sum(values) / len(values)) ** 2 for v in values)
                            / len(values)
                        )
                        ** 0.5
                    )
                    if len(values) > 1
                    else 0.0
                )
        return summaries

    def clear(self) -> None:
        self._results.clear()
        self._engine.clear()
