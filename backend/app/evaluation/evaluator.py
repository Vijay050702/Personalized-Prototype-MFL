from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.logging import logger
from app.evaluation.metrics import (
    ClassificationMetrics,
    CommunicationMetrics,
    KnowledgeTransferMetrics,
    PersonalizationMetrics as PersonalizationMetricComputer,
    PrototypeMetrics as PrototypeMetricComputer,
    TrainingMetrics,
)
from app.evaluation.registry import MetricRegistry


class EvaluationEngine:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or {}
        self._eval_history: list[dict[str, Any]] = []
        self._resource_usage: dict[str, list[float]] = {
            "gpu_usage": [],
            "cpu_usage": [],
            "memory_usage": [],
        }

    @property
    def eval_history(self) -> list[dict[str, Any]]:
        return list(self._eval_history)

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def evaluate_training(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        device = device or torch.device("cpu")
        model.to(device)
        model.eval()
        total_loss = 0.0
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        continue
                elif isinstance(batch, dict):
                    inputs = batch.get("data", batch.get("input", batch.get("inputs")))
                    targets = batch.get(
                        "label", batch.get("target", batch.get("labels"))
                    )
                    if inputs is None or targets is None:
                        continue
                else:
                    continue

                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

                if loss_fn is not None:
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item()
                num_batches += 1

        if not all_outputs:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }

        outputs_t = torch.cat(all_outputs, dim=0)
        targets_t = torch.cat(all_targets, dim=0)
        avg_loss = total_loss / max(num_batches, 1)

        clf_metrics = ClassificationMetrics.compute_all(outputs_t, targets_t)
        result = {"loss": avg_loss, **clf_metrics}
        return result

    def evaluate_validation(
        self,
        model: nn.Module,
        val_dataloader: DataLoader,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        return self.evaluate_training(model, val_dataloader, loss_fn, device)

    def evaluate_testing(
        self,
        model: nn.Module,
        test_dataloader: DataLoader,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        return self.evaluate_training(model, test_dataloader, loss_fn, device)

    def evaluate_single_client(
        self,
        client_id: str,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        metrics = self.evaluate_training(model, dataloader, loss_fn, device)
        metrics["client_id"] = client_id
        return metrics

    def evaluate_multi_client(
        self,
        client_models: dict[str, tuple[nn.Module, DataLoader]],
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for client_id, (model, loader) in client_models.items():
            results[client_id] = self.evaluate_single_client(
                client_id, model, loader, loss_fn, device
            )
        return results

    def evaluate_single_modality(
        self,
        model: nn.Module,
        modality_loader: DataLoader,
        modality_name: str = "unknown",
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        metrics = self.evaluate_training(model, modality_loader, loss_fn, device)
        metrics["modality"] = modality_name
        return metrics

    def evaluate_multi_modality(
        self,
        model: nn.Module,
        modality_loaders: dict[str, DataLoader],
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for mod_name, loader in modality_loaders.items():
            results[mod_name] = self.evaluate_single_modality(
                model, loader, mod_name, loss_fn, device
            )
        return results

    def evaluate_missing_modality(
        self,
        model: nn.Module,
        available_loaders: dict[str, DataLoader],
        missing_modalities: list[str],
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        combined_metrics: dict[str, float] = {
            "missing_modalities": float(len(missing_modalities))
        }
        for mod_name, loader in available_loaders.items():
            if mod_name not in missing_modalities:
                mod_metrics = self.evaluate_single_modality(
                    model, loader, mod_name, loss_fn, device
                )
                for k, v in mod_metrics.items():
                    if isinstance(v, (int, float)):
                        combined_metrics[f"{mod_name}_{k}"] = float(v)
        return combined_metrics

    def evaluate_personalized(
        self,
        personalized_models: dict[str, tuple[nn.Module, DataLoader]],
        global_model: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        gains: list[float] = []

        for client_id, (p_model, loader) in personalized_models.items():
            p_metrics = self.evaluate_single_client(
                client_id, p_model, loader, loss_fn, device
            )
            results[client_id] = p_metrics
            if global_model is not None:
                g_metrics = self.evaluate_training(
                    global_model, loader, loss_fn, device
                )
                gain = p_metrics.get("accuracy", 0.0) - g_metrics.get("accuracy", 0.0)
                gains.append(gain)
                results[client_id]["personalization_gain_vs_global"] = gain

        results["_summary"] = {
            "mean_accuracy": float(
                np.mean(
                    [
                        r.get("accuracy", 0.0)
                        for r in results.values()
                        if isinstance(r, dict)
                    ]
                )
            )
            if results
            else 0.0,
            "mean_personalization_gain": float(np.mean(gains)) if gains else 0.0,
        }
        return results

    def evaluate_prototypes(
        self,
        old_embeddings: list[torch.Tensor] | None = None,
        new_embeddings: list[torch.Tensor] | None = None,
        all_embeddings: list[torch.Tensor] | None = None,
        history: list[float] | None = None,
    ) -> dict[str, float]:
        return PrototypeMetricComputer.compute_all(
            old_embeddings=old_embeddings,
            new_embeddings=new_embeddings,
            all_embeddings=all_embeddings,
            history=history,
        )

    def evaluate_knowledge_transfer(
        self,
        source_embeddings: torch.Tensor | None = None,
        target_embeddings: torch.Tensor | None = None,
        predicted: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        valid_transfers: int = 0,
        total_attempts: int = 0,
        mod_a_embeddings: torch.Tensor | None = None,
        mod_b_embeddings: torch.Tensor | None = None,
    ) -> dict[str, float]:
        return KnowledgeTransferMetrics.compute_all(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
            predicted=predicted,
            targets=targets,
            valid_transfers=valid_transfers,
            total_attempts=total_attempts,
            mod_a_embeddings=mod_a_embeddings,
            mod_b_embeddings=mod_b_embeddings,
        )

    def evaluate_communication(
        self,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        duration_seconds: float = 1.0,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        return CommunicationMetrics.compute_all(
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            duration_seconds=duration_seconds,
            messages=messages,
        )

    def evaluate_personalization(
        self,
        personalized: torch.Tensor | None = None,
        global_prototype: torch.Tensor | None = None,
        pre_adaptation: torch.Tensor | None = None,
        post_adaptation: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        fusion_weights: dict[str, float] | None = None,
        confidences: list[float] | None = None,
        accuracies: list[float] | None = None,
    ) -> dict[str, float]:
        return PersonalizationMetricComputer.compute_all(
            personalized=personalized,
            global_prototype=global_prototype,
            pre_adaptation=pre_adaptation,
            post_adaptation=post_adaptation,
            targets=targets,
            fusion_weights=fusion_weights,
            confidences=confidences,
            accuracies=accuracies,
        )

    def evaluate_all(
        self,
        round_id: int,
        model: nn.Module | None = None,
        dataloader: DataLoader | None = None,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
        **extra_metrics: Any,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {"round_id": float(round_id)}

        if model is not None and dataloader is not None:
            clf_metrics = self.evaluate_training(model, dataloader, loss_fn, device)
            metrics.update(clf_metrics)

        for key, value in extra_metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        self._eval_history.append({"round_id": round_id, "metrics": dict(metrics)})
        return metrics

    def compute_metric(self, name: str, *args: Any, **kwargs: Any) -> float:
        try:
            return MetricRegistry.get(name)(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to compute metric '{name}': {e}")
            return 0.0

    def compute_training_time(
        self,
        start_time: float,
        end_time: float | None = None,
    ) -> float:
        return TrainingMetrics.training_time(start_time, end_time or time.time())

    def compute_inference_time(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        repetitions: int = 10,
    ) -> float:
        return TrainingMetrics.inference_time(model, inputs, repetitions)

    def record_resource_usage(
        self,
        gpu_usage: float = 0.0,
        cpu_usage: float = 0.0,
        memory_usage: float = 0.0,
    ) -> None:
        self._resource_usage["gpu_usage"].append(gpu_usage)
        self._resource_usage["cpu_usage"].append(cpu_usage)
        self._resource_usage["memory_usage"].append(memory_usage)

    def get_resource_summary(self) -> dict[str, float]:
        summary: dict[str, float] = {}
        for resource, values in self._resource_usage.items():
            if values:
                summary[f"avg_{resource}"] = float(np.mean(values))
                summary[f"max_{resource}"] = float(np.max(values))
                summary[f"min_{resource}"] = float(np.min(values))
            else:
                summary[f"avg_{resource}"] = 0.0
                summary[f"max_{resource}"] = 0.0
                summary[f"min_{resource}"] = 0.0
        return summary

    def summary(self) -> dict[str, Any]:
        if not self._eval_history:
            return {"num_evaluations": 0}

        all_metrics: dict[str, list[float]] = {}
        for entry in self._eval_history:
            for k, v in entry.get("metrics", {}).items():
                if isinstance(v, (int, float)):
                    all_metrics.setdefault(k, []).append(float(v))

        summary: dict[str, Any] = {
            "num_evaluations": len(self._eval_history),
            "rounds_evaluated": [e["round_id"] for e in self._eval_history],
        }
        for metric_name, values in all_metrics.items():
            if values:
                summary[f"{metric_name}_mean"] = float(np.mean(values))
                summary[f"{metric_name}_std"] = (
                    float(np.std(values)) if len(values) > 1 else 0.0
                )
                summary[f"{metric_name}_max"] = float(np.max(values))
                summary[f"{metric_name}_min"] = float(np.min(values))
                summary[f"{metric_name}_last"] = float(values[-1])

        return summary

    def clear(self) -> None:
        self._eval_history.clear()
        for key in self._resource_usage:
            self._resource_usage[key].clear()
