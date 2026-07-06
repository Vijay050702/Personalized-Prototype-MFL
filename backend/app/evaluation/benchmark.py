from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.logging import logger
from app.evaluation.baselines import BaselineFactory
from app.evaluation.evaluator import EvaluationEngine
from app.evaluation.experiment_runner import ExperimentRunner
from app.evaluation.metrics import ClassificationMetrics, TrainingMetrics


class Benchmark:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or {}
        self._engine = EvaluationEngine(config=self._config)
        self._results: dict[str, Any] = {}
        self._start_time: float = time.time()

    @property
    def engine(self) -> EvaluationEngine:
        return self._engine

    @property
    def results(self) -> dict[str, Any]:
        return dict(self._results)

    def benchmark_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str = "model",
        device: torch.device | None = None,
        loss_fn: nn.Module | None = None,
    ) -> dict[str, float]:
        device = device or torch.device("cpu")
        logger.info(f"Benchmarking model: {model_name}")

        start = time.time()
        metrics = self._engine.evaluate_training(model, dataloader, loss_fn, device)
        elapsed = time.time() - start

        sample_input = self._get_sample_input(dataloader, device)
        if sample_input is not None:
            inference_ms = TrainingMetrics.inference_time(model, sample_input)
        else:
            inference_ms = 0.0

        num_params = sum(p.numel() for p in model.parameters())
        model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        result = {
            **metrics,
            "model_name": model_name,
            "benchmark_time": elapsed,
            "inference_time_ms": inference_ms,
            "num_parameters": float(num_params),
            "model_size_bytes": float(model_size_bytes),
            "model_size_mb": float(model_size_bytes / (1024 * 1024)),
        }
        self._results[model_name] = result
        return result

    def benchmark_baseline(
        self,
        baseline_name: str,
        num_rounds: int = 10,
        server_model: nn.Module | None = None,
        dataloaders: list[DataLoader] | None = None,
    ) -> dict[str, Any]:
        logger.info(f"Benchmarking baseline: {baseline_name}")
        baseline = BaselineFactory.create(baseline_name, self._config)
        runner = ExperimentRunner(config=self._config)
        metrics = runner.run_baselines(
            [baseline_name], self._config, num_rounds, server_model, dataloaders
        )
        return metrics.get(baseline_name, [])

    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        batch_size: int = 1,
        repetitions: int = 100,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        device = device or torch.device("cpu")
        model.to(device)
        model.eval()
        dummy_input = torch.randn(batch_size, *input_shape, device=device)

        starter = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        ender = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )

        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)

        if starter is not None and ender is not None:
            starter.record()
            for _ in range(repetitions):
                model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            avg_ms = starter.elapsed_time(ender) / repetitions
        else:
            total = 0.0
            with torch.no_grad():
                for _ in range(repetitions):
                    t0 = time.time()
                    model(dummy_input)
                    total += time.time() - t0
            avg_ms = (total / repetitions) * 1000.0

        throughput = (
            float(batch_size / (avg_ms / 1000.0)) if avg_ms > 0 else float("inf")
        )
        return {
            "batch_size": float(batch_size),
            "input_shape": str(input_shape),
            "avg_inference_time_ms": avg_ms,
            "throughput_items_per_sec": throughput,
            "repetitions": float(repetitions),
        }

    def benchmark_training_speed(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        device = device or torch.device("cpu")
        model.to(device)
        model.train()

        start = time.time()
        total_samples = 0
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                total_samples += data.size(0)
        elapsed = time.time() - start

        return {
            "epochs": float(epochs),
            "total_samples": float(total_samples),
            "total_time_seconds": elapsed,
            "samples_per_second": float(total_samples / elapsed)
            if elapsed > 0
            else 0.0,
            "seconds_per_epoch": float(elapsed / epochs),
        }

    def _get_sample_input(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor | None:
        try:
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                return batch[0].to(device)
            if isinstance(batch, dict):
                for key in ("data", "input", "inputs"):
                    if key in batch:
                        return batch[key].to(device)
        except (StopIteration, IndexError, KeyError):
            pass
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "num_benchmarks": len(self._results),
            "benchmarked_models": list(self._results.keys()),
            "total_benchmark_time": time.time() - self._start_time,
            "results": self._results,
        }

    def clear(self) -> None:
        self._results.clear()
        self._engine.clear()
        self._start_time = time.time()
