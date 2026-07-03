from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.training.utils import (
    clip_gradients,
    compute_accuracy,
    compute_grad_norm,
    to_device,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: torch.device | None = None,
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device or torch.device("cpu")
        self._max_grad_norm = max_grad_norm
        self._use_amp = use_amp
        self._scaler = torch.amp.GradScaler() if use_amp else None
        self._model.to(self._device)

    def train_one_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int = 0,
    ) -> dict[str, float]:
        self._model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in dataloader:
            inputs, targets = self._prepare_batch(batch)
            loss, acc = self._train_step(inputs, targets)
            total_loss += loss
            total_acc += acc
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)

        return {"loss": avg_loss, "accuracy": avg_acc, "epoch": epoch}

    def _prepare_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 3:
                inputs, targets, _ = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            if isinstance(inputs, dict):
                inputs = {k: to_device(v, self._device) for k, v in inputs.items()}
            else:
                inputs = to_device(inputs, self._device)
            targets = to_device(targets, self._device)
            return inputs, targets
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    def _train_step(self, inputs: Any, targets: torch.Tensor) -> tuple[float, float]:
        self._optimizer.zero_grad()

        if self._use_amp:
            with torch.amp.autocast(device_type=self._device.type):
                outputs = self._model(inputs)
                loss = self._loss_fn(outputs, targets)
            self._scaler.scale(loss).backward()
            if self._max_grad_norm > 0:
                self._scaler.unscale_(self._optimizer)
                clip_gradients(self._model, self._max_grad_norm)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            outputs = self._model(inputs)
            loss = self._loss_fn(outputs, targets)
            loss.backward()
            if self._max_grad_norm > 0:
                clip_gradients(self._model, self._max_grad_norm)
            self._optimizer.step()

        if self._scheduler is not None:
            if hasattr(self._scheduler, "step"):
                self._scheduler.step()

        acc = compute_accuracy(outputs, targets).item()
        return loss.item(), acc

    def validate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        self._model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self._prepare_batch(batch)
                outputs = self._model(inputs)
                loss = self._loss_fn(outputs, targets)
                acc = compute_accuracy(outputs, targets).item()
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)

        return {"loss": avg_loss, "accuracy": avg_acc}

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self._model.eval()
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self._prepare_batch(batch)
                outputs = self._model(inputs)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        return all_outputs, all_targets

    def train_local(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 1,
    ) -> dict[str, Any]:
        epoch_metrics: list[dict[str, float]] = []
        for epoch in range(epochs):
            metrics = self.train_one_epoch(dataloader, epoch=epoch)
            epoch_metrics.append(metrics)
        return {
            "epoch_metrics": epoch_metrics,
            "final_loss": epoch_metrics[-1]["loss"] if epoch_metrics else 0.0,
            "final_accuracy": epoch_metrics[-1]["accuracy"] if epoch_metrics else 0.0,
            "grad_norm": compute_grad_norm(self._model),
        }

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def device(self) -> torch.device:
        return self._device
