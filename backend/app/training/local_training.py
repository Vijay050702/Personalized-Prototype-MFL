from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.training.trainer import Trainer


class LocalTraining:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: torch.device | None = None,
        max_grad_norm: float = 0.0,
    ) -> None:
        self._trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        self._training_history: list[dict[str, Any]] = []

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 1,
    ) -> dict[str, Any]:
        results = self._trainer.train_local(dataloader, epochs=epochs)
        self._training_history.append(results)
        return results

    def validate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        return self._trainer.validate(dataloader)

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self._trainer.predict(dataloader)

    def get_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        embedding_layer: nn.Module | None = None,
    ) -> dict[int, list[torch.Tensor]]:
        self._trainer.model.eval()
        embeddings_by_class: dict[int, list[torch.Tensor]] = {}

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self._trainer._prepare_batch(batch)
                if embedding_layer is not None:
                    emb = embedding_layer(inputs)
                else:
                    intermediate = self._extract_embeddings(inputs)
                    emb = intermediate
                for i in range(targets.size(0)):
                    label = int(targets[i].item())
                    if label not in embeddings_by_class:
                        embeddings_by_class[label] = []
                    embeddings_by_class[label].append(emb[i].cpu())

        return embeddings_by_class

    def _extract_embeddings(
        self,
        inputs: Any,
    ) -> torch.Tensor:
        model = self._trainer.model
        if hasattr(model, "encode"):
            return model.encode(inputs)
        if hasattr(model, "get_embeddings"):
            return model.get_embeddings(inputs)
        features = {}
        for name, module in model.named_children():
            if name != "classifier":
                try:
                    features[name] = module(inputs)
                except Exception:
                    continue
        if features:
            return torch.cat([f.view(f.size(0), -1) for f in features.values()], dim=1)
        return model(inputs)

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    @property
    def model(self) -> nn.Module:
        return self._trainer.model

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._training_history)

    def clear_history(self) -> None:
        self._training_history.clear()
