from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.evaluation.registry import BaselineRegistry
from app.federated.aggregation import PrototypeAggregator
from app.federated.models import ClientPrototypePackage
from app.training.local_training import LocalTraining
from app.training.optimizer import FedProxOptimizer, OptimizerFactory
from app.training.trainer import Trainer
from app.training.utils import (
    compute_accuracy,
    flatten_model_state,
    unflatten_model_state,
)


class Baseline(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._round_metrics: list[dict[str, Any]] = []

    @abstractmethod
    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]: ...

    @abstractmethod
    def name(self) -> str: ...

    @property
    def round_metrics(self) -> list[dict[str, Any]]:
        return list(self._round_metrics)

    def reset(self) -> None:
        self._round_metrics.clear()


class FedAvgBaseline(Baseline):
    def name(self) -> str:
        return "fedavg"

    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]:
        num_clients = len(clients)
        client_weights = []
        total_samples = 0
        sample_counts = []

        for i, client_cfg in enumerate(clients):
            model = copy.deepcopy(server_model)
            opt = OptimizerFactory.create(
                model,
                optimizer_type=self._config.get("optimizer", "sgd"),
                lr=client_cfg.get("lr", 0.01),
            )
            loss_fn = nn.CrossEntropyLoss()
            trainer = Trainer(model, loss_fn, opt)
            loader = dataloaders[i] if i < len(dataloaders) else dataloaders[0]
            metrics = trainer.train_one_epoch(loader, epoch=0)

            n_samples = len(loader.dataset)
            total_samples += n_samples
            sample_counts.append(n_samples)
            client_weights.append(
                (n_samples, {k: v.clone() for k, v in model.state_dict().items()})
            )

        aggregated = {}
        for key in client_weights[0][1]:
            stacked = torch.stack([w * sd[key].float() for w, sd in client_weights])
            aggregated[key] = (stacked.sum(dim=0) / total_samples).to(
                next(server_model.parameters()).dtype
            )
        server_model.load_state_dict(aggregated)

        acc = self._evaluate(server_model, dataloaders)
        metric = {
            "round_id": round_id,
            "accuracy": acc,
            "loss": metrics.get("loss", 0.0),
        }
        self._round_metrics.append(metric)
        return metric

    def _evaluate(self, model: nn.Module, dataloaders: list[DataLoader]) -> float:
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for loader in dataloaders:
                for data, target in loader:
                    output = model(data)
                    all_preds.append(output)
                    all_targets.append(target)
        if not all_preds:
            return 0.0
        preds_t = torch.cat(all_preds, dim=0)
        targets_t = torch.cat(all_targets, dim=0)
        return float(compute_accuracy(preds_t, targets_t).item())


class FedProxBaseline(Baseline):
    def name(self) -> str:
        return "fedprox"

    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]:
        mu = self._config.get("fedprox_mu", 0.01)
        num_clients = len(clients)
        total_samples = 0
        client_weights = []
        global_state = {k: v.clone() for k, v in server_model.state_dict().items()}

        for i, client_cfg in enumerate(clients):
            model = copy.deepcopy(server_model)
            opt = FedProxOptimizer(
                model.parameters(),
                lr=client_cfg.get("lr", 0.01),
                mu=mu,
            )
            OptimizerFactory.set_global_params(opt, model)
            loss_fn = nn.CrossEntropyLoss()
            trainer = Trainer(model, loss_fn, opt)
            loader = dataloaders[i] if i < len(dataloaders) else dataloaders[0]
            metrics = trainer.train_one_epoch(loader, epoch=0)

            n_samples = len(loader.dataset)
            total_samples += n_samples
            client_weights.append(
                (n_samples, {k: v.clone() for k, v in model.state_dict().items()})
            )

        aggregated = {}
        for key in client_weights[0][1]:
            stacked = torch.stack([w * sd[key].float() for w, sd in client_weights])
            aggregated[key] = (stacked.sum(dim=0) / total_samples).to(
                next(server_model.parameters()).dtype
            )
        server_model.load_state_dict(aggregated)

        acc = self._evaluate(server_model, dataloaders)
        metric = {
            "round_id": round_id,
            "accuracy": acc,
            "loss": metrics.get("loss", 0.0),
        }
        self._round_metrics.append(metric)
        return metric

    def _evaluate(self, model: nn.Module, dataloaders: list[DataLoader]) -> float:
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for loader in dataloaders:
                for data, target in loader:
                    output = model(data)
                    all_preds.append(output)
                    all_targets.append(target)
        if not all_preds:
            return 0.0
        preds_t = torch.cat(all_preds, dim=0)
        targets_t = torch.cat(all_targets, dim=0)
        return float(compute_accuracy(preds_t, targets_t).item())


class SCAFFOLDBaseline(Baseline):
    def name(self) -> str:
        return "scaffold"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._server_control = None
        self._client_controls: dict[int, Any] = {}

    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]:
        num_clients = len(clients)
        lr = clients[0].get("lr", 0.01) if clients else 0.01

        if self._server_control is None:
            self._server_control = {
                k: torch.zeros_like(v) for k, v in server_model.state_dict().items()
            }

        global_state = {k: v.clone() for k, v in server_model.state_dict().items()}
        total_samples = 0
        client_weights = []

        for i in range(num_clients):
            model = copy.deepcopy(server_model)
            opt = OptimizerFactory.create(model, optimizer_type="sgd", lr=lr)
            loss_fn = nn.CrossEntropyLoss()
            trainer = Trainer(model, loss_fn, opt)
            loader = dataloaders[i] if i < len(dataloaders) else dataloaders[0]

            if i not in self._client_controls:
                self._client_controls[i] = {
                    k: torch.zeros_like(v) for k, v in server_model.state_dict().items()
                }

            model.train()
            for data, target in loader:
                opt.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                for key, param in model.named_parameters():
                    if key in self._server_control and key in self._client_controls[i]:
                        if param.grad is not None:
                            param.grad.data -= (
                                self._client_controls[i][key].data
                                - self._server_control[key].data
                            )
                opt.step()

            n_samples = len(loader.dataset)
            total_samples += n_samples
            client_weights.append(
                (n_samples, {k: v.clone() for k, v in model.state_dict().items()})
            )

            with torch.no_grad():
                for key in self._client_controls[i]:
                    delta_c = -self._client_controls[i][key] + (
                        (global_state[key] - model.state_dict()[key])
                        / (lr * len(loader) if len(loader) > 0 else 1.0)
                    )
                    self._client_controls[i][key] = (
                        self._client_controls[i][key] + delta_c
                    )

        aggregated = {}
        for key in client_weights[0][1]:
            stacked = torch.stack([w * sd[key].float() for w, sd in client_weights])
            aggregated[key] = (stacked.sum(dim=0) / total_samples).to(
                next(server_model.parameters()).dtype
            )
        server_model.load_state_dict(aggregated)

        with torch.no_grad():
            for key in self._server_control:
                self._server_control[key] = self._server_control[key] + (
                    sum(
                        self._client_controls[i][key]
                        - self._client_controls[i].get(key, 0)
                        for i in range(num_clients)
                        if i in self._client_controls
                    )
                    / num_clients
                )

        acc = self._evaluate(server_model, dataloaders)
        metric = {"round_id": round_id, "accuracy": acc}
        self._round_metrics.append(metric)
        return metric

    def _evaluate(self, model: nn.Module, dataloaders: list[DataLoader]) -> float:
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for loader in dataloaders:
                for data, target in loader:
                    output = model(data)
                    all_preds.append(output)
                    all_targets.append(target)
        if not all_preds:
            return 0.0
        preds_t = torch.cat(all_preds, dim=0)
        targets_t = torch.cat(all_targets, dim=0)
        return float(compute_accuracy(preds_t, targets_t).item())

    def reset(self) -> None:
        super().reset()
        self._server_control = None
        self._client_controls.clear()


class PrototypeOnlyBaseline(Baseline):
    def name(self) -> str:
        return "prototype_only"

    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]:
        metric = {"round_id": round_id, "accuracy": 0.0, "loss": 0.0}
        self._round_metrics.append(metric)
        return metric


class WithoutPersonalizationBaseline(Baseline):
    def name(self) -> str:
        return "without_personalization"

    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]:
        metric = {"round_id": round_id, "accuracy": 0.0, "loss": 0.0}
        self._round_metrics.append(metric)
        return metric


class WithoutKnowledgeTransferBaseline(Baseline):
    def name(self) -> str:
        return "without_knowledge_transfer"

    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]:
        metric = {"round_id": round_id, "accuracy": 0.0, "loss": 0.0}
        self._round_metrics.append(metric)
        return metric


class WithoutAdaptiveAggregationBaseline(Baseline):
    def name(self) -> str:
        return "without_adaptive_aggregation"

    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]:
        metric = {"round_id": round_id, "accuracy": 0.0, "loss": 0.0}
        self._round_metrics.append(metric)
        return metric


class FullPPMFLLBaseline(Baseline):
    def name(self) -> str:
        return "full_pp_mfl"

    def train_round(
        self,
        round_id: int,
        clients: list[dict[str, Any]],
        server_model: nn.Module,
        dataloaders: list[DataLoader],
    ) -> dict[str, Any]:
        metric = {"round_id": round_id, "accuracy": 0.0, "loss": 0.0}
        self._round_metrics.append(metric)
        return metric


class BaselineFactory:
    _baseline_map: dict[str, type[Baseline]] = {
        "fedavg": FedAvgBaseline,
        "fedprox": FedProxBaseline,
        "scaffold": SCAFFOLDBaseline,
        "prototype_only": PrototypeOnlyBaseline,
        "without_personalization": WithoutPersonalizationBaseline,
        "without_knowledge_transfer": WithoutKnowledgeTransferBaseline,
        "without_adaptive_aggregation": WithoutAdaptiveAggregationBaseline,
        "full_pp_mfl": FullPPMFLLBaseline,
    }

    @classmethod
    def create(cls, name: str, config: dict[str, Any]) -> Baseline:
        name = name.lower().replace("-", "_")
        if name in cls._baseline_map:
            return cls._baseline_map[name](config)
        baseline_cls = BaselineRegistry.get(name)
        return baseline_cls(config)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._baseline_map.keys()) + BaselineRegistry.list()


BaselineRegistry.register("fedavg", FedAvgBaseline)
BaselineRegistry.register("fedprox", FedProxBaseline)
BaselineRegistry.register("scaffold", SCAFFOLDBaseline)
BaselineRegistry.register("prototype_only", PrototypeOnlyBaseline)
BaselineRegistry.register("without_personalization", WithoutPersonalizationBaseline)
BaselineRegistry.register(
    "without_knowledge_transfer", WithoutKnowledgeTransferBaseline
)
BaselineRegistry.register(
    "without_adaptive_aggregation", WithoutAdaptiveAggregationBaseline
)
BaselineRegistry.register("full_pp_mfl", FullPPMFLLBaseline)
