from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.logging import logger
from app.datasets.base import BaseDatasetAdapter, DatasetLoadResult
from app.datasets.dataset_factory import DatasetFactory
from app.federated.factory import FederatedFactory
from app.knowledge_transfer.factory import TransferFactory
from app.models.factory import ModelFactory
from app.personalization.factory import PersonalizationFactory
from app.training.callbacks import EarlyStopping
from app.training.checkpoint import CheckpointManager
from app.training.client import Client
from app.training.coordinator import Coordinator
from app.training.events import EventDispatcher
from app.training.hooks import HookManager
from app.training.local_training import LocalTraining
from app.training.logger import TrainingLogger
from app.training.monitor import ResourceMonitor
from app.training.optimizer import OptimizerFactory
from app.training.scheduler import SchedulerFactory
from app.training.server import Server
from app.training.state import TrainingState


class Experiment:
    def __init__(
        self,
        experiment_id: str,
        config: dict[str, Any],
    ) -> None:
        self._experiment_id = experiment_id
        self._config = config
        self._logger = TrainingLogger(experiment_id=experiment_id)
        self._state: TrainingState | None = None
        self._coordinator: Coordinator | None = None
        self._checkpoint_manager: CheckpointManager | None = None

        errors = self._validate_config()
        if errors:
            raise ValueError(f"Config validation errors: {errors}")

    def _validate_config(self) -> list[str]:
        errors: list[str] = []
        required = ["rounds", "clients", "model", "dataset"]
        for key in required:
            if key not in self._config:
                errors.append(f"Missing required config: {key}")
        if "num_clients" not in self._config.get("clients", {}):
            errors.append("Missing clients.num_clients")
        return errors

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def state(self) -> TrainingState:
        if self._state is None:
            raise ValueError("Experiment not initialized")
        return self._state

    @property
    def coordinator(self) -> Coordinator:
        if self._coordinator is None:
            raise ValueError("Experiment not initialized")
        return self._coordinator

    @property
    def logger(self) -> TrainingLogger:
        return self._logger

    def initialize(self) -> None:
        self._state = TrainingState(
            experiment_id=self._experiment_id,
            config=self._config,
            total_rounds=self._config.get("rounds", 100),
        )

        mod_config = self._config.get("modalities", {"image": 64, "text": 64})
        mappings = self._config.get("mappings", [("image", "text")])
        num_classes = self._config.get("num_classes", 10)

        # Initialize models
        models = self._create_models(mod_config, num_classes)
        self._model = models.get("model")
        self._encoders = models.get("encoders", {})

        # Prepare datasets
        dataloaders = self._prepare_datasets()

        # Initialize repositories and components
        federated_components = self._create_federated()
        kt_components = self._create_knowledge_transfer(mod_config, mappings)
        personalization_components = self._create_personalization(mod_config, mappings)

        # Create server
        server = Server(
            server_id="server",
            federated_aggregator=federated_components.get("aggregator"),
            federated_repository=federated_components.get("repository"),
            inference_engine=kt_components.get("inference"),
            logger_instance=self._logger,
        )

        # Create clients
        clients = self._create_clients(
            model=self._model,
            num_clients=self._config.get("clients", {}).get("num_clients", 5),
            dataloaders=dataloaders,
            mod_config=mod_config,
            personalization_components=personalization_components,
            kt_components=kt_components,
        )

        # Checkpoint manager
        ckpt_dir = self._config.get("checkpoint_dir", "checkpoints")
        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=ckpt_dir,
            experiment_id=self._experiment_id,
        )

        # Early stopping
        early_stopping = None
        es_config = self._config.get("early_stopping", {})
        if es_config.get("enabled", False):
            early_stopping = EarlyStopping(
                patience=es_config.get("patience", 10),
                metric=es_config.get("metric", "accuracy"),
            )

        # Monitoring
        monitor = ResourceMonitor()

        # Coordinator
        self._coordinator = Coordinator(
            server=server,
            clients=clients,
            training_state=self._state,
            logger_instance=self._logger,
            monitor=monitor,
            checkpoint_manager=self._checkpoint_manager,
            early_stopping=early_stopping,
        )

        self._logger.log_experiment_start(self._config)
        logger.info(
            f"Experiment {self._experiment_id} initialized: "
            f"{len(clients)} clients, {self._state.total_rounds} rounds"
        )

    def _create_models(
        self,
        mod_config: dict[str, int],
        num_classes: int,
    ) -> dict[str, Any]:
        encoders = {}
        for mod, dim in mod_config.items():
            encoders[mod] = ModelFactory.create_encoder(modality=mod, embedding_dim=dim)
        total_dim = sum(mod_config.values())
        classifier = ModelFactory.create_classifier(
            input_dim=total_dim, num_classes=num_classes
        )
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            classifier,
        )
        return {"model": model, "encoders": encoders}

    def _prepare_datasets(self) -> dict[str, DataLoader]:
        dataset_name = self._config.get("dataset", {}).get("name", "generic")
        batch_size = self._config.get("dataset", {}).get("batch_size", 32)
        num_clients = self._config.get("clients", {}).get("num_clients", 5)

        dataloaders: dict[str, DataLoader] = {}
        try:
            adapter = DatasetFactory.create(dataset_name)
        except Exception:
            logger.warning(
                f"Dataset '{dataset_name}' not found, creating synthetic loaders"
            )
            for i in range(num_clients):
                dummy_data = torch.randn(
                    100, sum(self._config.get("modalities", {"image": 64}).values())
                )
                dummy_labels = torch.randint(
                    0,
                    self._config.get("num_classes", 10),
                    (100,),
                )
                dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
                loader = DataLoader(dataset, batch_size=batch_size)
                dataloaders[f"client_{i}"] = loader
            return dataloaders

        for i in range(num_clients):
            try:
                result = adapter.load(Path(f"datasets/{dataset_name}"), split="train")
                client_data = self._create_client_dataset(result, i, num_clients)
                loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
                dataloaders[f"client_{i}"] = loader
            except Exception as e:
                logger.warning(f"Failed to load dataset for client {i}: {e}")
                dummy_data = torch.randn(
                    100, sum(self._config.get("modalities", {"image": 64}).values())
                )
                dummy_labels = torch.randint(
                    0, self._config.get("num_classes", 10), (100,)
                )
                dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
                loader = DataLoader(dataset, batch_size=batch_size)
                dataloaders[f"client_{i}"] = loader

        return dataloaders

    def _create_client_dataset(
        self,
        result: DatasetLoadResult,
        client_idx: int,
        num_clients: int,
    ) -> torch.utils.data.Dataset:
        total = len(result)
        chunk_size = total // num_clients
        start = client_idx * chunk_size
        end = start + chunk_size if client_idx < num_clients - 1 else total

        data_tensors = []
        for i in range(start, end):
            item_data = result.data[i]
            if isinstance(item_data, dict):
                concat = torch.cat(
                    [
                        torch.tensor(v, dtype=torch.float32).view(-1)
                        for v in item_data.values()
                    ]
                )
                data_tensors.append(concat)
            else:
                data_tensors.append(
                    torch.tensor(item_data, dtype=torch.float32).view(-1)
                )

        labels_t = torch.tensor(result.labels[start:end].tolist(), dtype=torch.long)
        data_t = torch.stack(data_tensors)
        return torch.utils.data.TensorDataset(data_t, labels_t)

    def _create_federated(self) -> dict[str, Any]:
        fed_config = self._config.get("federated", {})
        aggregator = FederatedFactory.create_with_config(fed_config)
        return {
            "aggregator": aggregator,
            "repository": aggregator._repository,
        }

    def _create_knowledge_transfer(
        self,
        modalities: dict[str, int],
        mappings: list[tuple[str, str]],
    ) -> dict[str, Any]:
        kt_config = self._config.get("knowledge_transfer", {})
        return TransferFactory.create_from_config(
            {
                "modalities": modalities,
                "mappings": mappings,
                "loss": kt_config.get("loss", {"type": "info_nce"}),
            }
        )

    def _create_personalization(
        self,
        modalities: dict[str, int],
        mappings: list[tuple[str, str]],
    ) -> dict[str, Any]:
        p_config = self._config.get("personalization", {})
        return PersonalizationFactory.create_from_config(
            {
                "modalities": modalities,
                "mappings": mappings,
                "fusion": p_config.get("fusion", {}),
                "selector": p_config.get("selector", {}),
                "adaptation": p_config.get("adaptation", {}),
                "loss": p_config.get("loss", {}),
                "memory_capacity": p_config.get("memory_capacity", 1000),
            }
        )

    def _create_clients(
        self,
        model: nn.Module,
        num_clients: int,
        dataloaders: dict[str, DataLoader],
        mod_config: dict[str, int],
        personalization_components: dict[str, Any],
        kt_components: dict[str, Any],
    ) -> list[Client]:
        clients: list[Client] = []
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        opt_config = self._config.get("optimizer", {})
        sched_config = self._config.get("scheduler", {})

        for i in range(num_clients):
            client_id = f"client_{i}"

            client_model = self._clone_model(model, device)
            optimizer = OptimizerFactory.create(
                client_model,
                optimizer_type=opt_config.get("type", "adam"),
                lr=opt_config.get("lr", 1e-3),
                weight_decay=opt_config.get("weight_decay", 0.0),
            )
            scheduler = None
            if sched_config.get("type"):
                scheduler = SchedulerFactory.create(
                    optimizer,
                    scheduler_type=sched_config["type"],
                    **sched_config.get("kwargs", {}),
                )

            local_training = LocalTraining(
                model=client_model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

            proto_gen = personalization_components.get("prototype_generator")
            proto_selector = personalization_components.get("prototype_selector")
            fusion_engine = personalization_components.get("fusion_engine")
            adapt_engine = personalization_components.get("adaptation_engine")
            inference_engine = kt_components.get("inference")

            client = Client(
                client_id=client_id,
                model=client_model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                local_training=local_training,
                prototype_generator=proto_gen,
                prototype_selector=proto_selector,
                fusion_engine=fusion_engine,
                adaptation_engine=adapt_engine,
                inference_engine=inference_engine,
                logger_instance=self._logger,
                device=device,
            )

            if client_id in dataloaders:
                client.load_local_dataset(
                    dataloader=dataloaders[client_id],
                    modalities=set(mod_config.keys()),
                    all_modalities=set(mod_config.keys()),
                )

            clients.append(client)

        return clients

    def _clone_model(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> nn.Module:
        import copy

        new_model = copy.deepcopy(model)
        new_model.to(device)
        return new_model

    def run(self) -> TrainingState:
        if self._coordinator is None:
            raise ValueError("Experiment not initialized. Call initialize() first.")
        self._coordinator.run()
        return self._state

    def resume(
        self,
        checkpoint_path: str | None = None,
    ) -> TrainingState:
        if self._coordinator is None:
            self.initialize()

        ckpt_path = checkpoint_path or (
            str(self._checkpoint_manager._checkpoint_path("latest"))
            if self._checkpoint_manager
            else None
        )

        if ckpt_path is None or not os.path.exists(ckpt_path):
            logger.warning(f"No checkpoint found at {ckpt_path}, starting fresh")
            return self.run()

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        resume_round = checkpoint.get("round_id", 0)

        if self._state is not None:
            self._state.current_round = resume_round
            self._state.phase = "resumed"

        logger.info(
            f"Resuming experiment {self._experiment_id} from round {resume_round}"
        )
        return self.run()

    def cleanup(self) -> None:
        self._logger.clear()
        logger.info(f"Experiment {self._experiment_id} cleaned up")
