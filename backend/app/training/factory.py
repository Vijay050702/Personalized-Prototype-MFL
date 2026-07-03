from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.federated.factory import FederatedFactory
from app.knowledge_transfer.factory import TransferFactory
from app.models.factory import ModelFactory
from app.personalization.factory import PersonalizationFactory
from app.training.callbacks import EarlyStopping
from app.training.checkpoint import CheckpointManager
from app.training.client import Client
from app.training.coordinator import Coordinator
from app.training.evaluator import Evaluator
from app.training.events import EventDispatcher
from app.training.experiment import Experiment
from app.training.hooks import HookManager
from app.training.local_training import LocalTraining
from app.training.logger import TrainingLogger
from app.training.monitor import ResourceMonitor
from app.training.optimizer import OptimizerFactory
from app.training.registry import TrainingRegistry
from app.training.scheduler import SchedulerFactory
from app.training.server import Server
from app.training.state import TrainingState


class TrainingFactory:
    @staticmethod
    def create_experiment(
        experiment_id: str,
        config: dict[str, Any],
    ) -> Experiment:
        return Experiment(experiment_id=experiment_id, config=config)

    @staticmethod
    def create_coordinator(
        server: Server,
        clients: list[Client],
        training_state: TrainingState,
        event_dispatcher: EventDispatcher | None = None,
        hook_manager: HookManager | None = None,
        logger_instance: TrainingLogger | None = None,
        monitor: ResourceMonitor | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> Coordinator:
        return Coordinator(
            server=server,
            clients=clients,
            training_state=training_state,
            event_dispatcher=event_dispatcher,
            hook_manager=hook_manager,
            logger_instance=logger_instance,
            monitor=monitor,
            checkpoint_manager=checkpoint_manager,
            early_stopping=early_stopping,
        )

    @staticmethod
    def create_client(
        client_id: str,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | None = None,
        logger_instance: TrainingLogger | None = None,
    ) -> Client:
        local_training = LocalTraining(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        return Client(
            client_id=client_id,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            local_training=local_training,
            logger_instance=logger_instance,
            device=device,
        )

    @staticmethod
    def create_server(
        federated_config: dict[str, Any] | None = None,
        kt_config: dict[str, Any] | None = None,
        logger_instance: TrainingLogger | None = None,
    ) -> Server:
        federated = None
        inference_engine = None

        if federated_config:
            federated = FederatedFactory.create_with_config(federated_config)

        if kt_config:
            kt = TransferFactory.create_from_config(kt_config)
            inference_engine = kt.get("inference")

        return Server(
            federated_aggregator=federated,
            inference_engine=inference_engine,
            logger_instance=logger_instance,
        )

    @staticmethod
    def create_evaluator(
        personalization_metrics_config: dict[str, Any] | None = None,
        logger_instance: TrainingLogger | None = None,
    ) -> Evaluator:
        from app.personalization.metrics import PersonalizationMetrics

        p_metrics = None
        if personalization_metrics_config:
            p_metrics = PersonalizationMetrics(
                similarity_metric=personalization_metrics_config.get(
                    "similarity_metric", "cosine"
                )
            )
        return Evaluator(
            personalization_metrics=p_metrics,
            logger_instance=logger_instance,
        )

    @staticmethod
    def create_checkpoint_manager(
        checkpoint_dir: str = "checkpoints",
        experiment_id: str = "",
        max_checkpoints: int = 5,
    ) -> CheckpointManager:
        return CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            experiment_id=experiment_id,
            max_checkpoints=max_checkpoints,
        )

    @staticmethod
    def create_monitor() -> ResourceMonitor:
        return ResourceMonitor()

    @staticmethod
    def create_logger(
        experiment_id: str = "",
    ) -> TrainingLogger:
        return TrainingLogger(experiment_id=experiment_id)

    @staticmethod
    def create_event_dispatcher() -> EventDispatcher:
        return EventDispatcher()

    @staticmethod
    def create_hook_manager() -> HookManager:
        return HookManager()

    @staticmethod
    def create_from_config(
        experiment_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        logger_instance = TrainingFactory.create_logger(experiment_id=experiment_id)

        mod_config = config.get("modalities", {"image": 64, "text": 64})
        mappings = config.get("mappings", [("image", "text")])
        num_classes = config.get("num_classes", 10)

        total_dim = sum(mod_config.values())
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, num_classes),
        )

        federated_config = config.get("federated", {})
        kt_config = {
            "modalities": mod_config,
            "mappings": mappings,
            "loss": config.get("knowledge_transfer", {}).get(
                "loss", {"type": "info_nce"}
            ),
        }

        server = TrainingFactory.create_server(
            federated_config=federated_config,
            kt_config=kt_config,
            logger_instance=logger_instance,
        )

        opt_config = config.get("optimizer", {})
        sched_config = config.get("scheduler", {})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        p_components = PersonalizationFactory.create_from_config(
            {
                "modalities": mod_config,
                "mappings": mappings,
                "fusion": config.get("personalization", {}).get("fusion", {}),
                "selector": config.get("personalization", {}).get("selector", {}),
                "adaptation": config.get("personalization", {}).get("adaptation", {}),
                "loss": config.get("personalization", {}).get("loss", {}),
                "memory_capacity": config.get("personalization", {}).get(
                    "memory_capacity", 1000
                ),
            }
        )

        num_clients = config.get("clients", {}).get("num_clients", 5)
        clients: list[Client] = []
        for i in range(num_clients):
            client_id = f"client_{i}"
            client_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(total_dim, total_dim),
                nn.ReLU(),
                nn.Linear(total_dim, num_classes),
            ).to(device)

            optimizer = OptimizerFactory.create(
                client_model,
                optimizer_type=opt_config.get("type", "adam"),
                lr=opt_config.get("lr", 1e-3),
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
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

            client = Client(
                client_id=client_id,
                model=client_model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                local_training=local_training,
                prototype_selector=p_components.get("prototype_selector"),
                fusion_engine=p_components.get("fusion_engine"),
                adaptation_engine=p_components.get("adaptation_engine"),
                logger_instance=logger_instance,
                device=device,
            )

            dummy_data = torch.randn(100, total_dim)
            dummy_labels = torch.randint(0, num_classes, (100,))
            dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
            loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=32)
            client.load_local_dataset(
                dataloader=loader,
                modalities=set(mod_config.keys()),
                all_modalities=set(mod_config.keys()),
            )
            clients.append(client)

        state = TrainingState(
            experiment_id=experiment_id,
            config=config,
            total_rounds=config.get("rounds", 100),
        )

        checkpoint_manager = TrainingFactory.create_checkpoint_manager(
            checkpoint_dir=config.get("checkpoint_dir", "checkpoints"),
            experiment_id=experiment_id,
        )

        early_stopping = None
        es_config = config.get("early_stopping", {})
        if es_config.get("enabled", False):
            early_stopping = EarlyStopping(
                patience=es_config.get("patience", 10),
                metric=es_config.get("metric", "accuracy"),
            )

        coordinator = Coordinator(
            server=server,
            clients=clients,
            training_state=state,
            logger_instance=logger_instance,
            checkpoint_manager=checkpoint_manager,
            early_stopping=early_stopping,
        )

        return {
            "experiment": None,
            "coordinator": coordinator,
            "server": server,
            "clients": clients,
            "state": state,
            "logger": logger_instance,
            "checkpoint_manager": checkpoint_manager,
            "model": model,
        }
