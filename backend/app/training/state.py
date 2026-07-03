from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ClientState:
    client_id: str
    model_state: dict[str, torch.Tensor] | None = None
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    prototype_state: dict[str, Any] | None = None
    personalized_state: dict[str, Any] | None = None
    current_round: int = 0
    epochs_completed: int = 0
    samples_processed: int = 0
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    is_active: bool = True
    joined_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "epochs_completed": self.epochs_completed,
            "samples_processed": self.samples_processed,
            "loss_history": self.loss_history[-100:],
            "accuracy_history": self.accuracy_history[-100:],
            "is_active": self.is_active,
            "joined_at": self.joined_at,
            "metadata": self.metadata,
        }


@dataclass
class ServerState:
    current_round: int = 0
    total_clients_ever: set[str] = field(default_factory=set)
    rounds_completed: int = 0
    global_prototype_count: int = 0
    started_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_round": self.current_round,
            "total_clients_ever": len(self.total_clients_ever),
            "rounds_completed": self.rounds_completed,
            "global_prototype_count": self.global_prototype_count,
            "started_at": self.started_at,
            "metadata": self.metadata,
        }


@dataclass
class TrainingState:
    experiment_id: str
    config: dict[str, Any] = field(default_factory=dict)
    current_round: int = 0
    total_rounds: int = 100
    server: ServerState = field(default_factory=ServerState)
    clients: dict[str, ClientState] = field(default_factory=dict)
    round_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    best_round: int = 0
    best_metric: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    phase: str = "initialized"
    metadata: dict[str, Any] = field(default_factory=dict)

    def register_client(self, client_id: str) -> ClientState:
        if client_id not in self.clients:
            self.clients[client_id] = ClientState(client_id=client_id)
            self.server.total_clients_ever.add(client_id)
        return self.clients[client_id]

    def get_client(self, client_id: str) -> ClientState:
        return self.clients[client_id]

    def active_clients(self) -> list[ClientState]:
        return [c for c in self.clients.values() if c.is_active]

    def client_ids(self) -> list[str]:
        return list(self.clients.keys())

    def record_round_metrics(self, round_id: int, metrics: dict[str, float]) -> None:
        self.round_metrics[round_id] = metrics
        if metrics.get("accuracy", 0.0) > self.best_metric:
            self.best_metric = metrics.get("accuracy", 0.0)
            self.best_round = round_id
        self.current_round = round_id
        self.server.current_round = round_id
        self.server.rounds_completed += 1

    def mark_completed(self) -> None:
        self.end_time = time.time()
        self.phase = "completed"

    def mark_failed(self) -> None:
        self.end_time = time.time()
        self.phase = "failed"

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def is_completed(self) -> bool:
        return self.phase == "completed"

    @property
    def is_running(self) -> bool:
        return self.phase == "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "phase": self.phase,
            "server": self.server.to_dict(),
            "clients": {k: v.to_dict() for k, v in self.clients.items()},
            "best_round": self.best_round,
            "best_metric": self.best_metric,
            "elapsed": self.elapsed,
            "metadata": self.metadata,
        }
