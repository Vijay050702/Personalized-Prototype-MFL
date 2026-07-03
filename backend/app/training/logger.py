from __future__ import annotations

import time
from typing import Any

from app.core.logging import logger as core_logger


class TrainingLogger:
    def __init__(self, experiment_id: str = "") -> None:
        self._experiment_id = experiment_id
        self._logs: list[dict[str, Any]] = []

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "experiment_id": self._experiment_id,
            **kwargs,
        }
        self._logs.append(entry)
        log_fn = getattr(core_logger, level.lower(), core_logger.info)
        extra = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        log_fn(f"[{self._experiment_id}] {message}" + (f" ({extra})" if extra else ""))

    def log_experiment_start(self, config: dict[str, Any]) -> None:
        self._log("info", "Experiment started", config_summary=str(config))

    def log_experiment_end(self, status: str = "completed") -> None:
        self._log("info", f"Experiment {status}")

    def log_round_start(self, round_id: int, num_clients: int) -> None:
        self._log("info", f"Round {round_id} started", num_clients=num_clients)

    def log_round_end(
        self, round_id: int, metrics: dict[str, float] | None = None
    ) -> None:
        metrics_str = str(metrics) if metrics else ""
        self._log("info", f"Round {round_id} ended", metrics=metrics_str)

    def log_client_update(
        self,
        client_id: str,
        round_id: int,
        loss: float | None = None,
        accuracy: float | None = None,
        num_samples: int = 0,
    ) -> None:
        self._log(
            "debug",
            f"Client {client_id} round {round_id} update",
            client_id=client_id,
            round_id=round_id,
            loss=loss,
            accuracy=accuracy,
            num_samples=num_samples,
        )

    def log_aggregation(
        self,
        round_id: int,
        num_clients: int,
        num_prototypes: int,
    ) -> None:
        self._log(
            "info",
            f"Aggregation round {round_id}",
            num_clients=num_clients,
            num_prototypes=num_prototypes,
        )

    def log_knowledge_transfer(
        self,
        round_id: int,
        num_synthesized: int,
        modalities: list[str] | None = None,
    ) -> None:
        self._log(
            "info",
            f"Knowledge transfer round {round_id}",
            num_synthesized=num_synthesized,
            modalities=modalities,
        )

    def log_personalization(
        self,
        round_id: int,
        num_personalized: int,
        fusion_strategy: str = "",
    ) -> None:
        self._log(
            "info",
            f"Personalization round {round_id}",
            num_personalized=num_personalized,
            fusion_strategy=fusion_strategy,
        )

    def log_evaluation(
        self,
        round_id: int,
        metrics: dict[str, float],
    ) -> None:
        self._log("info", f"Evaluation round {round_id}", metrics=str(metrics))

    def log_checkpoint(
        self,
        round_id: int,
        path: str,
        checkpoint_type: str = "latest",
    ) -> None:
        self._log(
            "info",
            f"Checkpoint saved round {round_id}",
            path=path,
            checkpoint_type=checkpoint_type,
        )

    def log_error(
        self,
        message: str,
        round_id: int | None = None,
    ) -> None:
        self._log("error", message, round_id=round_id)

    def log_warning(
        self,
        message: str,
        round_id: int | None = None,
    ) -> None:
        self._log("warning", message, round_id=round_id)

    def log_synchronization(
        self,
        round_id: int,
        num_clients: int,
        components: list[str] | None = None,
    ) -> None:
        self._log(
            "info",
            f"Synchronization round {round_id}",
            num_clients=num_clients,
            components=components,
        )

    def get_recent_logs(self, n: int = 50) -> list[dict[str, Any]]:
        return self._logs[-n:]

    def get_logs_by_level(self, level: str) -> list[dict[str, Any]]:
        return [log for log in self._logs if log["level"] == level]

    @property
    def log_count(self) -> int:
        return len(self._logs)

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self._experiment_id,
            "total_logs": self.log_count,
            "errors": len(self.get_logs_by_level("error")),
            "warnings": len(self.get_logs_by_level("warning")),
        }

    def clear(self) -> None:
        self._logs.clear()
