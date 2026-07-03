from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from app.core.logging import logger
from app.training.state import TrainingState


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        experiment_id: str = "",
        max_checkpoints: int = 5,
    ) -> None:
        self._base_dir = Path(checkpoint_dir) / experiment_id
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_id = experiment_id
        self._max_checkpoints = max_checkpoints
        self._checkpoints: list[dict[str, Any]] = []

    def _checkpoint_path(
        self, checkpoint_type: str, round_id: int | None = None
    ) -> Path:
        if checkpoint_type == "latest":
            return self._base_dir / "checkpoint_latest.pt"
        if checkpoint_type == "best":
            return self._base_dir / "checkpoint_best.pt"
        if round_id is not None:
            return self._base_dir / f"checkpoint_round_{round_id}.pt"
        return self._base_dir / f"checkpoint_{checkpoint_type}.pt"

    def save(
        self,
        state: TrainingState,
        round_id: int,
        server_state: dict[str, Any] | None = None,
        client_states: dict[str, dict[str, Any]] | None = None,
        model_state: dict[str, torch.Tensor] | None = None,
        optimizer_state: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        prototype_repo_state: dict[str, Any] | None = None,
        personalization_state: dict[str, Any] | None = None,
        kt_state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint: dict[str, Any] = {
            "experiment_id": self._experiment_id,
            "round_id": round_id,
            "timestamp": time.time(),
            "state": state.to_dict() if state else None,
            "server_state": server_state,
            "client_states": client_states,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "prototype_repo_state": prototype_repo_state,
            "personalization_state": personalization_state,
            "kt_state": kt_state,
            "extra": extra or {},
        }
        path = self._checkpoint_path("latest")
        torch.save(checkpoint, path)
        self._checkpoints.append(
            {"round_id": round_id, "path": str(path), "timestamp": time.time()}
        )
        self._enforce_max_checkpoints()
        logger.info(f"Checkpoint saved to {path} (round {round_id})")
        return path

    def save_best(
        self,
        round_id: int,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> Path:
        path = self._checkpoint_path("best")
        checkpoint: dict[str, Any] = {
            "experiment_id": self._experiment_id,
            "round_id": round_id,
            "timestamp": time.time(),
            "metrics": metrics,
        }
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
        self._checkpoints.append(
            {
                "round_id": round_id,
                "path": str(path),
                "timestamp": time.time(),
                "type": "best",
            }
        )
        self._enforce_max_checkpoints()
        logger.info(f"Best checkpoint saved to {path} (round {round_id})")
        return path

    def save_latest(
        self,
        round_id: int,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> Path:
        path = self._checkpoint_path("latest")
        checkpoint: dict[str, Any] = {
            "experiment_id": self._experiment_id,
            "round_id": round_id,
            "timestamp": time.time(),
            "metrics": metrics or {},
        }
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
        self._checkpoints.append(
            {
                "round_id": round_id,
                "path": str(path),
                "timestamp": time.time(),
                "type": "latest",
            }
        )
        self._enforce_max_checkpoints()
        logger.info(f"Latest checkpoint saved to {path} (round {round_id})")
        return path

    def load(
        self,
        checkpoint_type: str = "latest",
        round_id: int | None = None,
    ) -> dict[str, Any]:
        path = (
            self._checkpoint_path("latest")
            if checkpoint_type == "latest"
            else self._checkpoint_path("best")
        )
        if round_id is not None:
            path = self._checkpoint_path("round", round_id)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint

    def resume(
        self,
    ) -> dict[str, Any]:
        return self.load("latest")

    def list_checkpoints(self) -> list[dict[str, Any]]:
        return list(self._checkpoints)

    def latest_checkpoint(self) -> dict[str, Any] | None:
        if not self._checkpoints:
            return None
        return max(self._checkpoints, key=lambda c: c["round_id"])

    def has_checkpoint(self, checkpoint_type: str = "latest") -> bool:
        path = self._checkpoint_path(checkpoint_type)
        return path.exists()

    def _enforce_max_checkpoints(self) -> None:
        if len(self._checkpoints) <= self._max_checkpoints:
            return
        to_remove = self._checkpoints[: -self._max_checkpoints]
        self._checkpoints = self._checkpoints[-self._max_checkpoints :]
        for cp in to_remove:
            p = Path(cp["path"])
            if p.exists():
                p.unlink()

    def checkpoint_dir(self) -> str:
        return str(self._base_dir)

    def clear(self) -> None:
        import shutil

        if self._base_dir.exists():
            shutil.rmtree(self._base_dir)
            self._base_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints.clear()
