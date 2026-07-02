from __future__ import annotations

import copy
import time
from typing import Any

import torch

from app.core.logging import logger
from app.prototypes.prototype import Prototype
from app.prototypes.repository import PrototypeRepository


class PrototypeMemory:
    def __init__(self, max_global: int = 1000, max_local: int = 100):
        self._global_repo = PrototypeRepository()
        self._local_repo = PrototypeRepository()
        self._max_global = max_global
        self._max_local = max_local
        self._snapshots: list[dict[str, Any]] = []
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._max_snapshots = 50

    @property
    def global_repo(self) -> PrototypeRepository:
        return self._global_repo

    @property
    def local_repo(self) -> PrototypeRepository:
        return self._local_repo

    def store_global(self, prototype: Prototype) -> None:
        if self._global_repo.size >= self._max_global:
            self._evict_one(self._global_repo)
        self._global_repo.store(prototype)
        self._track_history(prototype, "stored")

    def store_local(self, prototype: Prototype) -> None:
        if self._local_repo.size >= self._max_local:
            self._evict_one(self._local_repo)
        self._local_repo.store(prototype)
        self._track_history(prototype, "stored_local")

    def promote_to_global(self, prototype_id: str) -> Prototype | None:
        try:
            proto = self._local_repo.retrieve(prototype_id)
            self._local_repo.remove(prototype_id)
            self.store_global(proto)
            logger.info(f"Promoted prototype {prototype_id} to global")
            return proto
        except KeyError:
            logger.warning(f"Prototype {prototype_id} not found in local memory")
            return None

    def snapshot(self) -> None:
        snapshot = {
            "timestamp": time.time(),
            "global_state": self._global_repo.export_state(),
            "local_state": self._local_repo.export_state(),
        }
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)
        logger.debug(f"Snapshot taken ({len(self._snapshots)} total)")

    def restore_snapshot(self, index: int = -1) -> None:
        if not self._snapshots:
            raise ValueError("No snapshots available")
        snapshot = self._snapshots[index]
        self._global_repo.import_state(snapshot["global_state"])
        self._local_repo.import_state(snapshot["local_state"])
        logger.info(f"Restored snapshot from index {index}")

    def clear_snapshots(self) -> None:
        self._snapshots.clear()
        logger.debug("Cleared all snapshots")

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def global_size(self) -> int:
        return self._global_repo.size

    @property
    def local_size(self) -> int:
        return self._local_repo.size

    @property
    def total_size(self) -> int:
        return self.global_size + self.local_size

    def get_history(self, prototype_id: str) -> list[dict[str, Any]]:
        return self._history.get(prototype_id, [])

    def _track_history(self, prototype: Prototype, event: str) -> None:
        pid = prototype.prototype_id
        if pid not in self._history:
            self._history[pid] = []
        self._history[pid].append(
            {
                "event": event,
                "timestamp": time.time(),
                "class_id": prototype.class_id,
                "confidence": prototype.confidence,
                "sample_count": prototype.sample_count,
            }
        )
        if len(self._history[pid]) > 100:
            self._history[pid] = self._history[pid][-100:]

    def _evict_one(self, repo: PrototypeRepository) -> None:
        prototypes = repo.list()
        if not prototypes:
            return
        oldest = min(prototypes, key=lambda p: p.timestamp)
        repo.remove(oldest.prototype_id)
        logger.debug(f"Evicted prototype {oldest.prototype_id} (oldest)")

    def age_prototypes(self, max_age: float = 86400.0) -> int:
        now = time.time()
        aged: list[Prototype] = []
        for proto in self._global_repo.list():
            if now - proto.timestamp > max_age:
                aged.append(proto)
        for proto in aged:
            self._global_repo.remove(proto.prototype_id)
        if aged:
            logger.info(f"Aged out {len(aged)} prototypes older than {max_age}s")
        return len(aged)

    def statistics(self) -> dict[str, Any]:
        return {
            "global_count": self.global_size,
            "local_count": self.local_size,
            "total": self.total_size,
            "max_global": self._max_global,
            "max_local": self._max_local,
            "snapshots": self.num_snapshots,
            "global": self._global_repo.statistics(),
            "local": self._local_repo.statistics(),
        }

    def clear(self) -> None:
        self._global_repo.clear()
        self._local_repo.clear()
        self._snapshots.clear()
        self._history.clear()
        logger.info("Cleared all memory")

    def __repr__(self) -> str:
        return f"PrototypeMemory(global={self.global_size}, local={self.local_size}, snapshots={self.num_snapshots})"
