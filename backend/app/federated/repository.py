from __future__ import annotations

import copy
import uuid
from typing import Any

from app.core.logging import logger
from app.federated.models import (
    AggregatedPrototype,
    AggregationRound,
    ClientPrototypePackage,
    FederatedState,
)


class FederatedRepository:
    def __init__(self):
        self._global_prototypes: dict[str, AggregatedPrototype] = {}
        self._round_history: dict[int, AggregationRound] = {}
        self._client_packages: dict[int, list[ClientPrototypePackage]] = {}
        self._previous_prototypes: dict[str, AggregatedPrototype] = {}
        self._version_counter: int = 0
        self._versions: dict[int, dict[str, AggregatedPrototype]] = {}
        self._state = FederatedState()

    @property
    def state(self) -> FederatedState:
        return self._state

    @property
    def current_round(self) -> int:
        return self._state.current_round

    def store_global_prototype(self, proto: AggregatedPrototype) -> str:
        key = f"{proto.modality}_c{proto.class_id}"
        self._previous_prototypes[key] = (
            copy.deepcopy(self._global_prototypes.get(key))
            if key in self._global_prototypes
            else proto
        )

        old = self._global_prototypes.get(key)
        if old is not None and old.round_id < proto.round_id:
            proto.metadata["previous_round"] = old.round_id
            proto.metadata["previous_confidence"] = old.confidence

        self._global_prototypes[key] = proto
        logger.debug(f"Stored global prototype {key} (round {proto.round_id})")
        return key

    def get_global_prototype(
        self, modality: str, class_id: int
    ) -> AggregatedPrototype | None:
        return self._global_prototypes.get(f"{modality}_c{class_id}")

    def list_global_prototypes(self) -> list[AggregatedPrototype]:
        return list(self._global_prototypes.values())

    def global_count(self) -> int:
        return len(self._global_prototypes)

    def has_prototype(self, modality: str, class_id: int) -> bool:
        return f"{modality}_c{class_id}" in self._global_prototypes

    def store_round(self, round_obj: AggregationRound) -> None:
        self._round_history[round_obj.round_id] = round_obj
        self._state.current_round = round_obj.round_id
        logger.debug(f"Stored round {round_obj.round_id}")

    def get_round(self, round_id: int) -> AggregationRound | None:
        return self._round_history.get(round_id)

    def list_rounds(self) -> list[AggregationRound]:
        return [self._round_history[rid] for rid in sorted(self._round_history.keys())]

    def latest_round(self) -> AggregationRound | None:
        if not self._round_history:
            return None
        return self._round_history[max(self._round_history.keys())]

    def store_client_packages(
        self, round_id: int, packages: list[ClientPrototypePackage]
    ) -> None:
        if round_id not in self._client_packages:
            self._client_packages[round_id] = []
        self._client_packages[round_id].extend(packages)
        self._state.packages_received += len(packages)
        for pkg in packages:
            self._state.total_clients_ever.add(pkg.client_id)
        logger.debug(f"Stored {len(packages)} client packages for round {round_id}")

    def get_client_packages(self, round_id: int) -> list[ClientPrototypePackage]:
        return self._client_packages.get(round_id, [])

    def get_client_ids_for_round(self, round_id: int) -> set[str]:
        return {p.client_id for p in self._client_packages.get(round_id, [])}

    def get_previous_prototype(
        self, modality: str, class_id: int
    ) -> AggregatedPrototype | None:
        return self._previous_prototypes.get(f"{modality}_c{class_id}")

    def create_snapshot(self) -> int:
        self._version_counter += 1
        snapshot = {k: copy.deepcopy(v) for k, v in self._global_prototypes.items()}
        self._versions[self._version_counter] = snapshot
        logger.debug(f"Created snapshot version {self._version_counter}")
        return self._version_counter

    def restore_snapshot(self, version: int) -> None:
        if version not in self._versions:
            raise ValueError(
                f"Snapshot version {version} not found. "
                f"Available: {sorted(self._versions.keys())}"
            )
        self._global_prototypes = copy.deepcopy(self._versions[version])
        logger.info(f"Restored snapshot version {version}")

    def list_snapshots(self) -> list[int]:
        return sorted(self._versions.keys())

    def round_count(self) -> int:
        return len(self._round_history)

    def export_state(self) -> dict[str, Any]:
        return {
            "global_prototypes": {
                k: v.model_dump() for k, v in self._global_prototypes.items()
            },
            "round_history": {
                str(rid): r.model_dump() for rid, r in self._round_history.items()
            },
            "state": self._state.to_dict(),
            "version_counter": self._version_counter,
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self._global_prototypes = {}
        for key, data in state.get("global_prototypes", {}).items():
            self._global_prototypes[key] = AggregatedPrototype(**data)
        self._round_history = {}
        for rid_str, data in state.get("round_history", {}).items():
            self._round_history[int(rid_str)] = AggregationRound(**data)
        self._version_counter = state.get("version_counter", 0)
        logger.info(
            f"Imported state: {self.global_count()} global prototypes, "
            f"{self.round_count()} rounds"
        )

    def clear(self) -> None:
        self._global_prototypes.clear()
        self._round_history.clear()
        self._client_packages.clear()
        self._previous_prototypes.clear()
        self._versions.clear()
        self._state = FederatedState()
        self._version_counter = 0
        logger.info("Cleared federated repository")
