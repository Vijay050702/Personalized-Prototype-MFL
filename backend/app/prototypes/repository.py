from __future__ import annotations

from typing import Any, Callable

import torch

from app.core.logging import logger
from app.prototypes.prototype import Prototype


class PrototypeRepository:
    def __init__(self) -> None:
        self._prototypes: dict[str, Prototype] = {}

    def store(self, prototype: Prototype) -> None:
        if prototype.prototype_id in self._prototypes:
            logger.warning(f"Overwriting prototype {prototype.prototype_id}")
        self._prototypes[prototype.prototype_id] = prototype
        logger.debug(f"Stored prototype {prototype}")

    def retrieve(self, prototype_id: str) -> Prototype:
        if prototype_id not in self._prototypes:
            raise KeyError(f"Prototype '{prototype_id}' not found")
        return self._prototypes[prototype_id]

    def replace(self, prototype: Prototype) -> None:
        if prototype.prototype_id not in self._prototypes:
            raise KeyError(
                f"Cannot replace: prototype '{prototype.prototype_id}' not found"
            )
        self._prototypes[prototype.prototype_id] = prototype
        logger.debug(f"Replaced prototype {prototype.prototype_id}")

    def update(self, prototype_id: str, **updates: Any) -> Prototype:
        proto = self.retrieve(prototype_id)
        if "embedding" in updates:
            proto.embedding = updates["embedding"]
        if "confidence" in updates:
            proto.confidence = updates["confidence"]
        if "sample_count" in updates:
            proto.sample_count = updates["sample_count"]
        if "class_id" in updates:
            proto.class_id = updates["class_id"]
        if "metadata" in updates:
            proto.metadata.update(updates["metadata"])
        logger.debug(f"Updated prototype {prototype_id}")
        return proto

    def remove(self, prototype_id: str) -> None:
        if prototype_id not in self._prototypes:
            raise KeyError(f"Prototype '{prototype_id}' not found")
        del self._prototypes[prototype_id]
        logger.debug(f"Removed prototype {prototype_id}")

    def list(self) -> list[Prototype]:
        return list(self._prototypes.values())

    def clear(self) -> None:
        self._prototypes.clear()
        logger.debug("Cleared all prototypes")

    def filter(self, predicate: Callable[[Prototype], bool]) -> list[Prototype]:
        return [p for p in self._prototypes.values() if predicate(p)]

    @property
    def size(self) -> int:
        return len(self._prototypes)

    @property
    def is_empty(self) -> bool:
        return len(self._prototypes) == 0

    def by_class(self, class_id: int) -> list[Prototype]:
        return self.filter(lambda p: p.class_id == class_id)

    def by_modality(self, modality: str) -> list[Prototype]:
        return self.filter(lambda p: p.modality == modality)

    def class_ids(self) -> set[int]:
        return {p.class_id for p in self._prototypes.values()}

    def modalities(self) -> set[str]:
        return {p.modality for p in self._prototypes.values()}

    def get_embeddings_matrix(self) -> torch.Tensor:
        if self.is_empty:
            return torch.empty(0, 0)
        return torch.stack([p.embedding for p in self._prototypes.values()], dim=0)

    def get_labels(self) -> list[int]:
        return [p.class_id for p in self._prototypes.values()]

    def statistics(self) -> dict[str, Any]:
        if self.is_empty:
            return {"count": 0, "classes": 0, "modalities": 0}
        return {
            "count": self.size,
            "classes": len(self.class_ids()),
            "modalities": len(self.modalities()),
            "class_ids": sorted(self.class_ids()),
            "modality_list": list(self.modalities()),
            "avg_confidence": float(
                torch.tensor([p.confidence for p in self._prototypes.values()]).mean()
            ),
            "total_samples": sum(p.sample_count for p in self._prototypes.values()),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "prototypes": [p.to_dict() for p in self._prototypes.values()],
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self.clear()
        for proto_dict in state["prototypes"]:
            proto = Prototype(
                embedding=torch.tensor(proto_dict["embedding"]),
                class_id=proto_dict["class_id"],
                modality=proto_dict.get("modality", "shared"),
                prototype_id=proto_dict["prototype_id"],
                sample_count=proto_dict["sample_count"],
                confidence=proto_dict["confidence"],
                metadata=proto_dict.get("metadata", {}),
            )
            self.store(proto)
        logger.info(f"Imported {self.size} prototypes")

    def __repr__(self) -> str:
        return f"PrototypeRepository(size={self.size})"
