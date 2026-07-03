from __future__ import annotations

import time
from typing import Any

from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.validation import validate_duplicate_prototypes


class PersonalizedMemory:
    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._prototypes: list[PersonalizedPrototype] = []
        self._history: dict[str, list[dict[str, Any]]] = {}

    def store(self, prototype: PersonalizedPrototype) -> None:
        existing = self.retrieve(
            prototype.client_id, prototype.class_id, prototype.modality
        )
        if existing is not None:
            raise ValueError(
                f"Duplicate PersonalizedPrototype for "
                f"client={prototype.client_id}, class={prototype.class_id}, "
                f"modality={prototype.modality}"
            )
        if len(self._prototypes) >= self._capacity:
            self._prototypes.pop(0)
        self._prototypes.append(prototype)
        pid = f"{prototype.client_id}_{prototype.class_id}_{prototype.modality}"
        self._history.setdefault(pid, []).append(
            {
                "confidence": prototype.confidence,
                "fusion_weights": prototype.fusion_weights,
                "timestamp": time.time(),
            }
        )

    def retrieve(
        self,
        client_id: str,
        class_id: int,
        modality: str,
    ) -> PersonalizedPrototype | None:
        for p in self._prototypes:
            if (
                p.client_id == client_id
                and p.class_id == class_id
                and p.modality == modality
            ):
                return p
        return None

    def retrieve_all(
        self,
        client_id: str | None = None,
        class_id: int | None = None,
        modality: str | None = None,
    ) -> list[PersonalizedPrototype]:
        results = self._prototypes
        if client_id is not None:
            results = [p for p in results if p.client_id == client_id]
        if class_id is not None:
            results = [p for p in results if p.class_id == class_id]
        if modality is not None:
            results = [p for p in results if p.modality == modality]
        return results

    def retrieve_by_client(self, client_id: str) -> list[PersonalizedPrototype]:
        return [p for p in self._prototypes if p.client_id == client_id]

    def retrieve_by_class(self, class_id: int) -> list[PersonalizedPrototype]:
        return [p for p in self._prototypes if p.class_id == class_id]

    def retrieve_by_modality(self, modality: str) -> list[PersonalizedPrototype]:
        return [p for p in self._prototypes if p.modality == modality]

    def update(
        self,
        prototype: PersonalizedPrototype,
    ) -> bool:
        for i, p in enumerate(self._prototypes):
            if (
                p.client_id == prototype.client_id
                and p.class_id == prototype.class_id
                and p.modality == prototype.modality
            ):
                self._prototypes[i] = prototype
                pid = f"{prototype.client_id}_{prototype.class_id}_{prototype.modality}"
                self._history.setdefault(pid, []).append(
                    {
                        "confidence": prototype.confidence,
                        "fusion_weights": prototype.fusion_weights,
                        "timestamp": time.time(),
                        "type": "update",
                    }
                )
                return True
        return False

    def remove(
        self,
        client_id: str,
        class_id: int,
        modality: str,
    ) -> bool:
        for i, p in enumerate(self._prototypes):
            if (
                p.client_id == client_id
                and p.class_id == class_id
                and p.modality == modality
            ):
                self._prototypes.pop(i)
                return True
        return False

    def clear_client(self, client_id: str) -> int:
        before = len(self._prototypes)
        self._prototypes = [p for p in self._prototypes if p.client_id != client_id]
        return before - len(self._prototypes)

    def get_history(
        self,
        client_id: str,
        class_id: int,
        modality: str,
    ) -> list[dict[str, Any]]:
        pid = f"{client_id}_{class_id}_{modality}"
        return self._history.get(pid, [])

    @property
    def size(self) -> int:
        return len(self._prototypes)

    @property
    def capacity(self) -> int:
        return self._capacity

    def statistics(self) -> dict[str, Any]:
        clients = len(set(p.client_id for p in self._prototypes))
        classes = len(set(p.class_id for p in self._prototypes))
        modalities = len(set(p.modality for p in self._prototypes))
        return {
            "size": self.size,
            "capacity": self._capacity,
            "unique_clients": clients,
            "unique_classes": classes,
            "unique_modalities": modalities,
        }

    def clear(self) -> None:
        self._prototypes.clear()
        self._history.clear()
