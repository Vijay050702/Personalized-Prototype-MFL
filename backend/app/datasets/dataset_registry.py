from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.datasets.cache import DatasetCache
from app.datasets.errors import DatasetAlreadyExistsError, DatasetNotFoundError


class DatasetRegistry:
    def __init__(self, cache: DatasetCache | None = None):
        self.cache = cache or DatasetCache()

    def register(
        self, name: str, metadata: dict[str, Any], force: bool = False
    ) -> None:
        key = f"reg_{name}"
        if self.cache.exists(key):
            if not force:
                raise DatasetAlreadyExistsError(
                    f"Dataset '{name}' is already registered"
                )
            self.cache.invalidate(key)
            self.cache.invalidate(f"reg_{name}_meta")
        metadata["_name"] = name
        self.cache.set(key, metadata)
        self.cache.set_json(f"reg_{name}_meta", metadata)
        logger.info(f"Registered dataset: {name}")

    def get(self, name: str) -> dict[str, Any]:
        key = f"reg_{name}"
        data = self.cache.get(key)
        if data is None:
            raise DatasetNotFoundError(f"Dataset '{name}' not found in registry")
        return data

    def list(self) -> list[dict[str, Any]]:
        results = []
        for name in self.list_names():
            try:
                results.append(self.get(name))
            except DatasetNotFoundError:
                continue
        return results

    def list_names(self) -> list[str]:
        names = []
        for p in Path(settings.datasets_root).rglob("*_meta.json"):
            parts = p.stem.split("_")
            if len(parts) >= 2 and parts[0] == "reg":
                names.append("_".join(parts[1:-1] if len(parts) > 2 else [parts[1]]))
        return list(set(names))

    def unregister(self, name: str) -> None:
        self.cache.invalidate(f"reg_{name}")
        self.cache.invalidate(f"reg_{name}_meta")
        logger.info(f"Unregistered dataset: {name}")

    def exists(self, name: str) -> bool:
        return self.cache.exists(f"reg_{name}")

    def update_status(self, name: str, status_key: str, status_value: str) -> None:
        try:
            meta = self.get(name)
            meta[status_key] = status_value
            self.cache.set(f"reg_{name}", meta)
            self.cache.set_json(f"reg_{name}_meta", meta)
        except DatasetNotFoundError:
            pass
