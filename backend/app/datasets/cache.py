from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import logger


class DatasetCache:
    def __init__(self, cache_root: str | Path | None = None):
        self.cache_root = Path(cache_root or settings.datasets_root) / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self.cache_root / f"{key}.cache"

    def get(self, key: str) -> Any | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read failed for {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def get_json(self, key: str) -> dict | None:
        path = self._path_for(key).with_suffix(".json")
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache JSON read failed for {key}: {e}")
            return None

    def set_json(self, key: str, value: dict) -> None:
        path = self._path_for(key).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f, indent=2)

    def exists(self, key: str) -> bool:
        return (
            self._path_for(key).exists()
            or self._path_for(key).with_suffix(".json").exists()
        )

    def invalidate(self, key: str) -> None:
        for p in [self._path_for(key), self._path_for(key).with_suffix(".json")]:
            if p.exists():
                p.unlink()

    def clear(self) -> None:
        for p in self.cache_root.iterdir():
            if p.is_file():
                p.unlink()
