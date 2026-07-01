from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.datasets.adapters.uci_har import uci_har_adapter
from app.datasets.adapters.meld import meld_adapter
from app.datasets.adapters.ptb_xl import ptbxl_adapter
from app.datasets.adapters.hateful_memes import hateful_memes_adapter
from app.datasets.adapters.generic import GenericDatasetAdapter
from app.datasets.base import BaseDatasetAdapter
from app.datasets.errors import DatasetNotFoundError


class DatasetFactory:
    _adapters: dict[str, type[BaseDatasetAdapter]] = {}
    _instances: dict[str, BaseDatasetAdapter] = {}

    @classmethod
    def register(
        cls, name: str, adapter: type[BaseDatasetAdapter] | BaseDatasetAdapter
    ) -> None:
        if isinstance(adapter, type):
            cls._adapters[name] = adapter
        else:
            cls._instances[name] = adapter

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseDatasetAdapter:
        if name in cls._instances:
            return cls._instances[name]

        if name in cls._adapters:
            instance = cls._adapters[name](**kwargs)
            cls._instances[name] = instance
            return instance

        raise DatasetNotFoundError(
            f"Dataset adapter not registered: '{name}'. Available: {cls.list_available()}"
        )

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(set(list(cls._adapters.keys()) + list(cls._instances.keys())))

    @classmethod
    def exists(cls, name: str) -> bool:
        return name in cls._instances or name in cls._adapters


DatasetFactory.register("uci_har", uci_har_adapter)
DatasetFactory.register("meld", meld_adapter)
DatasetFactory.register("ptb_xl", ptbxl_adapter)
DatasetFactory.register("hateful_memes", hateful_memes_adapter)
DatasetFactory.register("generic", GenericDatasetAdapter)
