from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from app.federated.models import (
    AggregatedPrototype,
    ClientPrototypePackage,
)


class PrototypeSerializer:
    @staticmethod
    def package_to_dict(package: ClientPrototypePackage) -> dict[str, Any]:
        return package.model_dump()

    @staticmethod
    def package_from_dict(data: dict[str, Any]) -> ClientPrototypePackage:
        return ClientPrototypePackage(**data)

    @staticmethod
    def package_to_json(package: ClientPrototypePackage) -> str:
        return package.model_dump_json()

    @staticmethod
    def package_from_json(data: str) -> ClientPrototypePackage:
        return ClientPrototypePackage.model_validate_json(data)

    @staticmethod
    def packages_to_json(packages: list[ClientPrototypePackage]) -> str:
        return json.dumps([p.model_dump() for p in packages])

    @staticmethod
    def packages_from_json(data: str) -> list[ClientPrototypePackage]:
        raw: list[dict[str, Any]] = json.loads(data)
        return [ClientPrototypePackage(**item) for item in raw]

    @staticmethod
    def aggregated_to_dict(proto: AggregatedPrototype) -> dict[str, Any]:
        return proto.model_dump()

    @staticmethod
    def aggregated_from_dict(data: dict[str, Any]) -> AggregatedPrototype:
        return AggregatedPrototype(**data)

    @staticmethod
    def compute_checksum(package: ClientPrototypePackage) -> str:
        raw = package.model_dump_json().encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def verify_checksum(package: ClientPrototypePackage, checksum: str) -> bool:
        return PrototypeSerializer.compute_checksum(package) == checksum

    @staticmethod
    def compress_tensor(tensor: torch.Tensor, decimals: int = 6) -> list[float]:
        return [round(v, decimals) for v in tensor.detach().cpu().tolist()]

    @staticmethod
    def serialize_with_metadata(
        packages: list[ClientPrototypePackage],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "packages": [p.model_dump() for p in packages],
            "metadata": metadata or {},
            "num_packages": len(packages),
        }
        return json.dumps(payload)

    @staticmethod
    def deserialize_with_metadata(
        data: str,
    ) -> tuple[list[ClientPrototypePackage], dict[str, Any]]:
        payload = json.loads(data)
        packages = [ClientPrototypePackage(**item) for item in payload["packages"]]
        return packages, payload.get("metadata", {})
