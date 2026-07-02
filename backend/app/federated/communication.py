from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.federated.models import ClientPrototypePackage
from app.federated.serialization import PrototypeSerializer


class CommunicationHandler:
    def __init__(self, serializer: PrototypeSerializer | None = None):
        self._serializer = serializer or PrototypeSerializer()
        self._received_packages: list[ClientPrototypePackage] = []
        self._sent_packages: list[ClientPrototypePackage] = []

    def receive_package(self, data: dict[str, Any]) -> ClientPrototypePackage:
        package = self._serializer.package_from_dict(data)
        self._validate_package(package)
        self._received_packages.append(package)
        logger.debug(
            f"Received package: client={package.client_id} "
            f"round={package.round_id} modality={package.modality} "
            f"class={package.class_id}"
        )
        return package

    def receive_package_json(self, data: str) -> ClientPrototypePackage:
        package = self._serializer.package_from_json(data)
        self._validate_package(package)
        self._received_packages.append(package)
        return package

    def receive_batch(self, data: list[dict[str, Any]]) -> list[ClientPrototypePackage]:
        packages = [self._serializer.package_from_dict(d) for d in data]
        for pkg in packages:
            self._validate_package(pkg)
            self._received_packages.append(pkg)
        logger.debug(f"Received batch of {len(packages)} packages")
        return packages

    def send_package(self, package: ClientPrototypePackage) -> dict[str, Any]:
        self._sent_packages.append(package)
        return self._serializer.package_to_dict(package)

    def send_package_json(self, package: ClientPrototypePackage) -> str:
        self._sent_packages.append(package)
        return self._serializer.package_to_json(package)

    def send_aggregated(
        self, packages: list[ClientPrototypePackage]
    ) -> list[dict[str, Any]]:
        for pkg in packages:
            self._sent_packages.append(pkg)
        return [self._serializer.package_to_dict(p) for p in packages]

    @staticmethod
    def _validate_package(package: ClientPrototypePackage) -> None:
        if not package.client_id:
            raise ValueError("Package missing client_id")
        if not package.modality:
            raise ValueError("Package missing modality")
        if package.sample_count < 1:
            raise ValueError(f"sample_count must be >= 1, got {package.sample_count}")
        if package.embedding_dim != len(package.prototype_vector):
            raise ValueError(
                f"embedding_dim ({package.embedding_dim}) does not match "
                f"vector length ({len(package.prototype_vector)})"
            )

    @property
    def received_count(self) -> int:
        return len(self._received_packages)

    @property
    def sent_count(self) -> int:
        return len(self._sent_packages)

    def clear_history(self) -> None:
        self._received_packages.clear()
        self._sent_packages.clear()
