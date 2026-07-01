from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.datasets import dataset_cache, dataset_factory, dataset_registry
from app.datasets.base import BaseDatasetAdapter
from app.datasets.downloader import download_manager
from app.datasets.adapters.generic import GenericDatasetAdapter
from app.datasets.errors import DatasetAlreadyExistsError, DatasetNotFoundError
from app.datasets.metadata import metadata_generator
from app.datasets.missing_modality import MissingModalitySimulator
from app.datasets.partitioners.dirichlet import DirichletPartitioner
from app.datasets.partitioners.iid import IIDPartitioner
from app.datasets.partitioners.shard import ShardPartitioner
from app.datasets.preprocessors.image import ImagePreprocessor
from app.datasets.preprocessors.text import TextPreprocessor
from app.datasets.preprocessors.audio import AudioPreprocessor
from app.datasets.preprocessors.sensor import SensorPreprocessor
from app.schemas.dataset import (
    DatasetDetailResponse,
    DatasetMetadataResponse,
    DatasetResponse,
)


class DatasetService:
    def __init__(self):
        self.missing_modality_simulator = MissingModalitySimulator()

    def _get_ds_root(self) -> Path:
        return Path(settings.datasets_root)

    def _get_adapter(self, name: str) -> BaseDatasetAdapter:
        return dataset_factory.create(name)

    def get_datasets(self) -> list[DatasetResponse]:
        registered = dataset_registry.list()
        if not registered:
            self._register_builtins()
            registered = dataset_registry.list()

        results = []
        for meta in registered:
            name = meta.get("_name", meta.get("dataset_name", "unknown"))
            try:
                adapter = self._get_adapter(name)
                m = adapter.get_metadata()
                results.append(
                    DatasetResponse(
                        id=name,
                        name=name.replace("_", " ").title(),
                        type=m.get("modalities", ["unknown"])[0],
                        modality=", ".join(m.get("modalities", ["unknown"])),
                        size_mb=0.0,
                        samples=m.get("num_samples", 0),
                        classes=m.get("num_classes", 0),
                        client_id="server",
                        distribution=meta.get("partition_status", "not_partitioned"),
                    )
                )
            except DatasetNotFoundError:
                continue
        return results

    def get_dataset_detail(self, name: str) -> DatasetMetadataResponse:
        try:
            adapter = self._get_adapter(name)
        except DatasetNotFoundError:
            if dataset_registry.exists(name):
                reg = dataset_registry.get(name)
                mods = reg.get("modalities", ["image"])
                adapter = GenericDatasetAdapter({"name": name, "modalities": mods})
            else:
                raise
        meta = adapter.get_metadata()
        reg = dataset_registry.get(name) if dataset_registry.exists(name) else {}
        return DatasetMetadataResponse(
            dataset_name=meta.get("name", name),
            modalities=meta.get("modalities", []),
            classes=meta.get("classes", []),
            num_classes=meta.get("num_classes", 0),
            input_shapes={k: list(v) for k, v in meta.get("input_shapes", {}).items()},
            num_samples=meta.get("num_samples", 0),
            client_count=reg.get("client_count", 0),
            missing_modality_ratio=reg.get("missing_modality_ratio", 0.0),
            download_status=reg.get("download_status", "not_downloaded"),
            preprocessing_status=reg.get("preprocessing_status", "not_preprocessed"),
            partition_status=reg.get("partition_status", "not_partitioned"),
        )

    def register_dataset(
        self, name: str, modalities: list[str] | None = None
    ) -> DatasetMetadataResponse:
        if dataset_registry.exists(name):
            raise DatasetAlreadyExistsError(f"Dataset '{name}' already registered")

        if dataset_factory.exists(name):
            adapter = self._get_adapter(name)
        else:
            adapter = GenericDatasetAdapter(
                {"name": name, "modalities": modalities or ["image"]}
            )

        meta = adapter.get_metadata()
        if modalities:
            meta["modalities"] = modalities
        meta.pop("_name", None)

        dataset_registry.register(
            name,
            {
                "download_status": "not_downloaded",
                "preprocessing_status": "not_preprocessed",
                "partition_status": "not_partitioned",
                "client_count": 0,
                "missing_modality_ratio": 0.0,
            },
        )

        saved = metadata_generator.generate(
            name=name,
            classes=meta.get("classes", []),
            modalities=meta.get("modalities", []),
            input_shapes=meta.get("input_shapes", {}),
            num_samples=meta.get("num_samples", 0),
        )
        metadata_generator.save(saved)

        return self.get_dataset_detail(name)

    def download_dataset(self, name: str, force: bool = False) -> dict[str, Any]:
        adapter = self._get_adapter(name)
        dataset_registry.update_status(name, "download_status", "downloading")
        try:
            dest = self._get_ds_root()
            adapter.download(dest, force=force)
            dataset_registry.update_status(name, "download_status", "downloaded")
            return {
                "status": "success",
                "message": f"Dataset '{name}' downloaded",
                "dataset_name": name,
            }
        except Exception as e:
            dataset_registry.update_status(name, "download_status", "failed")
            logger.error(f"Download failed for {name}: {e}")
            raise

    def preprocess_dataset(self, name: str, force: bool = False) -> dict[str, Any]:
        adapter = self._get_adapter(name)
        dataset_registry.update_status(name, "preprocessing_status", "preprocessing")

        try:
            ds_root = self._get_ds_root()
            processed_dir = ds_root / "processed" / name
            if processed_dir.exists() and not force:
                return {
                    "status": "success",
                    "message": f"Dataset '{name}' already preprocessed",
                    "dataset_name": name,
                }

            train_data = adapter.load(ds_root, split="train")
            processed_dir.mkdir(parents=True, exist_ok=True)

            preprocessor = self._select_preprocessor(name, adapter.modalities)
            if preprocessor:
                preprocessor.fit(train_data.data)
                processed_data = []
                for sample in train_data.data:
                    for mod, arr in sample.items():
                        if mod in adapter.modalities:
                            sample[mod] = preprocessor.process(arr)
                    processed_data.append(sample)

                np.save(
                    processed_dir / "train_data.npy",
                    np.array(processed_data, dtype=object),
                )
                np.save(processed_dir / "train_labels.npy", train_data.labels)

            meta = adapter.get_metadata()
            metadata_generator.save(
                meta | {"preprocessing_status": "preprocessed"}, processed_dir
            )
            dataset_registry.update_status(name, "preprocessing_status", "preprocessed")

            return {
                "status": "success",
                "message": f"Dataset '{name}' preprocessed",
                "dataset_name": name,
            }
        except Exception as e:
            dataset_registry.update_status(name, "preprocessing_status", "failed")
            logger.error(f"Preprocessing failed for {name}: {e}")
            raise

    def partition_dataset(
        self,
        dataset_name: str,
        strategy: str = "iid",
        num_clients: int = 10,
        alpha: float | None = 0.5,
        seed: int = 42,
        balanced: bool = True,
        shards_per_client: int = 2,
        min_samples: int = 1,
    ) -> dict[str, Any]:
        self._ensure_registered(dataset_name)
        adapter = self._get_adapter(dataset_name)
        ds_root = self._get_ds_root()
        dataset = adapter.load(ds_root, split="train")

        if strategy == "iid":
            partitioner = IIDPartitioner()
            assignment = partitioner.partition(
                dataset.labels, num_clients, seed=seed, balanced=balanced
            )
        elif strategy == "dirichlet":
            partitioner = DirichletPartitioner()
            assignment = partitioner.partition(
                dataset.labels,
                num_clients,
                seed=seed,
                alpha=alpha or 0.5,
                min_samples=min_samples,
            )
        elif strategy == "shard":
            partitioner = ShardPartitioner()
            assignment = partitioner.partition(
                dataset.labels,
                num_clients,
                seed=seed,
                shards_per_client=shards_per_client,
            )
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")

        partition_dir = ds_root / "processed" / dataset_name / "partitions"
        partition_dir.mkdir(parents=True, exist_ok=True)
        np.save(partition_dir / f"{strategy}_seed{seed}.npy", assignment)

        actual_clients = len(assignment)
        dataset_registry.update_status(dataset_name, "partition_status", "partitioned")

        client_distributions = []
        for cid, indices in assignment.items():
            client_labels = dataset.labels[indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            client_distributions.append(
                {
                    "client_id": cid,
                    "num_samples": len(indices),
                    "class_distribution": {
                        str(u): int(c) for u, c in zip(unique, counts)
                    },
                }
            )

        meta = dataset_registry.get(dataset_name)
        meta["client_count"] = actual_clients
        dataset_registry.register(dataset_name, meta, force=True)

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "strategy": strategy,
            "num_clients": actual_clients,
            "client_distributions": client_distributions,
            "seed": seed,
        }

    def apply_missing_modality(
        self,
        dataset_name: str,
        strategy: str = "random",
        missing_ratio: float = 0.3,
        modalities: list[str] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        self._ensure_registered(dataset_name)
        adapter = self._get_adapter(dataset_name)
        ds_root = self._get_ds_root()
        dataset = adapter.load(ds_root, split="train")

        modified = self.missing_modality_simulator.apply(
            dataset,
            strategy=strategy,
            missing_ratio=missing_ratio,
            modalities=modalities,
            seed=seed,
        )

        missing_dir = ds_root / "processed" / dataset_name / "missing_modality"
        missing_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            missing_dir / f"{strategy}_ratio{missing_ratio}_seed{seed}.npy",
            np.array(modified.data, dtype=object),
        )

        meta = dataset_registry.get(dataset_name)
        meta["missing_modality_ratio"] = missing_ratio
        dataset_registry.register(dataset_name, meta, force=True)

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "strategy": strategy,
            "missing_ratio": missing_ratio,
            "num_samples_affected": sum(
                1
                for d in modified.data
                if any(f"{mod}_missing" in d for mod in adapter.modalities)
            ),
            "seed": seed,
        }

    def validate_dataset(self, name: str) -> dict[str, Any]:
        adapter = self._get_adapter(name)
        ds_root = self._get_ds_root()
        result = adapter.validate(ds_root)
        return {
            "status": "success" if result["is_valid"] else "error",
            "dataset_name": name,
            "is_valid": result["is_valid"],
            "errors": result["errors"],
            "warnings": result["warnings"],
            "class_distribution": result.get("info", {}),
        }

    def get_metadata(self, name: str) -> dict[str, Any]:
        ds_root = self._get_ds_root()
        meta = metadata_generator.load(
            name, ds_root / "processed" / name / "metadata.json"
        )
        if meta is None:
            adapter = self._get_adapter(name)
            meta = adapter.get_metadata()
        return meta

    def delete_dataset(self, name: str) -> None:
        if dataset_registry.exists(name):
            dataset_registry.unregister(name)
        ds_root = self._get_ds_root()
        for d in [
            ds_root / "raw" / name,
            ds_root / "processed" / name,
            ds_root / "sample" / name,
        ]:
            if d.exists():
                import shutil

                shutil.rmtree(d)
        dataset_cache.invalidate(f"metadata_{name}")

    def _register_builtins(self) -> None:
        builtins = ["uci_har", "meld", "ptb_xl", "hateful_memes", "generic"]
        for name in builtins:
            if not dataset_registry.exists(name):
                try:
                    adapter = self._get_adapter(name)
                    meta = adapter.get_metadata()
                    dataset_registry.register(
                        name,
                        {
                            "download_status": "not_downloaded",
                            "preprocessing_status": "not_preprocessed",
                            "partition_status": "not_partitioned",
                            "client_count": 0,
                            "missing_modality_ratio": 0.0,
                        },
                    )
                    saved = metadata_generator.generate(
                        name=name,
                        classes=meta.get("classes", []),
                        modalities=meta.get("modalities", []),
                        input_shapes=meta.get("input_shapes", {}),
                        num_samples=meta.get("num_samples", 0),
                    )
                    metadata_generator.save(saved)
                except Exception as e:
                    logger.warning(f"Could not register builtin {name}: {e}")

    def _ensure_registered(self, name: str) -> None:
        if not dataset_registry.exists(name):
            try:
                adapter = self._get_adapter(name)
                meta = adapter.get_metadata()
                dataset_registry.register(
                    name,
                    {
                        "download_status": "not_downloaded",
                        "preprocessing_status": "not_preprocessed",
                        "partition_status": "not_partitioned",
                        "client_count": 0,
                        "missing_modality_ratio": 0.0,
                    },
                )
            except DatasetNotFoundError:
                pass

    def _select_preprocessor(self, dataset_name: str, modalities: list[str]) -> Any:
        modality = modalities[0] if modalities else "image"
        if modality == "image":
            return ImagePreprocessor()
        elif modality == "text":
            return TextPreprocessor()
        elif modality == "audio":
            return AudioPreprocessor()
        elif modality == "sensor":
            return SensorPreprocessor()
        return None


dataset_service = DatasetService()
