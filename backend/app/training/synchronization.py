from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from app.training.logger import TrainingLogger


class SynchronizationManager:
    def __init__(
        self,
        logger: TrainingLogger | None = None,
    ) -> None:
        self._logger = logger or TrainingLogger()
        self._sync_history: list[dict[str, Any]] = []

    def sync_model_state(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
    ) -> dict[str, Any]:
        source_state = source_model.state_dict()
        target_model.load_state_dict(source_state, strict=True)
        result = {
            "type": "model_state",
            "num_parameters": sum(p.numel() for p in source_model.parameters()),
        }
        self._sync_history.append(result)
        return result

    def sync_model_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        target_model: nn.Module,
    ) -> dict[str, Any]:
        target_model.load_state_dict(state_dict, strict=True)
        result = {
            "type": "model_state_dict",
            "num_keys": len(state_dict),
        }
        self._sync_history.append(result)
        return result

    def sync_prototype_repository(
        self,
        source_repo: Any,
        target_repo: Any,
    ) -> dict[str, Any]:
        if hasattr(source_repo, "export_state") and hasattr(
            target_repo, "import_state"
        ):
            state = source_repo.export_state()
            target_repo.import_state(state)
            result = {"type": "prototype_repository", "exported": True}
        elif hasattr(source_repo, "list_global_prototypes") and hasattr(
            target_repo, "store_global_prototype"
        ):
            prototypes = source_repo.list_global_prototypes()
            for proto in prototypes:
                target_repo.store_global_prototype(proto)
            result = {
                "type": "prototype_repository",
                "num_prototypes": len(prototypes),
            }
        else:
            raise ValueError("Source or target repository missing required methods")
        self._sync_history.append(result)
        return result

    def sync_knowledge_transfer_state(
        self,
        source_kt: Any,
        target_kt: Any,
    ) -> dict[str, Any]:
        if hasattr(source_kt, "mapper") and hasattr(target_kt, "mapper"):
            source_mapper = source_kt.mapper
            target_mapper = target_kt.mapper
            if hasattr(source_mapper, "available_mappings"):
                mappings = source_mapper.available_mappings()
                result = {
                    "type": "knowledge_transfer",
                    "num_mappings": len(mappings),
                }
            else:
                result = {"type": "knowledge_transfer", "num_mappings": 0}
        else:
            result = {"type": "knowledge_transfer", "num_mappings": 0}
        self._sync_history.append(result)
        return result

    def sync_personalized_prototypes(
        self,
        source_memory: Any,
        target_memory: Any,
    ) -> dict[str, Any]:
        if hasattr(source_memory, "statistics") and hasattr(target_memory, "store"):
            if hasattr(source_memory, "retrieve_all"):
                prototypes = source_memory.retrieve_all()
                for proto in prototypes:
                    target_memory.store(proto)
                result = {
                    "type": "personalized_prototypes",
                    "num_prototypes": len(prototypes),
                }
            else:
                result = {
                    "type": "personalized_prototypes",
                    "num_prototypes": 0,
                }
        else:
            result = {"type": "personalized_prototypes", "num_prototypes": 0}
        self._sync_history.append(result)
        return result

    def sync_optimizer_states(
        self,
        source_optimizer: torch.optim.Optimizer,
        target_optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        source_state = source_optimizer.state_dict()
        target_optimizer.load_state_dict(source_state)
        result = {
            "type": "optimizer_state",
            "num_groups": len(source_state.get("param_groups", [])),
        }
        self._sync_history.append(result)
        return result

    def sync_all(
        self,
        models: dict[str, nn.Module],
        repositories: dict[str, Any],
        optimizers: dict[str, torch.optim.Optimizer],
        knowledge_transfer: Any = None,
        personalization_memory: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        results: dict[str, list[dict[str, Any]]] = {}

        if "source" in models and "target" in models:
            results["model"] = [
                self.sync_model_state(models["source"], models["target"])
            ]

        if "source_repo" in repositories and "target_repo" in repositories:
            results["repository"] = [
                self.sync_prototype_repository(
                    repositories["source_repo"], repositories["target_repo"]
                )
            ]

        if "source_opt" in optimizers and "target_opt" in optimizers:
            results["optimizer"] = [
                self.sync_optimizer_states(
                    optimizers["source_opt"], optimizers["target_opt"]
                )
            ]

        if knowledge_transfer is not None:
            results["knowledge_transfer"] = [knowledge_transfer]

        if personalization_memory is not None:
            results["personalization"] = [personalization_memory]

        return results

    @property
    def sync_count(self) -> int:
        return len(self._sync_history)

    def history(self) -> list[dict[str, Any]]:
        return list(self._sync_history)

    def clear(self) -> None:
        self._sync_history.clear()
