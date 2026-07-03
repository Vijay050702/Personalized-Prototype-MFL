from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.utils.data

from app.core.logging import logger
from app.federated.models import ClientPrototypePackage
from app.knowledge_transfer.inference import InferenceEngine, InferenceOutput
from app.personalization.adaptation import AdaptationEngine
from app.personalization.fusion_engine import FusionEngine
from app.personalization.personalized_memory import PersonalizedMemory
from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.profile import ClientProfile
from app.personalization.prototype_selector import PrototypeSelector
from app.prototypes.generator import PrototypeGenerator as LocalProtoGenerator
from app.prototypes.memory import PrototypeMemory
from app.prototypes.prototype import Prototype
from app.training.local_training import LocalTraining
from app.training.logger import TrainingLogger
from app.training.state import ClientState
from app.training.utils import Timer


class Client:
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        local_training: LocalTraining | None = None,
        prototype_generator: LocalProtoGenerator | None = None,
        prototype_memory: PrototypeMemory | None = None,
        prototype_selector: PrototypeSelector | None = None,
        fusion_engine: FusionEngine | None = None,
        adaptation_engine: AdaptationEngine | None = None,
        inference_engine: InferenceEngine | None = None,
        personalized_memory: PersonalizedMemory | None = None,
        profile: ClientProfile | None = None,
        device: torch.device | None = None,
        logger_instance: TrainingLogger | None = None,
    ) -> None:
        self._client_id = client_id
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._local_training = local_training or LocalTraining(
            model=model, loss_fn=loss_fn, optimizer=optimizer, device=device
        )
        self._prototype_generator = prototype_generator
        self._prototype_memory = prototype_memory or PrototypeMemory()
        self._prototype_selector = prototype_selector
        self._fusion_engine = fusion_engine
        self._adaptation_engine = adaptation_engine
        self._inference_engine = inference_engine
        self._personalized_memory = personalized_memory or PersonalizedMemory()
        self._profile = profile or ClientProfile(client_id=client_id)
        self._device = device or torch.device("cpu")
        self._logger = logger_instance or TrainingLogger()
        self._state = ClientState(client_id=client_id)
        self._timer = Timer()
        self._latest_global_prototypes: list[Any] = []
        self._latest_transferred: list[InferenceOutput] = []

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def state(self) -> ClientState:
        return self._state

    @property
    def profile(self) -> ClientProfile:
        return self._profile

    @property
    def personalized_memory(self) -> PersonalizedMemory:
        return self._personalized_memory

    @property
    def prototype_memory(self) -> PrototypeMemory:
        return self._prototype_memory

    def load_local_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        modalities: set[str] | None = None,
        all_modalities: set[str] | None = None,
    ) -> None:
        self._dataloader = dataloader
        if modalities is not None:
            self._profile.update_modalities(
                available=modalities,
                all_modalities=all_modalities or modalities,
            )

    def train(
        self,
        epochs: int = 1,
    ) -> dict[str, Any]:
        self._timer.start()
        if not hasattr(self, "_dataloader") or self._dataloader is None:
            raise ValueError("No dataset loaded. Call load_local_dataset first.")

        results = self._local_training.train(self._dataloader, epochs=epochs)
        self._state.epochs_completed += epochs
        self._state.samples_processed += len(self._dataloader.dataset) * epochs
        self._state.loss_history.append(results.get("final_loss", 0.0))
        self._state.accuracy_history.append(results.get("final_accuracy", 0.0))
        self._profile.record_training_step()

        training_time = self._timer.stop()
        self._logger.log_client_update(
            client_id=self._client_id,
            round_id=self._state.current_round,
            loss=results.get("final_loss"),
            accuracy=results.get("final_accuracy"),
            num_samples=len(self._dataloader.dataset) * epochs,
        )

        return {
            **results,
            "training_time": training_time,
            "client_id": self._client_id,
        }

    def generate_prototypes(
        self,
        dataloader: torch.utils.data.DataLoader | None = None,
    ) -> list[Prototype]:
        loader = dataloader or getattr(self, "_dataloader", None)
        if loader is None:
            raise ValueError("No dataset available for prototype generation")

        if self._prototype_generator is None:
            raise ValueError("PrototypeGenerator not configured")

        embeddings_by_class = self._local_training.get_embeddings(loader)
        prototypes: list[Prototype] = []

        for class_id, emb_list in embeddings_by_class.items():
            if not emb_list:
                continue
            embeddings_t = torch.stack(emb_list)
            for modality in self._profile.available_modalities:
                proto = self._prototype_generator.generate_from_embeddings(
                    embeddings=embeddings_t,
                    labels=torch.full(
                        (embeddings_t.size(0),), class_id, dtype=torch.long
                    ),
                    class_id=class_id,
                    modality=modality,
                )
                self._prototype_memory.store_local(proto)
                prototypes.append(proto)

        self._logger.log_client_update(
            client_id=self._client_id,
            round_id=self._state.current_round,
            loss=None,
        )
        return prototypes

    def upload_results(
        self,
        round_id: int,
    ) -> list[ClientPrototypePackage]:
        packages: list[ClientPrototypePackage] = []
        local_protos = self._prototype_memory.local_repo.list()

        for proto in local_protos:
            package = ClientPrototypePackage(
                client_id=self._client_id,
                round_id=round_id,
                modality=proto.modality,
                class_id=proto.class_id,
                prototype_vector=proto.embedding.detach().cpu().tolist(),
                sample_count=proto.sample_count,
                embedding_dim=proto.embedding.size(-1),
                confidence=proto.confidence,
            )
            packages.append(package)

        self._state.current_round = round_id
        return packages

    def receive_updates(
        self,
        global_prototypes: list[Any],
        transferred: list[InferenceOutput] | None = None,
    ) -> None:
        self._latest_global_prototypes = global_prototypes
        self._latest_transferred = transferred or []

    def personalize(
        self,
        local_prototypes: list[Prototype] | None = None,
        global_prototypes: list[Any] | None = None,
        transferred: list[InferenceOutput] | None = None,
    ) -> list[PersonalizedPrototype]:
        if self._prototype_selector is None:
            raise ValueError("PrototypeSelector not configured")
        if self._fusion_engine is None:
            raise ValueError("FusionEngine not configured")

        local = local_prototypes or self._prototype_memory.local_repo.list()
        global_p = global_prototypes or self._latest_global_prototypes
        trans = transferred or self._latest_transferred

        all_classes = self._collect_class_ids(local, global_p, trans)
        all_modalities = (
            self._profile.available_modalities | (self._profile.missing_modalities)
            if self._profile.missing_modalities
            else self._profile.available_modalities
        )

        personalized_list: list[PersonalizedPrototype] = []
        for class_id in all_classes:
            for modality in sorted(all_modalities):
                pp = self._prototype_selector.select_sources(
                    local_prototypes=local,
                    global_prototypes=global_p,
                    transferred=trans,
                    class_id=class_id,
                    modality=modality,
                    client_profile=self._profile,
                )
                if pp.available_sources():
                    pp = self._fusion_engine.fuse(pp, client_profile=self._profile)
                    if self._adaptation_engine is not None:
                        pp = self._adaptation_engine.adapt(pp, self._profile)
                    self._personalized_memory.store(pp)
                    personalized_list.append(pp)

        self._logger.log_client_update(
            client_id=self._client_id,
            round_id=self._state.current_round,
            loss=None,
        )
        return personalized_list

    def _collect_class_ids(
        self,
        local: list[Prototype],
        global_p: list[Any],
        transferred: list[InferenceOutput],
    ) -> set[int]:
        class_ids: set[int] = set()
        for p in local:
            class_ids.add(p.class_id)
        for p in global_p:
            class_ids.add(p.class_id if hasattr(p, "class_id") else 0)
        for p in transferred:
            class_ids.add(p.class_id)
        return class_ids

    def update_model(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        self._model.load_state_dict(state_dict, strict=True)

    def get_model_state(self) -> dict[str, torch.Tensor]:
        return self._model.state_dict()

    def get_optimizer_state(self) -> dict[str, Any]:
        return self._optimizer.state_dict()

    def set_optimizer_state(self, state_dict: dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)

    def set_current_round(self, round_id: int) -> None:
        self._state.current_round = round_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self._client_id,
            "state": self._state.to_dict(),
            "profile": self._profile.to_dict(),
            "prototype_count": self._prototype_memory.local_repo.size,
            "personalized_count": len(self._personalized_memory.retrieve_all()),
        }
