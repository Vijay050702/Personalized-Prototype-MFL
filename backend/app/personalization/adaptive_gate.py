from __future__ import annotations

from typing import Any

import torch

from app.knowledge_transfer.similarity import Similarity
from app.personalization.gating_network import GatingNetwork


class AdaptiveGate:
    def __init__(
        self,
        gating_network: GatingNetwork,
        similarity_metric: str = "cosine",
    ):
        self._gating_network = gating_network
        self._similarity = Similarity(metric=similarity_metric)

    def compute_weights(
        self,
        source_embeddings: dict[str, torch.Tensor],
        global_confidence: float | None = None,
        profile_features: list[float] | None = None,
    ) -> dict[str, float]:
        source_names = sorted(source_embeddings.keys())
        features: list[torch.Tensor] = []

        for name in source_names:
            emb = source_embeddings[name]
            features.append(emb.flatten())

        for i in range(len(source_names)):
            for j in range(i + 1, len(source_names)):
                s = self._similarity.compute(
                    source_embeddings[source_names[i]],
                    source_embeddings[source_names[j]],
                )
                features.append(s.flatten())

        if global_confidence is not None:
            features.append(torch.tensor([global_confidence]))

        if profile_features:
            features.append(torch.tensor(profile_features, dtype=torch.float32))

        if not features:
            n = len(source_names)
            return {name: 1.0 / n for name in source_names}

        gate_input = torch.cat(features)
        weights_t = self._gating_network(gate_input.unsqueeze(0)).squeeze(0)

        result: dict[str, float] = {}
        for i, name in enumerate(source_names):
            result[name] = weights_t[i].item()

        return result

    @property
    def gating_network(self) -> GatingNetwork:
        return self._gating_network

    def to_config(self) -> dict[str, Any]:
        return {
            "gating_network": self._gating_network.to_config(),
            "similarity_metric": self._similarity.metric,
        }
