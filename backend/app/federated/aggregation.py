from __future__ import annotations

from typing import Any

import torch

from app.federated.models import (
    AggregatedPrototype,
    ClientPrototypePackage,
    WeightedPrototype,
)


class PrototypeAggregator:
    def __init__(self, epsilon: float = 1e-8):
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon

    def aggregate_weighted(
        self,
        packages: list[ClientPrototypePackage],
        weights: list[float],
    ) -> AggregatedPrototype:
        if not packages:
            raise ValueError("No packages to aggregate")
        if len(packages) != len(weights):
            raise ValueError(
                f"Number of packages ({len(packages)}) must match "
                f"number of weights ({len(weights)})"
            )

        self._validate_packages_compatible(packages)

        class_id = packages[0].class_id
        modality = packages[0].modality

        tensors = torch.stack([p.to_tensor() for p in packages])
        weight_t = torch.tensor(weights, dtype=torch.float32)
        weight_sum = weight_t.sum()
        if weight_sum < self._epsilon:
            weight_t = torch.ones_like(weight_t) / len(weight_t)
        else:
            weight_t = weight_t / weight_sum

        aggregated = (tensors * weight_t.unsqueeze(1)).sum(dim=0)

        total_samples = sum(p.sample_count for p in packages)
        variances = tensors.var(dim=0).mean().item()
        confidence = float(
            torch.sigmoid(
                (aggregated.norm(p=2) / (len(aggregated) ** 0.5)) * 5.0 - 2.5
            ).squeeze()
        )

        return AggregatedPrototype(
            class_id=class_id,
            modality=modality,
            prototype_vector=aggregated.detach().cpu().tolist(),
            embedding_dim=aggregated.size(0),
            sample_count=total_samples,
            confidence=confidence,
            variance=variances,
            num_contributors=len(packages),
        )

    def aggregate_simple(
        self,
        packages: list[ClientPrototypePackage],
    ) -> AggregatedPrototype:
        n = len(packages)
        equal_weights = [1.0 / n] * n if n > 0 else []
        return self.aggregate_weighted(packages, equal_weights)

    def aggregate_by_client_weights(
        self,
        packages_by_client: dict[str, list[ClientPrototypePackage]],
        client_weights: dict[str, float],
    ) -> AggregatedPrototype:
        all_packages: list[ClientPrototypePackage] = []
        all_weights: list[float] = []

        for client_id, pkgs in packages_by_client.items():
            cw = client_weights.get(client_id, 1.0)
            per_pkg_weight = cw / len(pkgs) if pkgs else 0.0
            for pkg in pkgs:
                all_packages.append(pkg)
                all_weights.append(per_pkg_weight)

        return self.aggregate_weighted(all_packages, all_weights)

    def per_class_aggregation(
        self,
        packages: list[ClientPrototypePackage],
        weights: list[float],
    ) -> dict[tuple[int, str], AggregatedPrototype]:
        grouped: dict[
            tuple[int, str], tuple[list[ClientPrototypePackage], list[float]]
        ] = {}

        for pkg, w in zip(packages, weights):
            key = (pkg.class_id, pkg.modality)
            if key not in grouped:
                grouped[key] = ([], [])
            grouped[key][0].append(pkg)
            grouped[key][1].append(w)

        result: dict[tuple[int, str], AggregatedPrototype] = {}
        for key, (pkgs, wts) in grouped.items():
            result[key] = self.aggregate_weighted(pkgs, wts)
        return result

    def _validate_packages_compatible(
        self, packages: list[ClientPrototypePackage]
    ) -> None:
        if not packages:
            return
        class_id = packages[0].class_id
        modality = packages[0].modality
        dim = packages[0].embedding_dim
        for pkg in packages:
            if pkg.class_id != class_id:
                raise ValueError(
                    f"Mixed class_ids in aggregation: {class_id} vs {pkg.class_id}"
                )
            if pkg.modality != modality:
                raise ValueError(
                    f"Mixed modalities in aggregation: {modality} vs {pkg.modality}"
                )
            if pkg.embedding_dim != dim:
                raise ValueError(
                    f"Mixed embedding dimensions: {dim} vs {pkg.embedding_dim}"
                )
