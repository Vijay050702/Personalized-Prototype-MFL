from __future__ import annotations

import torch

from app.federated.models import ClientPrototypePackage


class DivergenceCalculator:
    def __init__(self, metric: str = "cosine"):
        if metric not in {"cosine", "euclidean", "manhattan"}:
            raise ValueError(
                f"Unsupported divergence metric '{metric}'. "
                f"Choose from: cosine, euclidean, manhattan"
            )
        self._metric = metric

    @property
    def metric(self) -> str:
        return self._metric

    def compute(
        self,
        package_a: ClientPrototypePackage,
        package_b: ClientPrototypePackage,
    ) -> float:
        if package_a.embedding_dim != package_b.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"{package_a.embedding_dim} vs {package_b.embedding_dim}"
            )
        if package_a.class_id != package_b.class_id:
            raise ValueError(
                f"Class ID mismatch: {package_a.class_id} vs {package_b.class_id}"
            )
        if package_a.modality != package_b.modality:
            raise ValueError(
                f"Modality mismatch: {package_a.modality} vs {package_b.modality}"
            )
        emb_a = package_a.to_tensor()
        emb_b = package_b.to_tensor()
        return self._compute_distance(emb_a, emb_b)

    def compute_from_tensors(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        return self._compute_distance(a, b)

    def _compute_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if self._metric == "cosine":
            sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
            return float((1.0 - sim).squeeze())
        elif self._metric == "euclidean":
            d = torch.nn.functional.pairwise_distance(a.unsqueeze(0), b.unsqueeze(0))
            return float(d.squeeze())
        elif self._metric == "manhattan":
            return float(torch.abs(a - b).sum())
        return 0.0

    def compute_pairwise(
        self,
        packages: list[ClientPrototypePackage],
    ) -> list[tuple[str, str, float]]:
        result: list[tuple[str, str, float]] = []
        for i in range(len(packages)):
            for j in range(i + 1, len(packages)):
                if (
                    packages[i].class_id == packages[j].class_id
                    and packages[i].modality == packages[j].modality
                ):
                    d = self.compute(packages[i], packages[j])
                    result.append((packages[i].client_id, packages[j].client_id, d))
        return result

    def batch_divergence(
        self,
        packages: list[ClientPrototypePackage],
        reference: ClientPrototypePackage,
    ) -> list[float]:
        return [
            self.compute(pkg, reference)
            for pkg in packages
            if pkg.class_id == reference.class_id and pkg.modality == reference.modality
        ]
