from __future__ import annotations

import torch

from app.knowledge_transfer.validation import validate_shape_match


class Similarity:
    def __init__(self, metric: str = "cosine"):
        if metric not in {"cosine", "euclidean", "dot"}:
            raise ValueError(
                f"Unsupported similarity metric '{metric}'. "
                f"Choose from: cosine, euclidean, dot"
            )
        self._metric = metric

    @property
    def metric(self) -> str:
        return self._metric

    def compute(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if a.shape != b.shape:
            validate_shape_match(a, b)
        a_2d = a.unsqueeze(0) if a.dim() == 1 else a
        b_2d = b.unsqueeze(0) if b.dim() == 1 else b
        if self._metric == "cosine":
            result = torch.nn.functional.cosine_similarity(a_2d, b_2d)
        elif self._metric == "euclidean":
            dist = torch.nn.functional.pairwise_distance(a_2d, b_2d)
            result = 1.0 / (1.0 + dist)
        elif self._metric == "dot":
            result = (a_2d * b_2d).sum(dim=-1)
        else:
            return torch.tensor(0.0)
        if a.dim() == 1:
            result = result.squeeze(0)
        return result

    def pairwise(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if self._metric == "cosine":
            a_norm = a / (a.norm(p=2, dim=1, keepdim=True) + 1e-8)
            b_norm = b / (b.norm(p=2, dim=1, keepdim=True) + 1e-8)
            return a_norm @ b_norm.T
        elif self._metric == "euclidean":
            dist = torch.cdist(a, b, p=2.0)
            return 1.0 / (1.0 + dist)
        elif self._metric == "dot":
            return a @ b.T
        raise ValueError(f"Unsupported metric: {self._metric}")

    def to_config(self) -> dict[str, str]:
        return {"metric": self._metric}
