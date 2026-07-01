from __future__ import annotations

from typing import Any

import numpy as np

from app.datasets.base import BasePartitioner


class IIDPartitioner(BasePartitioner):
    def partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        seed: int = 42,
        balanced: bool = True,
        **kwargs: Any,
    ) -> dict[int, list[int]]:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(labels))
        rng.shuffle(indices)

        if balanced:
            splits = np.array_split(indices, num_clients)
        else:
            rng = np.random.default_rng(seed + 1)
            proportions = rng.dirichlet(np.ones(num_clients) * 0.5)
            cumsum = np.concatenate(([0], np.cumsum(proportions)))
            split_pts = (cumsum[1:-1] * len(indices)).astype(int)
            splits = np.split(indices, split_pts)

        return {i: list(splits[i]) for i in range(num_clients)}
