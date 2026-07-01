from __future__ import annotations

from typing import Any

import numpy as np

from app.datasets.base import BasePartitioner


class DirichletPartitioner(BasePartitioner):
    def partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        seed: int = 42,
        alpha: float = 0.5,
        min_samples: int = 1,
        **kwargs: Any,
    ) -> dict[int, list[int]]:
        rng = np.random.default_rng(seed)
        num_classes = len(np.unique(labels))
        n_samples = len(labels)
        class_indices = {
            c: np.where(labels == c)[0].tolist() for c in range(num_classes)
        }
        client_indices: dict[int, list[int]] = {i: [] for i in range(num_clients)}

        for c in range(num_classes):
            class_size = len(class_indices[c])
            if class_size == 0:
                continue
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.maximum(proportions, 1e-6)
            proportions /= proportions.sum()
            proportions = (proportions * class_size).astype(int)
            diff = class_size - proportions.sum()
            if diff > 0:
                proportions[:diff] += 1
            elif diff < 0:
                proportions[0] += diff

            rng.shuffle(class_indices[c])
            start = 0
            for i in range(num_clients):
                end = start + proportions[i]
                if end > len(class_indices[c]):
                    end = len(class_indices[c])
                client_indices[i].extend(class_indices[c][start:end])
                start = end

        valid = [i for i in range(num_clients) if len(client_indices[i]) >= min_samples]
        return {i: client_indices[i] for i in valid}
