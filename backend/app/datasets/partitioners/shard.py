from __future__ import annotations

from typing import Any

import numpy as np

from app.datasets.base import BasePartitioner


class ShardPartitioner(BasePartitioner):
    def partition(
        self,
        labels: np.ndarray,
        num_clients: int,
        seed: int = 42,
        shards_per_client: int = 2,
        **kwargs: Any,
    ) -> dict[int, list[int]]:
        rng = np.random.default_rng(seed)
        num_classes = len(np.unique(labels))
        class_indices = {
            c: np.where(labels == c)[0].tolist() for c in range(num_classes)
        }
        for c in range(num_classes):
            rng.shuffle(class_indices[c])

        shards: list[list[int]] = []
        for c in range(num_classes):
            n = len(class_indices[c])
            num_shards = num_clients * shards_per_client
            shard_size = max(1, n // num_shards)
            for i in range(0, n, shard_size):
                shards.append(class_indices[c][i : i + shard_size])

        rng.shuffle(shards)
        total_shards_needed = num_clients * shards_per_client
        shards = shards[:total_shards_needed]

        while len(shards) < total_shards_needed:
            shards.append([])

        rng.shuffle(shards)

        client_indices: dict[int, list[int]] = {}
        for i in range(num_clients):
            client_indices[i] = []
            for j in range(shards_per_client):
                idx = i * shards_per_client + j
                if idx < len(shards):
                    client_indices[i].extend(shards[idx])

        return client_indices
