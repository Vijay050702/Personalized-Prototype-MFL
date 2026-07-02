from __future__ import annotations

from typing import Any

import torch

from app.prototypes.prototype import Prototype


class PrototypeClustering:
    def __init__(self, strategy: str = "kmeans"):
        self._strategy = strategy

    def cluster(
        self,
        embeddings: torch.Tensor,
        n_clusters: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self._strategy == "kmeans":
            return self._kmeans(embeddings, n_clusters, **kwargs)
        elif self._strategy == "hierarchical":
            return self._hierarchical(embeddings, n_clusters, **kwargs)
        elif self._strategy == "dbscan":
            return self._dbscan(embeddings, **kwargs)
        else:
            raise ValueError(f"Unknown clustering strategy: {self._strategy}")

    def _kmeans(
        self,
        embeddings: torch.Tensor,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> dict[str, Any]:
        n_samples = embeddings.size(0)
        indices = torch.randperm(n_samples)[:n_clusters]
        centroids = embeddings[indices].clone()

        for _ in range(max_iter):
            dists = torch.cdist(embeddings, centroids, p=2.0)
            assignments = torch.argmin(dists, dim=1)
            new_centroids = torch.stack(
                [embeddings[assignments == k].mean(dim=0) for k in range(n_clusters)]
            )
            if new_centroids.shape != centroids.shape:
                break
            shift = (new_centroids - centroids).norm().item()
            centroids = new_centroids
            if shift < tol:
                break

        return {
            "centroids": centroids,
            "assignments": assignments.tolist(),
            "n_clusters": n_clusters,
            "strategy": "kmeans",
            "inertia": float(
                sum(
                    ((embeddings[assignments == k] - centroids[k]) ** 2).sum()
                    for k in range(n_clusters)
                    if (assignments == k).sum() > 0
                )
            ),
        }

    def _hierarchical(
        self,
        embeddings: torch.Tensor,
        n_clusters: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        n = embeddings.size(0)
        if n <= n_clusters:
            return {
                "centroids": embeddings.clone(),
                "assignments": list(range(n)),
                "n_clusters": n,
                "strategy": "hierarchical",
            }

        dist_matrix = torch.cdist(embeddings, embeddings, p=2.0)
        clusters = {i: [i] for i in range(n)}
        current_dist = dist_matrix.clone()
        current_dist.fill_diagonal_(float("inf"))

        while len(clusters) > n_clusters:
            flat_idx = torch.argmin(current_dist).item()
            i = flat_idx // n
            j = flat_idx % n
            if i > j:
                i, j = j, i
            clusters[i].extend(clusters[j])
            del clusters[j]
            for k in list(clusters.keys()):
                if k == i:
                    continue
                d_ik = current_dist[i, k].item() if i < n and k < n else float("inf")
                d_jk = current_dist[j, k].item() if j < n and k < n else float("inf")
                if k < n and i < n:
                    current_dist[i, k] = min(d_ik, d_jk)
                    current_dist[k, i] = min(d_ik, d_jk)
            current_dist[j, :] = float("inf")
            current_dist[:, j] = float("inf")
            if len(clusters) <= n_clusters:
                break

        centroids = []
        assignments = [0] * n
        for cluster_idx, (orig_idx, members) in enumerate(clusters.items()):
            centroids.append(embeddings[members].mean(dim=0))
            for m in members:
                assignments[m] = cluster_idx

        return {
            "centroids": torch.stack(centroids)
            if centroids
            else torch.empty(0, embeddings.size(1)),
            "assignments": assignments,
            "n_clusters": len(clusters),
            "strategy": "hierarchical",
        }

    def _dbscan(
        self,
        embeddings: torch.Tensor,
        eps: float = 0.5,
        min_samples: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        n = embeddings.size(0)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2.0)
        labels = torch.full((n,), -1, dtype=torch.long)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue
            neighbors = (dist_matrix[i] < eps).nonzero(as_tuple=True)[0]
            if neighbors.size(0) < min_samples:
                labels[i] = 0
                continue

            labels[i] = cluster_id
            seed_set = neighbors[neighbors != i].tolist()

            while seed_set:
                q = seed_set.pop()
                if labels[q] == 0:
                    labels[q] = cluster_id
                if labels[q] != -1:
                    continue
                labels[q] = cluster_id
                q_neighbors = (dist_matrix[q] < eps).nonzero(as_tuple=True)[0]
                if q_neighbors.size(0) >= min_samples:
                    seed_set.extend(
                        n.item() for n in q_neighbors if labels[n.item()] < 1
                    )
            cluster_id += 1

        unique_labels = sorted(set(labels.tolist()))
        centroids = []
        for cid in unique_labels:
            if cid <= 0:
                continue
            members = (labels == cid).nonzero(as_tuple=True)[0]
            if members.size(0) > 0:
                centroids.append(embeddings[members].mean(dim=0))

        return {
            "centroids": torch.stack(centroids)
            if centroids
            else torch.empty(0, embeddings.size(1)),
            "assignments": labels.tolist(),
            "n_clusters": len(centroids),
            "strategy": "dbscan",
        }

    def cluster_prototypes(
        self,
        prototypes: list[Prototype],
        n_clusters: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any]:
        embeddings = torch.stack([p.embedding for p in prototypes], dim=0)
        result = self.cluster(embeddings, n_clusters=n_clusters, **kwargs)
        result["prototype_ids"] = [p.prototype_id for p in prototypes]
        return result
