from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


class ModalityGraph:
    def __init__(self):
        self._edges: dict[str, set[str]] = defaultdict(set)
        self._embedding_dims: dict[str, int] = {}

    def add_modality(self, modality: str, embedding_dim: int = -1) -> None:
        if modality not in self._edges:
            self._edges[modality] = set()
        if embedding_dim > 0:
            self._embedding_dims[modality] = embedding_dim

    def add_mapping(self, source: str, target: str) -> None:
        self.add_modality(source)
        self.add_modality(target)
        self._edges[source].add(target)
        self._edges[target].add(source)

    def set_embedding_dim(self, modality: str, dim: int) -> None:
        self._embedding_dims[modality] = dim

    def get_embedding_dim(self, modality: str) -> int | None:
        return self._embedding_dims.get(modality)

    def has_modality(self, modality: str) -> bool:
        return modality in self._edges

    def modalities(self) -> list[str]:
        return sorted(self._edges.keys())

    def direct_neighbors(self, modality: str) -> list[str]:
        return sorted(self._edges.get(modality, set()))

    def is_directly_connected(self, source: str, target: str) -> bool:
        return target in self._edges.get(source, set())

    def find_path(self, source: str, target: str) -> list[str] | None:
        if source not in self._edges or target not in self._edges:
            return None
        visited: set[str] = set()
        queue: deque[tuple[str, list[str]]] = deque()
        queue.append((source, [source]))
        visited.add(source)
        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            for neighbor in self._edges.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def reachable_modalities(self, source: str) -> set[str]:
        if source not in self._edges:
            return set()
        visited: set[str] = set()
        queue: deque[str] = deque([source])
        visited.add(source)
        while queue:
            current = queue.popleft()
            for neighbor in self._edges.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        visited.discard(source)
        return visited

    def paths_to_missing(
        self, available: set[str], missing: set[str]
    ) -> dict[str, list[str] | None]:
        results: dict[str, list[str] | None] = {}
        for target in missing:
            best_path: list[str] | None = None
            for source in available:
                path = self.find_path(source, target)
                if path is not None:
                    if best_path is None or len(path) < len(best_path):
                        best_path = path
            results[target] = best_path
        return results

    def count_edges(self) -> int:
        return sum(len(v) for v in self._edges.values()) // 2

    def to_config(self) -> dict[str, Any]:
        edges_list: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for source, targets in self._edges.items():
            for target in targets:
                edge = tuple(sorted([source, target]))
                if edge not in seen:
                    seen.add(edge)
                    edges_list.append((source, target))
        return {
            "modalities": self.modalities(),
            "edges": edges_list,
            "embedding_dims": dict(self._embedding_dims),
        }
