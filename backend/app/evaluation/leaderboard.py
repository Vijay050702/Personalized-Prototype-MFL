from __future__ import annotations

from typing import Any


class Leaderboard:
    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def add_entry(
        self,
        experiment_id: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "experiment_id": experiment_id,
            **metrics,
            **(metadata or {}),
        }
        self._entries.append(entry)

    def add_entries(
        self,
        entries: list[dict[str, Any]],
    ) -> None:
        for entry in entries:
            if "experiment_id" in entry:
                self.add_entry(
                    entry["experiment_id"],
                    {k: v for k, v in entry.items() if k != "experiment_id"},
                    {},
                )

    def rank_by(
        self,
        metric: str = "accuracy",
        ascending: bool = False,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        valid = [
            e
            for e in self._entries
            if metric in e and isinstance(e[metric], (int, float))
        ]
        sorted_entries = sorted(valid, key=lambda e: e[metric], reverse=not ascending)
        if top_k is not None:
            sorted_entries = sorted_entries[:top_k]
        ranked = []
        for i, entry in enumerate(sorted_entries):
            ranked.append({"rank": i + 1, **entry})
        return ranked

    def rank_by_accuracy(self, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.rank_by("accuracy", ascending=False, top_k=top_k)

    def rank_by_f1(self, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.rank_by("f1_score", ascending=False, top_k=top_k)

    def rank_by_communication_cost(
        self, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        return self.rank_by("communication_cost", ascending=True, top_k=top_k)

    def rank_by_training_time(self, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.rank_by("training_time", ascending=True, top_k=top_k)

    def rank_by_prototype_quality(
        self, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        return self.rank_by("prototype_fusion_quality", ascending=False, top_k=top_k)

    def rank_by_personalization_gain(
        self, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        return self.rank_by("personalization_gain", ascending=False, top_k=top_k)

    def get_best(self, metric: str = "accuracy") -> dict[str, Any] | None:
        ranked = self.rank_by(metric, top_k=1)
        return ranked[0] if ranked else None

    def get_worst(self, metric: str = "accuracy") -> dict[str, Any] | None:
        ranked = self.rank_by(metric, ascending=True, top_k=1)
        return ranked[0] if ranked else None

    def summary(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "experiments": [e.get("experiment_id", "unknown") for e in self._entries],
            "ranking_by_accuracy": self.rank_by_accuracy(),
            "best_accuracy": self.get_best("accuracy"),
            "best_f1": self.get_best("f1_score"),
            "best_personalization": self.get_best("personalization_gain"),
        }

    def clear(self) -> None:
        self._entries.clear()

    def remove_entry(self, experiment_id: str) -> None:
        self._entries = [
            e for e in self._entries if e.get("experiment_id") != experiment_id
        ]
