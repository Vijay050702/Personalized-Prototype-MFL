from __future__ import annotations

from typing import Any

import torch

from app.knowledge_transfer.similarity import Similarity
from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.profile import ClientProfile


class PersonalizationMetrics:
    def __init__(self, similarity_metric: str = "cosine"):
        self._similarity = Similarity(metric=similarity_metric)

    def personalization_gain(
        self,
        personalized: torch.Tensor,
        global_prototype: torch.Tensor,
    ) -> float:
        sim = self._similarity.compute(personalized, global_prototype)
        return (1.0 - sim).item()

    def prototype_drift(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> float:
        return torch.norm(current - previous, p=2).item()

    def alignment_score(
        self,
        personalized: torch.Tensor,
        local: torch.Tensor | None = None,
        global_p: torch.Tensor | None = None,
    ) -> float:
        scores: list[float] = []
        if local is not None:
            sim = self._similarity.compute(personalized, local)
            scores.append(sim.item())
        if global_p is not None:
            sim = self._similarity.compute(personalized, global_p)
            scores.append(sim.item())
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def fusion_quality(
        self,
        personalized: PersonalizedPrototype,
    ) -> float:
        weights = personalized.fusion_weights
        if not weights:
            return 0.0
        values = list(weights.values())
        if len(values) <= 1:
            return 1.0
        vals_t = torch.tensor(values, dtype=torch.float32)
        entropy = -(vals_t * torch.log(vals_t + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(len(values), dtype=torch.float32))
        if max_entropy == 0:
            return 1.0
        return 1.0 - (entropy / max_entropy).item()

    def client_diversity(
        self,
        prototypes: list[PersonalizedPrototype],
    ) -> float:
        if len(prototypes) < 2:
            return 0.0
        similarities: list[float] = []
        for i in range(len(prototypes)):
            p1 = prototypes[i]
            if p1.personalized_prototype is None:
                continue
            e1 = torch.tensor(p1.personalized_prototype, dtype=torch.float32)
            for j in range(i + 1, len(prototypes)):
                p2 = prototypes[j]
                if p2.personalized_prototype is None:
                    continue
                e2 = torch.tensor(p2.personalized_prototype, dtype=torch.float32)
                sim = self._similarity.compute(e1, e2)
                similarities.append(sim.item())
        if not similarities:
            return 0.0
        return 1.0 - (sum(similarities) / len(similarities))

    def confidence_trend(
        self,
        profile: ClientProfile,
    ) -> float:
        return profile.confidence_trend

    def prototype_stability(
        self,
        history: list[float],
    ) -> float:
        if len(history) < 2:
            return 1.0
        vals = torch.tensor(history, dtype=torch.float32)
        variance = vals.var().item()
        return 1.0 / (1.0 + variance)

    def compute_all(
        self,
        personalized_prototypes: list[PersonalizedPrototype],
        profiles: dict[str, ClientProfile] | None = None,
    ) -> dict[str, float]:
        if not personalized_prototypes:
            return {
                "personalization_gain": 0.0,
                "prototype_drift": 0.0,
                "alignment_score": 0.0,
                "fusion_quality": 0.0,
                "client_diversity": 0.0,
                "confidence_trend": 0.0,
                "prototype_stability": 0.0,
            }

        total_gain = 0.0
        total_alignment = 0.0
        total_quality = 0.0
        gain_count = 0
        align_count = 0

        for pp in personalized_prototypes:
            if (
                pp.personalized_prototype is not None
                and pp.global_prototype is not None
            ):
                pers = torch.tensor(pp.personalized_prototype, dtype=torch.float32)
                glob = torch.tensor(pp.global_prototype, dtype=torch.float32)
                total_gain += self.personalization_gain(pers, glob)
                gain_count += 1

            if pp.personalized_prototype is not None:
                pers = torch.tensor(pp.personalized_prototype, dtype=torch.float32)
                loc = (
                    torch.tensor(pp.local_prototype, dtype=torch.float32)
                    if pp.local_prototype is not None
                    else None
                )
                glob = (
                    torch.tensor(pp.global_prototype, dtype=torch.float32)
                    if pp.global_prototype is not None
                    else None
                )
                total_alignment += self.alignment_score(pers, loc, glob)
                align_count += 1

            total_quality += self.fusion_quality(pp)

        n = len(personalized_prototypes)
        diversity = self.client_diversity(personalized_prototypes)

        trend = 0.0
        stability = 0.0
        if profiles:
            confs = [p.average_confidence for p in profiles.values()]
            if confs:
                trend = sum(prof.confidence_trend for prof in profiles.values()) / len(
                    profiles
                )

                all_hist: list[float] = []
                for prof in profiles.values():
                    all_hist.extend(e["confidence"] for e in prof.confidence_history)
                if all_hist:
                    stability = self.prototype_stability(all_hist)

        return {
            "personalization_gain": total_gain / max(gain_count, 1),
            "prototype_drift": 0.0,
            "alignment_score": total_alignment / max(align_count, 1),
            "fusion_quality": total_quality / n,
            "client_diversity": diversity,
            "confidence_trend": trend,
            "prototype_stability": stability,
        }
