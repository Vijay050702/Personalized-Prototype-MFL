from __future__ import annotations

from typing import Any

import torch

from app.prototypes.prototype import Prototype


class PrototypeMetrics:
    def __init__(self, prototypes: list[Prototype]):
        self._prototypes = prototypes

    def intra_class_distance(self, class_id: int) -> float:
        class_protos = [p for p in self._prototypes if p.class_id == class_id]
        if len(class_protos) < 2:
            return 0.0
        embeds = torch.stack([p.embedding for p in class_protos])
        dists = torch.cdist(embeds, embeds, p=2.0)
        mask = 1.0 - torch.eye(len(class_protos))
        return float((dists * mask).sum() / mask.sum())

    def inter_class_distance(self, class_a: int, class_b: int) -> float:
        protos_a = [p for p in self._prototypes if p.class_id == class_a]
        protos_b = [p for p in self._prototypes if p.class_id == class_b]
        if not protos_a or not protos_b:
            return 0.0
        embeds_a = torch.stack([p.embedding for p in protos_a])
        embeds_b = torch.stack([p.embedding for p in protos_b])
        dists = torch.cdist(embeds_a, embeds_b, p=2.0)
        return float(dists.mean())

    def prototype_purity(self, class_id: int) -> float:
        class_protos = [p for p in self._prototypes if p.class_id == class_id]
        if not class_protos:
            return 0.0
        if len(class_protos) == 1:
            return 1.0
        embeds = torch.stack([p.embedding for p in class_protos])
        sims = embeds @ embeds.T
        mask = 1.0 - torch.eye(len(class_protos), device=sims.device)
        return float((sims * mask).sum() / mask.sum())

    def prototype_coverage(self) -> dict[int, float]:
        class_counts: dict[int, int] = {}
        for p in self._prototypes:
            class_counts[p.class_id] = class_counts.get(p.class_id, 0) + 1
        if not class_counts:
            return {}
        max_count = max(class_counts.values())
        return {cid: count / max_count for cid, count in class_counts.items()}

    def prototype_variance(self) -> dict[int, float]:
        from collections import defaultdict

        class_embeds: dict[int, list[torch.Tensor]] = defaultdict(list)
        for p in self._prototypes:
            class_embeds[p.class_id].append(p.embedding)
        variances = {}
        for cid, embs in class_embeds.items():
            if len(embs) >= 2:
                stacked = torch.stack(embs)
                variances[cid] = float(stacked.var(dim=0).mean())
            else:
                variances[cid] = 0.0
        return variances

    def prototype_drift(
        self,
        old_prototypes: list[Prototype],
        new_prototypes: list[Prototype],
    ) -> float:
        old_map = {p.prototype_id: p for p in old_prototypes}
        new_map = {p.prototype_id: p for p in new_prototypes}
        common = set(old_map.keys()) & set(new_map.keys())
        if not common:
            return 0.0
        drifts = []
        for pid in common:
            drift = (old_map[pid].embedding - new_map[pid].embedding).norm().item()
            drifts.append(drift)
        return float(torch.tensor(drifts).mean())

    def average_confidence(self) -> float:
        if not self._prototypes:
            return 0.0
        confs = [p.confidence for p in self._prototypes]
        if not confs:
            return 0.0
        return sum(confs) / len(confs)

    def to_dict(self) -> dict[str, Any]:
        class_ids = sorted(set(p.class_id for p in self._prototypes))
        inter_intra = {}
        for cid in class_ids:
            inter_intra[cid] = {
                "intra_class_distance": self.intra_class_distance(cid),
                "purity": self.prototype_purity(cid),
            }
        return {
            "num_prototypes": len(self._prototypes),
            "num_classes": len(class_ids),
            "average_confidence": self.average_confidence(),
            "prototype_coverage": self.prototype_coverage(),
            "prototype_variance": self.prototype_variance(),
            "per_class": inter_intra,
        }
