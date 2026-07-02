from __future__ import annotations

from collections import Counter
from typing import Any

from app.federated.models import ClientPrototypePackage, ModalityCompletenessReport


class CompletenessScorer:
    def __init__(self, expected_modalities: list[str] | None = None):
        self._expected_modalities: list[str] = (
            sorted(expected_modalities) if expected_modalities else []
        )
        self._running_counts: Counter[str] = Counter()

    @property
    def expected_modalities(self) -> list[str]:
        return list(self._expected_modalities)

    @expected_modalities.setter
    def expected_modalities(self, modalities: list[str]) -> None:
        self._expected_modalities = sorted(modalities)

    def score_package(self, package: ClientPrototypePackage) -> float:
        return 1.0

    def score_client_packages(
        self, packages: list[ClientPrototypePackage]
    ) -> ModalityCompletenessReport:
        if not self._expected_modalities:
            modalities_present = sorted({p.modality for p in packages})
            self._expected_modalities = modalities_present

        modalities_present: set[str] = set()
        for pkg in packages:
            modalities_present.add(pkg.modality)
            self._running_counts[pkg.modality] += 1

        available = sorted(modalities_present)
        missing = sorted(set(self._expected_modalities) - modalities_present)
        total = len(self._expected_modalities) if self._expected_modalities else 1
        ratio = len(available) / total if total > 0 else 0.0

        return ModalityCompletenessReport(
            available_modalities=available,
            missing_modalities=missing,
            total_possible=total,
            completeness_ratio=ratio,
        )

    def score_modality_set(self, modalities: set[str]) -> ModalityCompletenessReport:
        if not self._expected_modalities:
            available = sorted(modalities)
            return ModalityCompletenessReport(
                available_modalities=available,
                missing_modalities=[],
                total_possible=len(available),
                completeness_ratio=1.0,
            )
        available = sorted(modalities)
        missing = sorted(set(self._expected_modalities) - modalities)
        total = len(self._expected_modalities)
        ratio = len(available) / total if total > 0 else 0.0
        return ModalityCompletenessReport(
            available_modalities=available,
            missing_modalities=missing,
            total_possible=total,
            completeness_ratio=ratio,
        )

    def running_statistics(self) -> dict[str, Any]:
        return {
            "expected_modalities": self._expected_modalities,
            "modality_counts": dict(self._running_counts),
            "total_packages_seen": sum(self._running_counts.values()),
        }

    def reset(self) -> None:
        self._running_counts.clear()

    def client_completeness_scores(
        self,
        client_packages: dict[str, list[ClientPrototypePackage]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for client_id, packages in client_packages.items():
            report = self.score_client_packages(packages)
            scores[client_id] = report.completeness_ratio
        return scores
