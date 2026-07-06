from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


class StatisticalAnalysis:
    @staticmethod
    def mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.mean(values))

    @staticmethod
    def median(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.median(values))

    @staticmethod
    def variance(values: list[float], ddof: int = 1) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.var(values, ddof=ddof))

    @staticmethod
    def std(values: list[float], ddof: int = 1) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=ddof))

    @staticmethod
    def confidence_interval(
        values: list[float],
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        if len(values) < 2:
            return (0.0, 0.0)
        mean_val = float(np.mean(values))
        se = float(np.std(values, ddof=1)) / math.sqrt(len(values))
        h = se * scipy_stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        return (mean_val - h, mean_val + h)

    @staticmethod
    def paired_ttest(
        before: list[float],
        after: list[float],
    ) -> dict[str, float]:
        if len(before) != len(after):
            n = min(len(before), len(after))
            before = before[:n]
            after = after[:n]
        if len(before) < 2:
            return {"t_statistic": 0.0, "p_value": 1.0, "degrees_of_freedom": 0}
        t_stat, p_val = scipy_stats.ttest_rel(before, after)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "degrees_of_freedom": len(before) - 1,
        }

    @staticmethod
    def wilcoxon_signed_rank(
        before: list[float],
        after: list[float],
    ) -> dict[str, float]:
        if len(before) != len(after):
            n = min(len(before), len(after))
            before = before[:n]
            after = after[:n]
        if len(before) < 2:
            return {"statistic": 0.0, "p_value": 1.0}
        stat, p_val = scipy_stats.wilcoxon(before, after)
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
        }

    @staticmethod
    def effect_size(
        before: list[float],
        after: list[float],
    ) -> dict[str, float]:
        if len(before) != len(after):
            n = min(len(before), len(after))
            before = before[:n]
            after = after[:n]
        if len(before) < 2:
            return {"cohens_d": 0.0, "hedges_g": 0.0}
        mean_diff = float(np.mean(after) - np.mean(before))
        pooled_std = math.sqrt(
            (float(np.var(before, ddof=1)) + float(np.var(after, ddof=1))) / 2
        )
        if pooled_std == 0:
            return {"cohens_d": 0.0, "hedges_g": 0.0}
        cohens_d = mean_diff / pooled_std
        correction = 1 - (3 / (4 * (len(before) + len(after)) - 9))
        hedges_g = cohens_d * correction
        return {
            "cohens_d": float(cohens_d),
            "hedges_g": float(hedges_g),
        }

    @staticmethod
    def describe(values: list[float]) -> dict[str, float]:
        return {
            "count": float(len(values)),
            "mean": StatisticalAnalysis.mean(values),
            "median": StatisticalAnalysis.median(values),
            "std": StatisticalAnalysis.std(values),
            "variance": StatisticalAnalysis.variance(values),
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
        }

    @staticmethod
    def compare_groups(
        group_a: list[float],
        group_b: list[float],
        test: str = "paired_ttest",
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "group_a": StatisticalAnalysis.describe(group_a),
            "group_b": StatisticalAnalysis.describe(group_b),
        }
        result["mean_difference"] = (
            float(np.mean(group_b) - np.mean(group_a)) if group_a and group_b else 0.0
        )
        result["relative_improvement"] = (
            float((np.mean(group_b) - np.mean(group_a)) / abs(np.mean(group_a)) * 100)
            if group_a and abs(np.mean(group_a)) > 1e-10
            else 0.0
        )
        if test == "paired_ttest":
            result["test"] = "paired_ttest"
            result["test_results"] = StatisticalAnalysis.paired_ttest(group_a, group_b)
        elif test == "wilcoxon":
            result["test"] = "wilcoxon_signed_rank"
            result["test_results"] = StatisticalAnalysis.wilcoxon_signed_rank(
                group_a, group_b
            )
        result["effect_size"] = StatisticalAnalysis.effect_size(group_a, group_b)
        return result
