from __future__ import annotations

import json
from datetime import datetime
from typing import Any


class ReportGenerator:
    def __init__(self, title: str = "PP-MFL Experiment Report") -> None:
        self._title = title

    def generate(
        self,
        experiment_summary: dict[str, Any],
        dataset_info: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        communication_stats: dict[str, Any] | None = None,
        prototype_stats: dict[str, Any] | None = None,
        knowledge_transfer_stats: dict[str, Any] | None = None,
        personalization_stats: dict[str, Any] | None = None,
        best_model: dict[str, Any] | None = None,
        conclusions: list[str] | None = None,
    ) -> str:
        report_parts: list[str] = []

        report_parts.append(self._section("1. Experiment Summary", experiment_summary))
        report_parts.append(self._section("2. Dataset Information", dataset_info))
        report_parts.append(self._section("3. Configuration", config))
        report_parts.append(self._section("4. Metrics", metrics))
        report_parts.append(
            self._section("5. Communication Statistics", communication_stats)
        )
        report_parts.append(self._section("6. Prototype Statistics", prototype_stats))
        report_parts.append(
            self._section("7. Knowledge Transfer Statistics", knowledge_transfer_stats)
        )
        report_parts.append(
            self._section("8. Personalization Statistics", personalization_stats)
        )
        report_parts.append(self._section("9. Best Model", best_model))

        if conclusions:
            report_parts.append("## 10. Conclusions\n")
            for c in conclusions:
                report_parts.append(f"- {c}\n")

        report = self._format_report(report_parts)
        return report

    def generate_from_runner(
        self,
        runner: Any,
        extra_notes: list[str] | None = None,
    ) -> str:
        results = getattr(runner, "_results", {}) if hasattr(runner, "_results") else {}
        config = getattr(runner, "_config", {}) if hasattr(runner, "_config") else {}
        engine = getattr(runner, "engine", None) if hasattr(runner, "engine") else None
        engine_summary = engine.summary() if engine else {}

        experiment_summary = {
            "num_experiments": len(results),
            "experiment_ids": list(results.keys()),
            "generated_at": datetime.now().isoformat(),
        }

        if hasattr(runner, "summarize_experiments"):
            try:
                experiment_summary.update(runner.summarize_experiments())
            except Exception:
                pass

        metrics = engine_summary if engine_summary else {}
        best_id, best_val = ("", 0.0)
        if hasattr(runner, "get_best_experiment"):
            try:
                best_id, best_val = runner.get_best_experiment()
            except Exception:
                pass

        best_model = (
            {
                "best_experiment_id": best_id,
                "best_accuracy": best_val,
            }
            if best_id
            else {}
        )

        conclusions = extra_notes or []
        if best_id:
            conclusions.append(
                f"Best experiment '{best_id}' achieved accuracy {best_val:.4f}."
            )

        return self.generate(
            experiment_summary=experiment_summary,
            config=config,
            metrics=metrics,
            best_model=best_model,
            conclusions=conclusions,
        )

    def _section(self, title: str, data: Any) -> str:
        parts: list[str] = [f"## {title}\n"]
        if data is None:
            parts.append("No data available.\n\n")
            return "".join(parts)
        if isinstance(data, dict):
            if all(isinstance(v, (int, float, str, bool)) for v in data.values()):
                parts.append("| Metric | Value |\n| --- | --- |\n")
                for k, v in data.items():
                    parts.append(f"| {k} | {v} |\n")
            else:
                parts.append("```json\n")
                parts.append(json.dumps(data, indent=2, default=str))
                parts.append("\n```\n")
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                cols = list(data[0].keys())
                parts.append("| " + " | ".join(cols) + " |\n")
                parts.append("| " + " | ".join("---" for _ in cols) + " |\n")
                for row in data:
                    parts.append(
                        "| "
                        + " | ".join(
                            f"{v:.4f}" if isinstance(v, float) else str(v)
                            for v in row.values()
                        )
                        + " |\n"
                    )
            else:
                for item in data:
                    parts.append(f"- {item}\n")
        else:
            parts.append(f"{data}\n")
        parts.append("\n")
        return "".join(parts)

    def _format_report(self, parts: list[str]) -> str:
        header = [
            f"# {self._title}\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            "---\n\n",
        ]
        return "".join(header) + "".join(parts)
