from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd

from app.core.logging import logger


class Exporter:
    def __init__(self, output_dir: str = "exports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def export_json(
        self,
        data: dict[str, Any],
        filename: str = "results.json",
    ) -> str:
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported JSON to {path}")
        return str(path)

    def export_csv(
        self,
        data: list[dict[str, Any]] | dict[str, Any],
        filename: str = "results.csv",
    ) -> str:
        path = self._output_dir / filename
        if isinstance(data, dict):
            rows = self._flatten_dict(data)
        else:
            rows = data
        if not rows:
            with open(path, "w", newline="") as f:
                f.write("")
            return str(path)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Exported CSV to {path}")
        return str(path)

    def export_excel(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        filename: str = "results.xlsx",
        sheet_name: str = "Results",
    ) -> str:
        path = self._output_dir / filename
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and all(
            isinstance(v, (int, float, str)) for v in data.values()
        ):
            df = pd.DataFrame([data])
        else:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                if isinstance(data, dict):
                    for key, value in data.items():
                        if (
                            isinstance(value, list)
                            and value
                            and isinstance(value[0], dict)
                        ):
                            pd.DataFrame(value).to_excel(
                                writer, sheet_name=str(key)[:31]
                            )
                        elif isinstance(value, dict):
                            flat = pd.DataFrame([value])
                            flat.to_excel(writer, sheet_name=str(key)[:31])
                else:
                    pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name)
            logger.info(f"Exported Excel to {path}")
            return str(path)
        df.to_excel(path, sheet_name=sheet_name, index=False)
        logger.info(f"Exported Excel to {path}")
        return str(path)

    def export_markdown(
        self,
        content: str,
        filename: str = "results.md",
    ) -> str:
        path = self._output_dir / filename
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Exported Markdown to {path}")
        return str(path)

    def export_latex(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        filename: str = "results.tex",
        caption: str = "Experimental Results",
        label: str = "tab:results",
    ) -> str:
        path = self._output_dir / filename
        if isinstance(data, list) and data:
            rows = data
        elif isinstance(data, dict):
            rows = self._flatten_dict(data)
        else:
            rows = []

        if not rows:
            latex = "% No data available"
        else:
            columns = list(rows[0].keys())
            latex = "\\begin{table}[h!]\n"
            latex += f"\\centering\n\\caption{{{caption}}}\n"
            latex += f"\\label{{{label}}}\n"
            col_format = "|" + "c|" * len(columns)
            latex += f"\\begin{{tabular}}{{{col_format}}}\n\\hline\n"
            latex += (
                " & ".join(f"\\textbf{{{col}}}" for col in columns) + " \\\\ \\hline\n"
            )
            for row in rows:
                line = " & ".join(
                    f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()
                )
                latex += line + " \\\\ \\hline\n"
            latex += "\\end{tabular}\n\\end{table}"
        with open(path, "w") as f:
            f.write(latex)
        logger.info(f"Exported LaTeX to {path}")
        return str(path)

    def export_all(
        self,
        data: dict[str, Any],
        base_filename: str = "results",
        formats: list[str] | None = None,
    ) -> dict[str, str]:
        if formats is None:
            formats = ["json", "csv", "md", "tex"]
        exported: dict[str, str] = {}
        if "json" in formats:
            exported["json"] = self.export_json(data, f"{base_filename}.json")
        if "csv" in formats:
            exported["csv"] = self.export_csv(data, f"{base_filename}.csv")
        if "xlsx" in formats or "excel" in formats:
            exported["excel"] = self.export_excel(data, f"{base_filename}.xlsx")
        if "md" in formats or "markdown" in formats:
            exported["markdown"] = self.export_markdown(
                self._data_to_markdown(data), f"{base_filename}.md"
            )
        if "tex" in formats or "latex" in formats:
            exported["latex"] = self.export_latex(data, f"{base_filename}.tex")
        return exported

    def _flatten_dict(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if all(isinstance(v, dict) for v in data.values()):
            for key, subdict in data.items():
                row = {"key": key}
                if isinstance(subdict, dict):
                    row.update(subdict)
                rows.append(row)
        else:
            rows.append(dict(data))
        return rows

    def _data_to_markdown(self, data: dict[str, Any]) -> str:
        lines = ["# Evaluation Results\n"]
        for key, value in data.items():
            lines.append(f"## {key}\n")
            if isinstance(value, dict):
                if all(isinstance(v, (int, float, str)) for v in value.values()):
                    lines.append("| Metric | Value |\n| --- | --- |\n")
                    for k, v in value.items():
                        lines.append(f"| {k} | {v} |\n")
                else:
                    lines.append(
                        f"```json\n{json.dumps(value, indent=2, default=str)}\n```\n"
                    )
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    cols = list(value[0].keys())
                    lines.append("| " + " | ".join(cols) + " |\n")
                    lines.append("| " + " | ".join("---" for _ in cols) + " |\n")
                    for row in value:
                        lines.append(
                            "| "
                            + " | ".join(
                                f"{v:.4f}" if isinstance(v, float) else str(v)
                                for v in row.values()
                            )
                            + " |\n"
                        )
                else:
                    for item in value:
                        lines.append(f"- {item}\n")
            else:
                lines.append(f"{value}\n")
            lines.append("\n")
        return "".join(lines)

    def export_dataframe(
        self,
        df: pd.DataFrame,
        filename: str = "dataframe.csv",
    ) -> str:
        path = self._output_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Exported DataFrame to {path}")
        return str(path)

    def clear(self) -> None:
        for f in self._output_dir.iterdir():
            if f.is_file():
                f.unlink()
        logger.info(f"Cleared exports in {self._output_dir}")
