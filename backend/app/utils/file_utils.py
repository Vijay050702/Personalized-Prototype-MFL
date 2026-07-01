import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
