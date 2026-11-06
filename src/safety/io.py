"""Shared JSON IO for safety response/label files (list of {prompt, response, ...})."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(data).__name__}.")
    return data


def load_response_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in load_json_rows(path):
        if not isinstance(item, dict) or "prompt" not in item or "response" not in item:
            raise ValueError("Each input row must be a dict with keys 'prompt' and 'response'.")
        rows.append({"prompt": str(item["prompt"]), "response": str(item["response"])})
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
