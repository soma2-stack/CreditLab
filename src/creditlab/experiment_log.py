"""Append-only JSON Lines logging for small experiment records."""

import json
from pathlib import Path
from typing import Any, Mapping


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one finite, JSON-serializable record to a JSONL file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True, allow_nan=False)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")