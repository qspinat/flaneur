"""Archive sources. Each module exposes `harvest(client, cache_dir) -> Iterator[Photo]`."""

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO


def write_atomic(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Write a file whole, then name it: an interrupted write never leaves a truncated one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as out:
        write(out)
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    write_atomic(path, lambda out: out.write(text.encode()))


def append_row(path: Path, row: dict) -> dict:
    """Append a row to a JSON-lines file (the overrides: hand decisions), stamped with when."""
    row = {**row, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def number(value: Any) -> bool:
    """Whether a value a page sent is a finite number (not a bool, a text, NaN or infinity)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def seconds_taken(seconds: Any) -> float:
    """The seconds the user took on a decision, checked (a page sends them)."""
    if not number(seconds) or seconds < 0:
        raise ValueError("les secondes sont un nombre positif")
    return round(float(seconds), 1)


def rows(path: Path) -> list[dict]:
    """The rows of a JSON-lines file (none when it does not exist yet), in the order written."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False))


def cached_json(path: Path, fetch: Callable[[], Any]) -> Any:
    """Return the JSON stored at `path`, fetching and storing it first if it is missing."""
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    write_json(path, data)
    return data
