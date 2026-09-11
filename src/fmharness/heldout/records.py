"""The completion record every rung 1 stage writes beside its output (ruling 6).

A stage writes a file, then a ``<name>.done.json`` record beside it holding that file's
sha256 and size. A later run treats the file as already done only when the record exists
AND its sha256 still matches the file on disk -- so a hand-edited or truncated output is
correctly seen as not-done, and a rerun after a real change does not silently skip it.
One shared implementation, so no stage re-invents (or subtly varies) what "done" means.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """The sha256 hex digest of ``path``'s bytes, read in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_path(path: Path) -> Path:
    return path.with_name(path.name + ".done.json")


def write_record(path: Path, extra: dict[str, Any] | None = None) -> Path:
    """Write ``<path>.done.json`` recording ``path``'s name, sha256 and size.

    ``extra`` fields (if any) are merged in alongside ``file``, ``sha256`` and ``bytes``.
    Returns the record's path.
    """
    record_path = _record_path(path)
    record: dict[str, Any] = {
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **(extra or {}),
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record_path


def is_done(path: Path) -> bool:
    """True only when ``path`` exists, its record exists, and the record's sha256 matches."""
    record_path = _record_path(path)
    if not path.exists() or not record_path.exists():
        return False
    try:
        record = json.loads(record_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    recorded_sha256 = record.get("sha256")
    if not isinstance(recorded_sha256, str):
        return False
    return recorded_sha256 == sha256_file(path)


__all__ = ["is_done", "sha256_file", "write_record"]
