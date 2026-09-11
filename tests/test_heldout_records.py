"""The shared completion record every rung 1 stage writes beside its output.

``write_record`` writes ``<name>.done.json`` holding the file's sha256 and size;
``is_done`` is true only when that record exists and its sha256 still matches the file --
so a rerun skips finished work, and a hand-edited or truncated output is correctly seen as
not done.
"""

from __future__ import annotations

import json
from pathlib import Path

from fmharness.heldout.records import is_done, sha256_file, write_record


def test_write_record_then_is_done(tmp_path: Path) -> None:
    target = tmp_path / "output.csv"
    target.write_text("a,b\n1,2\n")

    assert not is_done(target)

    record_path = write_record(target, extra={"rows": 1})

    assert record_path == tmp_path / "output.csv.done.json"
    record = json.loads(record_path.read_text())
    assert record["file"] == "output.csv"
    assert record["sha256"] == sha256_file(target)
    assert record["bytes"] == target.stat().st_size
    assert record["rows"] == 1
    assert is_done(target)


def test_tampered_file_is_not_done(tmp_path: Path) -> None:
    target = tmp_path / "output.csv"
    target.write_text("a,b\n1,2\n")
    write_record(target)
    assert is_done(target)

    target.write_text("a,b\n1,2\n3,4\n")

    assert not is_done(target)


def test_missing_record_is_not_done(tmp_path: Path) -> None:
    target = tmp_path / "output.csv"
    target.write_text("a,b\n1,2\n")
    assert not is_done(target)
