"""Write rung 1's grid, ceiling table, and restriction record (design.md sections 2, 6, 7).

Reads rung 0's promoted per-pair table and dose-strata table, and the Tahoe drug metadata
(Hugging Face revision ``2dc57900b7981cfcf5e211527169a0b006546a95``, as a parquet or the
offline CSV fixture), and writes to ``--out-dir``:

* ``rung1_grid.json`` -- the grid's lines, drugs, screen-to-metadata drug-name crosswalk,
  the two pairs excluded for sci-Plex leakage, and the restriction record (dose, sha256 of
  the sorted lines and drugs, and each source file's sha256).
* ``rung1_ceiling.csv`` -- the grid's split-half ceiling for both gene sets, beside rung 0's
  promoted 5 uM row.

Each output is followed by a ``<name>.done.json`` completion record (``fmharness.heldout.records``);
a rerun skips both outputs when they are already done and valid.

    uv run python scripts/heldout_grid.py \\
        --per-pair results/rung0-assay-reliability/rung0_per_pair_r.csv \\
        --dose-strata results/rung0-assay-reliability/rung0_dose_strata.csv \\
        --drug-metadata path/to/drug_metadata.parquet \\
        --out-dir docs/tasks/rung1-held-out-prediction
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmharness.heldout.grid import (  # noqa: E402
    attach_drug_metadata,
    ceiling_table,
    grid_from_rung0,
    load_drug_metadata,
    restriction_record,
    sciplex_exposed_pairs,
)
from fmharness.heldout.records import is_done, write_record  # noqa: E402


def _read_per_pair(path: Path) -> pd.DataFrame:
    """Rung 0's per-pair table, keeping the cell line whose DepMap identifier is literal ``NA``."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _read_dose_strata(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--per-pair", type=Path, required=True)
    ap.add_argument("--dose-strata", type=Path, required=True)
    ap.add_argument("--drug-metadata", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.out_dir / "rung1_grid.json"
    ceiling_path = args.out_dir / "rung1_ceiling.csv"

    if is_done(grid_path) and is_done(ceiling_path):
        print(f"{grid_path} and {ceiling_path} already done, skipping")
        return

    per_pair = _read_per_pair(args.per_pair)
    dose_strata = _read_dose_strata(args.dose_strata)
    drug_metadata = load_drug_metadata(args.drug_metadata)

    grid = grid_from_rung0(per_pair)
    grid = attach_drug_metadata(grid, drug_metadata)
    excluded_pairs = sciplex_exposed_pairs(grid, drug_metadata)
    grid = replace(grid, excluded_pairs=excluded_pairs)

    sources = {
        "per_pair": args.per_pair,
        "dose_strata": args.dose_strata,
        "drug_metadata": args.drug_metadata,
    }
    record = restriction_record(grid, sources)
    grid_json = {"metadata_name": grid.metadata_name, **record}
    grid_path.write_text(json.dumps(grid_json, indent=2, sort_keys=True) + "\n")
    write_record(grid_path)
    print(f"wrote {grid_path}")

    ceiling = ceiling_table(per_pair, grid, dose_strata)
    ceiling.to_csv(ceiling_path, index=False)
    write_record(ceiling_path)
    print(f"wrote {ceiling_path}")


if __name__ == "__main__":
    main()
