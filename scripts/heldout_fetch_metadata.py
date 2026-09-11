"""Fetch Tahoe-100M's drug and gene metadata tables for local, offline lookups (task 11).

Several rung 1 jobs need ``metadata/drug_metadata.parquet`` (the grid, and the drug-fingerprint
descriptions) and ``metadata/gene_metadata.parquet`` (the DMSO-cell gene panel) as local files
rather than one Hugging Face read apiece -- a job array of 64 DMSO-cell tasks each opening its
own ``HfFileSystem`` handle is the read pattern rung 0 saw rate-limited. This script downloads
both, once, at the revision already pinned for this tranche (``heldout_dmso_cells.TAHOE_REVISION``,
reused rather than re-typed as a second literal), and copies them into ``--out-dir``.

Each file is followed by a ``<name>.done.json`` completion record
(``fmharness.heldout.records``); a rerun skips a file whose record is already valid.
``huggingface_hub`` is imported inside the function that needs it -- Alpine-only, heavy.

    uv run python scripts/heldout_fetch_metadata.py \\
        --out-dir /scratch/alpine/$USER/rung1_cache/metadata
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from heldout_dmso_cells import TAHOE, TAHOE_REVISION  # noqa: E402

from fmharness.heldout.records import is_done, write_record  # noqa: E402

METADATA_FILES = ("drug_metadata.parquet", "gene_metadata.parquet")


def fetch_metadata(out_dir: Path) -> None:
    """Download the pinned drug and gene metadata tables into ``out_dir``.

    Skips a file whose completion record already matches it on disk; only imports
    ``huggingface_hub`` (and only touches the network) when at least one file is missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: out_dir / name for name in METADATA_FILES}
    missing = {name: path for name, path in targets.items() if not is_done(path)}
    if not missing:
        print(f"{out_dir}: metadata already done, skipping")
        return

    import huggingface_hub  # Alpine-only, heavy import

    for name, target in missing.items():
        downloaded = huggingface_hub.hf_hub_download(
            TAHOE,
            f"metadata/{name}",
            repo_type="dataset",
            revision=TAHOE_REVISION,
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copyfile(downloaded, target)
        write_record(target, {"revision": TAHOE_REVISION})
        print(f"wrote {target}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    fetch_metadata(args.out_dir)


if __name__ == "__main__":
    main()
