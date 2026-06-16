"""Reproducibility check for the Soragni (sarcoma) download + processing.

Run on two machines (e.g. locally and on Alpine) and compare the printed
hashes. Matching hashes mean the downloaded data content and our processing into
(x_df, design) are reproducible across environments.

  - raw content hashes test the *download* (decompressed content, so the gzip
    timestamp does not matter).
  - processed hashes test the *processing* (build_sample_design) -- order is
    sorted out first so row/column ordering cannot change the hash.

Version lines are printed too: if a hash differs, check whether it is a data
version or a numpy/pandas/coderdata version difference.

  uv run python scripts/check_data_repro.py
"""

from __future__ import annotations

import gzip
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design
from fmharness.stack_panel import build_panel


def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _content_sha(path: Path) -> str:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return _sha16(raw)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    for pkg in ("coderdata", "numpy", "pandas", "anndata"):
        print(f"  {pkg:10s} {version(pkg)}")

    bundle = load_tranche("sarcoma", repo)
    x_df, design = build_sample_design(bundle, "organoid", "viability")

    raw = repo / "data/raw/coderdata"
    print("\n-- raw download content (sha256[:16]) --")
    for f in sorted(raw.glob("sarcoma_*")) + sorted(raw.glob("genes.*")):
        print(f"  {f.name:42s} {_content_sha(f)}")

    # processed frames, hashed order-independently
    xs = x_df.sort_index().reindex(sorted(x_df.columns), axis=1)
    xh = hashlib.sha256()
    xh.update("|".join(map(str, xs.index)).encode())
    xh.update("|".join(map(str, xs.columns)).encode())
    xh.update(np.ascontiguousarray(xs.to_numpy(np.float64)).tobytes())

    d = design.sort_values(["patient", "drug"]).reset_index(drop=True)
    dh = hashlib.sha256()
    dh.update("|".join(d["patient"].astype(str)).encode())
    dh.update("|".join(d["drug"].astype(str)).encode())
    dh.update(np.ascontiguousarray(d["y"].to_numpy(np.float64)).tobytes())

    xsha, dsha = xh.hexdigest()[:16], dh.hexdigest()[:16]
    print("\n-- processed (organoid, auc) --")
    print(f"  x_df    {x_df.shape[0]} x {x_df.shape[1]} genes   sha={xsha}")
    print(f"  design  {design.shape[0]} rows, {design['drug'].nunique()} drugs   sha={dsha}")

    # gene filter: pinned Stack vocabulary + the panel regenerated from it
    stack = [
        g.strip()
        for g in (repo / "data/static/stack_hvg_genes.txt").read_text().splitlines()
        if g.strip()
    ]
    genes_csv = pd.read_csv(repo / "data/raw/coderdata/genes.csv.gz")
    panel = build_panel(genes_csv, {int(c) for c in x_df.columns}, stack)
    vsha = _sha16("\n".join(sorted(stack)).encode())
    pp = panel.sort_values("stack_symbol").reset_index(drop=True)
    psha = _sha16(
        (
            "|".join(pp["stack_symbol"].astype(str)) + "##" + "|".join(pp["entrez_id"].astype(str))
        ).encode()
    )
    print("\n-- stack gene filter --")
    print(f"  vocabulary  {len(stack)} symbols   sha={vsha}")
    print(f"  panel       {len(panel)} genes (regenerated)   sha={psha}")


if __name__ == "__main__":
    main()
