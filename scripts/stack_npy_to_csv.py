"""Convert Stack's embedding output (.npy) to a CSV indexed by organoid id.

Stack writes one embedding row per input pseudo-cell, in input order. We pair it
with the organoid ids from the AnnData we sent, producing the stack_soragni.csv
that scripts/benchmark_soragni.py reads via --stack-embeddings.

  python scripts/stack_npy_to_csv.py stack_soragni.npy \\
      data/reference/stack_input_soragni.h5ad stack_soragni.csv
"""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npy", help="Stack embedding output (.npy), one row per pseudo-cell")
    ap.add_argument("adata", help="the .h5ad sent to Stack (for organoid ids, in order)")
    ap.add_argument("out", help="destination CSV")
    args = ap.parse_args()

    emb = np.load(args.npy)
    obs = [str(o) for o in ad.read_h5ad(args.adata).obs_names]
    if emb.shape[0] != len(obs):
        raise SystemExit(f"row mismatch: {emb.shape[0]} embeddings vs {len(obs)} organoids")

    df = pd.DataFrame(emb, index=pd.Index(obs, name="organoid"))
    df.to_csv(args.out)
    print(f"wrote {args.out}  ({df.shape[0]} organoids x {df.shape[1]} dims)")


if __name__ == "__main__":
    main()
