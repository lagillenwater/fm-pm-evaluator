"""Build the Stack embedding input for the Soragni organoids (run locally, ship to Alpine).

Stack embeds cells from an AnnData, matching genes by symbol against its
15,012-gene vocabulary. We have 15 *bulk* organoid profiles, so each organoid is
written as one pseudo-cell. Genes are restricted to the Stack<->Soragni panel
(data/reference/stack_soragni_gene_map.csv, from build_stack_panel.py) and
labelled by Stack symbol in var['feature_name'] -- the column Stack matches on.

Two caveats to test on Alpine (not bugs, modelling choices):
  - bulk-as-single-cell is off Stack's training distribution; it expects single
    cells in sets of ~256, and 15 pseudo-cells is a tiny context set.
  - expression is raw TPM here; Stack may expect counts. Try --log1p as well.

Output: data/reference/stack_input_soragni.h5ad   (send this file to Alpine)

Alpine recipe (GPU node; `pip install arc-stack` first):
  python -c "from huggingface_hub import snapshot_download; \\
             snapshot_download('arcinstitute/Stack-Large', local_dir='stack-large')"
  stack-embedding --checkpoint stack-large/bc_large.ckpt \\
      --adata stack_input_soragni.h5ad \\
      --genelist stack-large/basecount_1000per_15000max.pkl \\
      --gene-name-col feature_name --batch-size 16 \\
      --output stack_soragni.npy
  python scripts/stack_npy_to_csv.py stack_soragni.npy \\
      data/reference/stack_input_soragni.h5ad stack_soragni.csv

Then bring stack_soragni.csv back and run:
  uv run python scripts/benchmark_soragni.py --stack-embeddings stack_soragni.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.data.loaders import load_coderdata_tranche
from fmharness.evaluation import build_sample_design


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log1p", action="store_true", help="log1p the expression (default: raw TPM)")
    ap.add_argument("--out", default="data/reference/stack_input_soragni.h5ad")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    gmap = pd.read_csv(repo / "data/reference/stack_soragni_gene_map.csv")
    bundle = load_coderdata_tranche("sarcoma", repo)
    x_df, _ = build_sample_design(bundle, "organoid", "auc")

    cols = [str(e) for e in gmap["entrez_id"]]
    x = x_df[cols].to_numpy(dtype=np.float32)
    if args.log1p:
        x = np.log1p(x)

    adata = ad.AnnData(X=x)
    adata.obs_names = [str(i) for i in x_df.index]
    adata.var_names = gmap["stack_symbol"].tolist()
    adata.var["feature_name"] = gmap["stack_symbol"].to_numpy()

    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    kind = "log1p" if args.log1p else "raw TPM"
    print(f"wrote {out.relative_to(repo)}")
    print(f"  {adata.n_obs} organoids x {adata.n_vars} genes ({kind})")
    print("Stack matches var['feature_name'] against its 15,012-gene vocabulary at embed time.")


if __name__ == "__main__":
    main()
