"""Build a Stack embedding input from a CoderData cohort (run locally, ship to Alpine).

Stack embeds cells from an AnnData, matching genes by symbol against its
15,012-gene vocabulary. CoderData expression is bulk, so each sample (organoid or
cell line) is written as one pseudo-cell. Genes are restricted to the shared
Stack panel (data/static/stack_soragni_gene_map.csv, from build_stack_panel.py)
intersected with the cohort's measured genes, and labelled by Stack symbol in
var['feature_name'] -- the column Stack matches on. Both cohorts use the same
panel so their embeddings live in the same gene space and stay comparable.

Two caveats to test on Alpine (not bugs, modelling choices):
  - bulk-as-single-cell is off Stack's training distribution; it expects single
    cells in sets of ~256.
  - expression is raw TPM here; Stack may expect counts. Try --log1p as well.

  # Soragni organoids (test cohort)
  uv run python scripts/prep_stack_input.py
  # GDSC2 sarcoma cell lines (powered cohort)
  uv run python scripts/prep_stack_input.py --dataset gdscv2 --rna-source all --sarcoma-only

Output: data/reference/stack_input_<dataset>.h5ad   (send this file to Alpine)

Alpine recipe (GPU node; `pip install arc-stack` first):
  python -c "from huggingface_hub import snapshot_download; \\
             snapshot_download('arcinstitute/Stack-Large', local_dir='stack-large')"
  stack-embedding --checkpoint stack-large/bc_large.ckpt --adata <input>.h5ad \\
      --genelist stack-large/basecount_1000per_15000max.pkl \\
      --gene-name-col feature_name --batch-size 16 --output <out>.npy
  python scripts/stack_npy_to_csv.py <out>.npy <input>.h5ad <out>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.data.loaders import load_coderdata_tranche
from fmharness.evaluation import build_sample_design

GDSC_SARCOMA = [
    "Alveolar Rhabdomyosarcoma",
    "Chondrosarcoma",
    "Ewing's Sarcoma",
    "Osteosarcoma",
    "Other Sarcomas",
    "Rhabdomyosarcoma",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sarcoma")
    ap.add_argument("--rna-source", default="organoid", choices=["all", "organoid", "tumor"])
    ap.add_argument("--sarcoma-only", action="store_true", help="restrict gdscv2 to sarcoma types")
    ap.add_argument("--log1p", action="store_true", help="log1p the expression (default: raw TPM)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    gmap = pd.read_csv(repo / "data/static/stack_soragni_gene_map.csv")
    ctf = GDSC_SARCOMA if args.sarcoma_only and args.dataset == "gdscv2" else None
    bundle = load_coderdata_tranche(args.dataset, repo, cancer_type_filter=ctf)
    x_df, _ = build_sample_design(bundle, args.rna_source, "auc")

    # keep panel genes this cohort actually measures, in panel order
    present = set(x_df.columns.astype(str))
    gmap = gmap[gmap["entrez_id"].astype(str).isin(present)]
    cols = [str(e) for e in gmap["entrez_id"]]
    x = x_df[cols].to_numpy(dtype=np.float32)
    if args.log1p:
        x = np.log1p(x)

    adata = ad.AnnData(X=x)
    adata.obs_names = [str(i) for i in x_df.index]
    adata.var_names = gmap["stack_symbol"].tolist()
    adata.var["feature_name"] = gmap["stack_symbol"].to_numpy()

    out = repo / (args.out or f"data/reference/stack_input_{args.dataset}.h5ad")
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    kind = "log1p" if args.log1p else "raw TPM"
    print(f"wrote {out.relative_to(repo)}")
    print(f"  {adata.n_obs} samples x {adata.n_vars} genes ({kind})")
    print("Stack matches var['feature_name'] against its 15,012-gene vocabulary at embed time.")


if __name__ == "__main__":
    main()
