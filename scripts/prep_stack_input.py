"""Build a Stack embedding input from a tranche (run locally, ship to Alpine).

Stack is a single-cell count model: its dataloader feeds expression straight into
``log1p`` + a negative-binomial decoder, so it expects raw-count-derived,
gene-length-free values -- not TPM (TPM divides by gene length, which scRNA-seq
UMI data never does). We therefore feed both cohorts the same length-free,
per-million (CPM) representation:

  - GDSC2: DepMap raw read counts -> CPM (the loader keeps the raw counts in
    ``layers['raw_counts']``).
  - Soragni: the deposited matrix is already CPM (length-free, +0.46 gene-length
    coupling); re-normalized to per-million over measured genes.

Each row is then renormalized to 1e6 over the shared Stack panel genes so the two
cohorts enter Stack on an identical scale. We do NOT log1p by default -- Stack
applies log1p internally; ``--log1p`` is kept only as an ablation.

Genes are restricted to the shared Stack panel
(data/static/stack_soragni_gene_map.csv, from build_stack_panel.py) intersected
with the cohort's measured genes, and labelled by Stack symbol in
var['feature_name'] -- the column Stack matches on.

One caveat to test on Alpine (a modelling choice, not a bug): bulk-as-single-cell
is off Stack's training distribution in *depth* -- it expects shallow single
cells in sets of ~256. CPM fixes the length/format axis, not the depth axis.

  # Soragni organoids (test cohort)
  uv run python scripts/prep_stack_input.py
  # GDSC2 sarcoma cell lines (powered cohort)
  uv run python scripts/prep_stack_input.py --dataset gdscv2 --rna-source all

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

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design, cpm_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sarcoma")
    ap.add_argument("--rna-source", default="tumor", choices=["all", "organoid", "tumor"])
    ap.add_argument(
        "--sarcoma-only",
        action="store_true",
        help="restrict gdscv2 to sarcoma lineages, so Stack's in-context prompt is sarcoma-only",
    )
    ap.add_argument(
        "--log1p", action="store_true", help="log1p the CPM (ablation; Stack log1ps internally)"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    gmap = pd.read_csv(repo / "data/static/stack_soragni_gene_map.csv")
    # A non-None cancer_type_filter signals the gdscv2 loader to restrict to sarcoma.
    ctf = ["sarcoma"] if args.sarcoma_only else None
    bundle = cpm_bundle(load_tranche(args.dataset, repo, cancer_type_filter=ctf))
    # metric is irrelevant here (we keep only the expression frame); pass the
    # cohort's real metric so build_sample_design does not warn on an empty design.
    metric = "viability" if args.dataset in ("sarcoma", "soragni") else "auc"
    x_df, _ = build_sample_design(bundle, args.rna_source, metric)

    # keep panel genes this cohort actually measures, in panel order
    present = set(x_df.columns.astype(str))
    gmap = gmap[gmap["entrez_id"].astype(str).isin(present)]
    cols = [str(e) for e in gmap["entrez_id"]]
    x = x_df[cols].to_numpy(dtype=np.float64)
    # renormalize each pseudo-cell to 1e6 over the shared panel genes, so both
    # cohorts enter Stack on an identical length-free, per-million scale.
    row = x.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    x = x / row * 1e6
    if args.log1p:
        x = np.log1p(x)
    x = x.astype(np.float32)

    adata = ad.AnnData(X=x)
    adata.obs_names = [str(i) for i in x_df.index]
    adata.var_names = gmap["stack_symbol"].tolist()
    adata.var["feature_name"] = gmap["stack_symbol"].to_numpy()

    out = repo / (args.out or f"data/reference/stack_input_{args.dataset}.h5ad")
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    kind = "log1p(CPM)" if args.log1p else "CPM (linear)"
    print(f"wrote {out.relative_to(repo)}")
    print(f"  {adata.n_obs} pseudo-cells x {adata.n_vars} panel genes ({kind}, length-free)")
    print("Stack matches var['feature_name'] against its 15,012-gene vocabulary at embed time.")


if __name__ == "__main__":
    main()
