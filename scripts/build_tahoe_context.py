"""Build the Tahoe-100M single-cell perturbation context for Stack-Large-Aligned generation.

Tahoe is the in-domain single-cell drug context (replacing bulk L1000): given a perturbation
context of drug-treated cells and a query baseline, Stack generates the query's treated state.
This streams a *subset* -- the target drugs plus their DMSO_TF vehicle controls, in the target
cell lines -- from the HuggingFace parquet, reconstructs expression over the Stack gene panel
from the tokenized (``genes`` token-id + ``expressions`` value) format, maps the Cellosaurus
``cell_line_id`` to its DepMap id, and writes a context AnnData whose obs schema matches
``build_l1000_context`` (pert_id / pert_iname / cell_id / is_control) so the stack-generation
call and the delta builders consume it unchanged. Treated and DMSO cells are tagged, so the
per-line baseline (is_control) and the real treated state (the truth for generation-quality)
are both slices of this one file -- no separate query/baseline build needed for cell lines.

Run on Alpine (needs ``datasets``; streams from HF so no full ~100M-cell download):
  python scripts/build_tahoe_context.py --drugs-cid 5330286 11707110 --dose-um 5 \\
      --out tahoe_context.h5ad
then generate (same call shape as the L1000 path, with the Tahoe context as base-adata):
  stack-generation --checkpoint stack-aligned/bc_large_aligned.ckpt \\
      --base-adata tahoe_context.h5ad --test-adata stack_input_sarcoma.h5ad \\
      --genelist stack-aligned/basecount_1000per_15000max.pkl --gene-name-col feature_name \\
      --output-dir generated/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from fmharness.tahoe import parse_dose_um, scatter_tokens

TAHOE = "tahoebio/Tahoe-100M"
DMSO = "DMSO_TF"  # Tahoe's vehicle-control drug name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--drugs-cid", nargs="*", default=None, help="PubChem CIDs to keep (default all)"
    )
    ap.add_argument(
        "--cell-lines", nargs="*", default=None, help="Cellosaurus cell_line_ids (default all)"
    )
    ap.add_argument("--dose-um", type=float, default=None, help="keep only this drug dose in uM")
    ap.add_argument("--out", default="tahoe_context.h5ad")
    ap.add_argument(
        "--batch", type=int, default=50000, help="cells scattered per chunk (bounds memory)"
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    from datasets import load_dataset  # type: ignore  # Alpine-only, heavy import

    # small metadata tables load fully; the cell matrix streams.
    gm = load_dataset(TAHOE, "gene_metadata", split="train").to_pandas()
    clm = load_dataset(TAHOE, "cell_line_metadata", split="train").to_pandas()
    sm = load_dataset(TAHOE, "sample_metadata", split="train").to_pandas()

    # Stack's native 15,012-gene vocabulary (its genelist). The context uses the full panel;
    # Stack zero-pads any of these a query (e.g. the Soragni ~12.8k subset) does not measure.
    # Match by uppercased gene symbol, mirroring Stack's own .str.upper() gene alignment.
    hvg = pd.read_csv(repo / "data/static/stack_hvg_genes.txt", header=None)[0].astype(str)
    panel = {s.upper() for s in hvg}
    sym_u = gm["gene_symbol"].astype(str).str.upper()
    pan = gm[sym_u.isin(panel)].drop_duplicates("token_id")
    panel_syms = [s.upper() for s in pan["gene_symbol"].astype(str)]
    token_to_col = {int(t): i for i, t in enumerate(pan["token_id"])}
    print(f"panel: {len(panel_syms)} of Stack's 15,012-gene vocabulary covered by Tahoe genes")

    # Cellosaurus cell_line_id -> DepMap id (column name not fixed across releases).
    dep_col = next((c for c in clm.columns if "depmap" in c.lower()), None)
    id_col = "cell_line_id" if "cell_line_id" in clm.columns else clm.columns[0]
    cl2dep: dict[str, str] = {}
    if dep_col:
        cl2dep = dict(zip(clm[id_col].astype(str), clm[dep_col].astype(str), strict=False))
    sample_dose = {
        str(s): parse_dose_um(str(c))
        for s, c in zip(sm["sample"], sm["drugname_drugconc"], strict=False)
    }

    cids = set(map(str, args.drugs_cid)) if args.drugs_cid else None
    lines = set(args.cell_lines) if args.cell_lines else None
    stream = load_dataset(TAHOE, "expression_data", split="train", streaming=True)

    g_acc: list[np.ndarray] = []
    e_acc: list[np.ndarray] = []
    obs_rows: list[tuple[object, ...]] = []
    mats: list[sparse.csr_matrix] = []

    def flush() -> None:
        if g_acc:
            mats.append(scatter_tokens(g_acc, e_acc, token_to_col, len(panel_syms)))
            g_acc.clear()
            e_acc.clear()

    for r in stream:
        is_ctl = r["drug"] == DMSO
        if cids is not None and not is_ctl and str(r["pubchem_cid"]) not in cids:
            continue
        if lines is not None and r["cell_line_id"] not in lines:
            continue
        dose = sample_dose.get(str(r["sample"]), float("nan"))
        if args.dose_um is not None and not is_ctl and not np.isclose(dose, args.dose_um):
            continue
        g_acc.append(r["genes"])
        e_acc.append(r["expressions"])
        obs_rows.append(
            (
                r["drug"],
                str(r["pubchem_cid"]),
                r["cell_line_id"],
                cl2dep.get(str(r["cell_line_id"]), ""),
                bool(is_ctl),
                r["plate"],
                r["sample"],
                dose,
            )
        )
        if len(g_acc) >= args.batch:
            flush()
    flush()

    n_cols = len(panel_syms)
    X = sparse.vstack(mats).tocsr() if mats else sparse.csr_matrix((0, n_cols), dtype=np.float32)
    obs = pd.DataFrame(
        obs_rows,
        columns=pd.Index(
            [
                "pert_iname",
                "pubchem_cid",
                "cell_line_id",
                "cell_id",
                "is_control",
                "plate",
                "sample",
                "dose_um",
            ]
        ),
    )
    obs["pert_id"] = obs["pert_iname"]  # mirror build_l1000_context (pert_id keys generated files)
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs_names = [str(i) for i in range(adata.n_obs)]
    adata.var_names = panel_syms
    adata.var["feature_name"] = panel_syms
    out = repo / args.out if not Path(args.out).is_absolute() else Path(args.out)
    adata.write_h5ad(out)
    print(
        f"wrote {out}  ({adata.n_obs} cells x {adata.n_vars} genes, "
        f"{int(obs['is_control'].sum())} DMSO, {obs['cell_id'].ne('').sum()} with a DepMap id)"
    )


if __name__ == "__main__":
    main()
