"""Build the L1000 perturbation context for Stack-Large-Aligned generation (Path B).

stack-generation needs a context (base-adata) of cells under each drug condition;
it then generates those conditions for the query cells (Soragni baseline). L1000
provides control (DMSO) + drug-treated bulk profiles for ~19 of Soragni's 26
drugs, which we treat as pseudo-cells over the Stack gene panel.

This selects, for the Soragni drugs present in L1000, their treated wells plus
DMSO controls (in chosen cell lines), reads those columns from the Level-3
expression matrix, maps genes to the Stack panel, and writes a context AnnData.

Run on Alpine (the Level-3 matrix is ~30 GB). First fetch the L1000 files:
  base=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl
  for f in pert_info inst_info gene_info; do
    curl -sLO $base/GSE92742_Broad_LINCS_${f}.txt.gz; done
  curl -sLO $base/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz
  gunzip GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz
  pip install cmapPy anndata
then:
  python scripts/build_l1000_context.py --l1000-dir . \\
      --gctx GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx --out l1000_context.h5ad
and generate:
  stack-generation --checkpoint stack-aligned/bc_large_aligned.ckpt \\
      --base-adata l1000_context.h5ad --test-adata stack_input_sarcoma.h5ad \\
      --genelist stack-aligned/basecount_1000per_15000max.pkl --gene-name-col feature_name \\
      --output-dir generated/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse_gctx import parse

from fmharness.data.loaders import load_coderdata_tranche
from fmharness.evaluation import build_sample_design

# Well-profiled L1000 core lines; keep the context modest. Override with --cell-lines.
CORE_LINES = ["A375", "A549", "HA1E", "HCC515", "HEPG2", "HT29", "MCF7", "PC3", "VCAP"]


def _ncid(x: object) -> str:
    try:
        return str(int(float(x)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return ""


def soragni_pert_ids(repo: Path, pert_info: pd.DataFrame) -> dict[str, str]:
    """Map Soragni drug name -> L1000 pert_id via PubChem CID or InChIKey prefix."""
    dr = pd.read_csv(repo / "data/raw/coderdata/sarcoma_drugs.tsv.gz", sep="\t")
    _, ds = build_sample_design(load_coderdata_tranche("sarcoma", repo), "organoid", "auc")
    sor = dr[dr["improve_drug_id"].astype(str).isin(set(ds["drug"].astype(str)))]
    sor = sor.drop_duplicates("improve_drug_id")
    cp = pert_info[pert_info["pert_type"] == "trt_cp"]
    by_cid = {_ncid(c): p for c, p in zip(cp["pubchem_cid"], cp["pert_id"], strict=True)}
    by_ikb = {str(k): p for k, p in zip(cp["inchi_key_prefix"], cp["pert_id"], strict=True)}
    out: dict[str, str] = {}
    for _, r in sor.iterrows():
        pid = by_cid.get(_ncid(r["pubchem_id"])) or by_ikb.get(str(r["InChIKey"])[:14])
        if pid:
            out[str(r["chem_name"])] = pid
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-dir", default=".", help="dir with GSE92742 *_info.txt.gz files")
    ap.add_argument("--gctx", required=True, help="path to the Level-3 .gctx")
    ap.add_argument("--cell-lines", nargs="*", default=CORE_LINES,
                    help="L1000 cell lines to use as context, or 'all' for every line")
    ap.add_argument("--out", default="l1000_context.h5ad")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="gctx columns read per chunk (caps peak memory)")
    ap.add_argument("--treated-cap", type=int, default=50,
                    help="max treated wells kept per drug (context size)")
    ap.add_argument("--dmso-cap", type=int, default=25, help="max DMSO wells kept per cell line")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    d = Path(args.l1000_dir)
    pert = pd.read_csv(d / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", low_memory=False)
    inst = pd.read_csv(d / "GSE92742_Broad_LINCS_inst_info.txt.gz", sep="\t", low_memory=False)
    gene = pd.read_csv(d / "GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t")

    pids = soragni_pert_ids(repo, pert)
    print(f"matched {len(pids)} Soragni drugs to L1000 pert_ids")

    in_lines = (pd.Series(True, index=inst.index) if args.cell_lines == ["all"]
                else inst["cell_id"].isin(args.cell_lines))
    treated = inst[in_lines & inst["pert_id"].isin(set(pids.values()))].copy()
    control = inst[in_lines & (inst["pert_iname"] == "DMSO")].copy()
    # Cap wells per condition (keeps the context modest and avoids OOM on the read);
    # Stack only needs a representative set of cells per drug, not every well.
    print(f"wells: {len(treated)} treated + {len(control)} DMSO; "
          f"capping to <= {args.treated_cap}/drug, <= {args.dmso_cap}/cell")
    treated = treated.sort_values("inst_id").groupby("pert_id", sort=False).head(args.treated_cap)
    control = control.sort_values("inst_id").groupby("cell_id", sort=False).head(args.dmso_cap)
    sel = pd.concat([treated, control])
    print(f"  after cap: {len(treated)} treated + {len(control)} DMSO = {len(sel)}; reading ...")

    # read selected columns from the Level-3 matrix in chunks (bounds peak memory)
    ids = sel["inst_id"].tolist()
    blocks = [parse(args.gctx, cid=ids[i:i + args.chunk]).data_df
              for i in range(0, len(ids), args.chunk)]
    gx = pd.concat(blocks, axis=1) if len(blocks) > 1 else blocks[0]  # genes (pr_gene_id) x wells
    sym = gene.set_index("pr_gene_id")["pr_gene_symbol"].astype(str)
    gx.index = [sym.get(int(i), "") for i in gx.index]

    panel = set(pd.read_csv(repo / "data/static/stack_soragni_gene_map.csv")["stack_symbol"])
    gx = gx[gx.index.isin(panel)]
    gx = gx[~gx.index.duplicated()]
    print(f"genes: {gx.shape[0]} of the Stack panel covered by L1000")

    obs = sel.set_index("inst_id").loc[gx.columns]
    adata = ad.AnnData(X=gx.T.to_numpy(dtype=np.float32))
    adata.obs_names = [str(c) for c in gx.columns]
    adata.obs["pert_id"] = obs["pert_id"].to_numpy()
    adata.obs["pert_iname"] = obs["pert_iname"].to_numpy()
    adata.obs["cell_id"] = obs["cell_id"].to_numpy()
    adata.obs["is_control"] = (obs["pert_iname"] == "DMSO").to_numpy()
    adata.var_names = list(gx.index)
    adata.var["feature_name"] = list(gx.index)

    adata.write_h5ad(repo / args.out if not Path(args.out).is_absolute() else Path(args.out))
    print(f"wrote {args.out}  ({adata.n_obs} wells x {adata.n_vars} genes, "
          f"{int(adata.obs['is_control'].sum())} DMSO)")


if __name__ == "__main__":
    main()
