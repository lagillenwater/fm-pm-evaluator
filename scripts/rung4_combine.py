"""Combine rung 4's observed run with its matched permutation runs into one table per readout.

Rows: what predicted the expression response (mean, knn, pca, nmf, stack_base, stack_cytokine,
stack_sciplex, measured). Columns: overall r, its empirical p, interaction r, its p, and each
minus its null mean (with the null mean and SD). p = (1 + #null >= observed) / (1 + B).
Tables: rung4_<readout>.csv for proliferation, ridge, lasso; rung4_all_readouts.csv stacks them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = ("mean", "knn", "pca", "nmf", "stack_base", "stack_cytokine", "stack_sciplex", "measured")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observed", type=Path, required=True)
    ap.add_argument("--perm-dir", type=Path, required=True, help="directory holding perm_<k>/ result folders")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(args.observed / "rung4_summary.csv")
    perms = [pd.read_csv(p / "rung4_summary.csv").assign(perm=p.name) for p in sorted(args.perm_dir.glob("perm_*")) if (p / "rung4_summary.csv").exists()]
    null = pd.concat(perms, ignore_index=True) if perms else obs.iloc[0:0].assign(perm=None)
    ceiling = obs[obs["readout"] == "ceiling"].iloc[0]
    tables = []
    for ro in ("proliferation", "ridge", "lasso"):
        rows = []
        for r in ROWS:
            o = obs[(obs["readout"] == ro) & (obs["row"] == r)]
            if o.empty:
                continue
            o = o.iloc[0]
            n = null[(null["readout"] == ro) & (null["row"] == r)]
            rec = {"readout": ro, "row": r, "n_drugs": int(o["n_drugs"]), "n_perm": len(n)}
            for score in ("overall_r", "interaction_r"):
                x = float(o[score])
                nv = n[score].to_numpy(dtype=float)
                rec[score] = x
                rec[score.replace("_r", "_p")] = (1 + int(np.sum(nv >= x))) / (1 + len(nv)) if len(nv) and np.isfinite(x) else np.nan
                rec[score.replace("_r", "_null_mean")] = float(np.nanmean(nv)) if len(nv) else np.nan
                rec[score.replace("_r", "_null_sd")] = float(np.nanstd(nv, ddof=1)) if len(nv) > 1 else np.nan
                rec[score.replace("_r", "_minus_null")] = x - float(np.nanmean(nv)) if len(nv) else np.nan
            rows.append(rec)
        tab = pd.DataFrame(rows)
        tab["ceiling_overall"], tab["ceiling_interaction"] = float(ceiling["overall_r"]), float(ceiling["interaction_r"])
        cols = ["readout", "row", "overall_r", "overall_p", "interaction_r", "interaction_p", "overall_minus_null", "interaction_minus_null",
                "overall_null_mean", "overall_null_sd", "interaction_null_mean", "interaction_null_sd", "ceiling_overall", "ceiling_interaction", "n_drugs", "n_perm"]
        tab = tab[cols].round(4)
        tab.to_csv(args.out / f"rung4_{ro}.csv", index=False)
        tables.append(tab)
        print(f"=== {ro} (ceiling: overall {ceiling['overall_r']:.3f}, interaction {ceiling['interaction_r']:.3f}) ===")
        print(tab[["row", "overall_r", "overall_p", "interaction_r", "interaction_p", "overall_minus_null", "interaction_minus_null"]].to_string(index=False))
        print()
    pd.concat(tables, ignore_index=True).to_csv(args.out / "rung4_all_readouts.csv", index=False)


if __name__ == "__main__":
    main()
