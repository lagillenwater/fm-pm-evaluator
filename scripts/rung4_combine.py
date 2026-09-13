"""Combine rung 4's observed run with its matched permutation runs.

For every column and both scores (per-drug r; interaction r with the line mean removed):
observed mean, null mean and SD over the permutations, observed minus null, and an empirical
p = (1 + #null >= observed) / (1 + B).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observed", type=Path, required=True)
    ap.add_argument("--perm-dir", type=Path, required=True, help="directory holding perm_<k>/ result folders")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    perms = sorted(p for p in args.perm_dir.glob("perm_*") if (p / "rung4_summary.csv").exists())
    for name, col in (("rung4_summary.csv", "mean_r"), ("rung4_summary_interaction.csv", "mean_r_interaction")):
        obs = pd.read_csv(args.observed / name, index_col=0)[col]
        null = pd.concat([pd.read_csv(p / name, index_col=0)[col].rename(p.name) for p in perms], axis=1)
        table = pd.DataFrame({"observed": obs, "null_mean": null.mean(axis=1), "null_sd": null.std(axis=1), "n_perm": null.count(axis=1)})
        table["observed_minus_null"] = table["observed"] - table["null_mean"]
        table["p_empirical"] = [(1 + (null.loc[m] >= obs[m]).sum()) / (1 + null.loc[m].count()) if m in null.index and np.isfinite(obs[m]) else np.nan for m in obs.index]
        table.round(4).to_csv(args.out / name.replace(".csv", "_vs_null.csv"))
        print(f"=== {name} ===\n{table.round(4).to_string()}\n")


if __name__ == "__main__":
    main()
