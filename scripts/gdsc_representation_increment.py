"""Within GDSC2 sarcoma: how much drug-specific signal does each representation
capture, relative to the subtype label, and how does it scale with components?

GDSC2 sarcoma is where the interaction signal is real and powered (54 cell lines,
154 drugs). The subtype label alone recovers most of it, so the question for any
richer representation -- expression PCs, NMF gene programs, or Stack's embedding
-- is whether it adds drug-relevant biology *beyond lineage*, and whether Stack's
1600-dim embedding needs more components than 10 to show its worth. Same per-drug
head, same leave-cell-line-out CV; we sweep n_components and compare interaction.

Note: with 54 cell lines, PCA/NMF cap near 42 components in CV regardless of the
embedding's native width -- Stack's full 1600 dims cannot be used in this cohort.

  uv run python scripts/gdsc_representation_increment.py --stack-gdsc stack_gdsc.csv
  # widen the sweep with:  --components 10,20,40
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design, grouped_cv_predict, interaction_rho
from fmharness.probe import SimpleProbe

SEED = 0
GDSC_SARCOMA = [
    "Alveolar Rhabdomyosarcoma",
    "Chondrosarcoma",
    "Ewing's Sarcoma",
    "Osteosarcoma",
    "Other Sarcomas",
    "Rhabdomyosarcoma",
]


def inter(factory, feat: pd.DataFrame, design: pd.DataFrame, n_splits: int) -> float:
    return interaction_rho(grouped_cv_predict(factory, feat, design, n_splits=n_splits, seed=SEED))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--components", default="10,20,40", help="comma-separated n_components sweep")
    ap.add_argument("--std-floor", type=float, default=0.5)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--stack-gdsc", default=None, help="CSV of Stack embeddings by cell-line id")
    args = ap.parse_args()
    ks = [int(x) for x in args.components.split(",")]

    repo = Path(__file__).resolve().parent.parent
    bundle = load_tranche("gdscv2", repo, cancer_type_filter=GDSC_SARCOMA)
    xg, dg = build_sample_design(bundle, "all", "auc")
    ct = (
        pd.read_csv(repo / "data/raw/coderdata/gdscv2_samples.csv")
        .drop_duplicates("common_name")
        .set_index("common_name")["cancer_type"]
    )
    onehot = pd.get_dummies(ct.reindex([str(i) for i in xg.index]).fillna("NA")).astype(float)
    onehot.index = xg.index
    expr = np.log1p(xg)

    sub_it = inter(
        partial(SimpleProbe, n_components=onehot.shape[1], per_drug=True, std_floor=0.0),
        onehot,
        dg,
        args.n_splits,
    )
    reps: list[tuple[str, pd.DataFrame, str]] = [("expression", expr, "pca"), ("nmf", expr, "nmf")]
    if args.stack_gdsc:
        emb = pd.read_csv(args.stack_gdsc, index_col=0)
        emb.index = emb.index.astype(str)
        emb = emb.loc[emb.index.intersection(xg.index.astype(str))]
        reps.append(("stack", emb, "pca"))

    print(
        f"GDSC2 sarcoma: {xg.shape[0]} cell lines, {dg['drug'].nunique()} drugs | "
        f"interaction rho, leave-cell-line-out {args.n_splits}-fold, per-drug head"
    )
    print(f"subtype baseline interaction = {sub_it:+.3f}\n")
    print(f"{'representation':14s}" + "".join(f"{'k=' + str(k):>9}" for k in ks))
    for name, feat, red in reps:
        cells = []
        for k in ks:
            fl = 0.0 if red == "nmf" else args.std_floor
            factory = partial(SimpleProbe, n_components=k, per_drug=True, reducer=red, std_floor=fl)
            cells.append(f"{inter(factory, feat, dg, args.n_splits):+.3f}")
        print(f"{name:14s}" + "".join(f"{c:>9}" for c in cells))
    print(f"\nincrement over subtype = value - ({sub_it:+.3f}); PCA/NMF cap ~42 components at n=54")


if __name__ == "__main__":
    main()
