"""Minimal Soragni drug-response benchmark.

Question: can an embedding of an organoid's expression predict its drug response
(AUC) beyond just knowing the drug? We compare predictors within the Soragni PDTO
cohort under leave-one-organoid-out cross-validation:

  drug_mean : per-drug mean only, no organoid information      -- the floor
  linear    : drug mean + one shared ridge slope on the top-k PCs of log1p
              expression, standardized with a per-gene SD floor so a held-out
              organoid cannot blow up (a general-sensitivity model: same organoid
              offset for every drug, so it cannot express drug-specific response)
  nmf       : same as linear, but the k-dim summary is non-negative gene
              programs (NMF) instead of principal components
  linear_pd : drug mean + a per-drug ridge slope on the same PCs (the slope
              depends on the drug, so it can rank organoids within a drug)
  stack     : drug mean + ridge on the top-k PCs of Stack embeddings
              (only if --stack-embeddings is given)

Each predictor uses the SAME head (SimpleProbe) and the SAME CV, so the only
thing that changes is the representation. Scored by within-drug and interaction
(double-centered) Spearman rho -- signal beyond the drug mean -- each against a
within-drug permutation null. global rho is reported for context (it is mostly
the drug mean and the drug_mean predictor already scores high on it).

Run:
  uv run python scripts/benchmark_soragni.py
  uv run python scripts/benchmark_soragni.py --stack-embeddings stack_soragni.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.controls import permute_within_drug
from fmharness.data.loaders import load_coderdata_tranche
from fmharness.evaluation import (
    build_sample_design,
    global_spearman,
    grouped_cv_predict,
    interaction_rho,
    within_drug_rho,
)
from fmharness.probe import SimpleProbe

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sarcoma")
    ap.add_argument("--rna-source", default="organoid", choices=["all", "organoid", "tumor"])
    ap.add_argument("--n-components", type=int, default=10, help="PCs for the linear/stack probe")
    ap.add_argument(
        "--std-floor",
        type=float,
        default=0.5,
        help="min per-gene SD in PCA standardization; prevents out-of-sample blow-ups",
    )
    ap.add_argument("--n-splits", type=int, default=5, help="leave-organoid-out CV folds")
    ap.add_argument("--n-permutations", type=int, default=100)
    ap.add_argument(
        "--stack-embeddings",
        default=None,
        help="CSV of Stack embeddings indexed by organoid id (one row per organoid)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    bundle = load_coderdata_tranche(args.dataset, repo)
    x_df, design = build_sample_design(bundle, args.rna_source, "auc")
    print(
        f"{args.dataset} ({args.rna_source}): {x_df.shape[0]} organoids x {x_df.shape[1]} genes; "
        f"{len(design)} (organoid,drug) rows; {design['drug'].nunique()} drugs"
    )

    x_expr = np.log1p(x_df)  # the linear baseline's representation

    # (factory, feature frame). drug_mean uses n_components=0 so the feature is ignored.
    floor = args.std_floor
    predictors: dict[str, tuple] = {
        "drug_mean": (lambda: SimpleProbe(n_components=0), x_expr),
        "linear": (lambda: SimpleProbe(n_components=args.n_components, std_floor=floor), x_expr),
        "nmf": (lambda: SimpleProbe(n_components=args.n_components, reducer="nmf"), x_expr),
        "linear_pd": (
            lambda: SimpleProbe(n_components=args.n_components, per_drug=True, std_floor=floor),
            x_expr,
        ),
    }
    if args.stack_embeddings:
        emb = pd.read_csv(args.stack_embeddings, index_col=0)
        emb.index = emb.index.astype(str)
        common = emb.index.intersection(x_df.index.astype(str))
        emb = emb.loc[common]
        print(f"  stack embeddings: {emb.shape[0]} organoids x {emb.shape[1]} dims")
        predictors["stack"] = (
            lambda: SimpleProbe(n_components=args.n_components, std_floor=floor),
            emb,
        )

    print(f"\n=== leave-one-organoid-out CV ({args.n_splits} folds) ===")
    rows: list[dict[str, object]] = []
    for name, (factory, feat) in predictors.items():
        preds = grouped_cv_predict(factory, feat, design, n_splits=args.n_splits, seed=SEED)
        g = global_spearman(preds)
        wd = within_drug_rho(preds, "y_resid")
        inter = interaction_rho(preds, "y_resid")
        # permutation null for the headline (interaction): shuffle y within drug, re-run CV.
        null = np.empty(args.n_permutations)
        for b in range(args.n_permutations):
            rng = np.random.default_rng(SEED + 1 + b)
            d_perm = design.assign(y=permute_within_drug(design["drug"], design["y"], rng))
            p = grouped_cv_predict(factory, feat, d_perm, n_splits=args.n_splits, seed=SEED)
            null[b] = interaction_rho(p, "y_resid")
        pval = float(np.mean(null >= inter))
        rows.append(
            {
                "predictor": name,
                "global_rho": g,
                "within_drug_rho": wd,
                "interaction_rho": inter,
                "interaction_null_mean": float(null.mean()),
                "interaction_null_p95": float(np.quantile(null, 0.95)),
                "interaction_p": pval,
            }
        )
        print(
            f"  {name:10s} global={g:+.3f}  within_drug={wd:+.3f}  interaction={inter:+.3f}  "
            f"(null mean {null.mean():+.3f}, 95th {np.quantile(null, 0.95):+.3f}, p={pval:.3f})"
        )

    out = repo / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "benchmark_soragni.csv", index=False)
    print("\nWrote results/benchmark_soragni.csv")


if __name__ == "__main__":
    main()
