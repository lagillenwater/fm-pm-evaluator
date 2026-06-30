"""Score the Tahoe generation-mode eval: generation quality + the cell-line end-to-end check.

The Tahoe single-cell context (built on Alpine) gives, per (cell line, drug), a REAL
treated-minus-DMSO pseudobulk delta and a per-line baseline. Two checks run over the same
delta sources, so each source is judged on equal footing:

  Check 1 (generation quality, label-free): how faithfully a source reproduces the real
  Tahoe delta, per (cell line, drug) -- the per-pair delta-Pearson (Stack's own generation
  metric) plus the off-diagonal correlation and a specificity rank, which catch a source that
  is merely smooth (correlates with every condition) rather than specific.

  Check 2 (cell-line end-to-end, leave-cell-line-out): each source -> the fixed Hallmark
  death/proliferation readout -> a predicted sensitivity, scored against external MEASURED
  viability (GDSC2 AUC) on the shared (DepMap line, drug) pairs, by interaction rho with a
  within-drug label-permutation null and regret@k.

Delta sources form a ladder: ``additive`` (each drug's mean real delta, line-independent --
the floor) and ``knn`` (the mean real delta of the lines whose baseline is nearest the held
line -- uses the baseline, so it can express line x drug specificity). Both are rebuilt
leaving the scored line out, so a baseline never sees the held line's own treated cells.
Stack's generated delta plugs into the same ``sources`` map as a third rung once the Tahoe
generation run is produced on Alpine.

  PYTHONPATH=src python scripts/score_generation_eval.py --context tahoe_context.h5ad --k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from fmharness.adapters import build_adapters
from fmharness.data.loaders import load_tranche
from fmharness.deltas import build_additive_deltas, build_knn_deltas, build_tahoe_deltas
from fmharness.evaluation import build_sample_design, delta_fidelity, score_predictions
from fmharness.signatures import load_hallmark


def _loo_baseline_source(
    kind: str,
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    base: pd.DataFrame,
    *,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-cell-line-out baseline deltas: for each line, rebuild the source from the
    OTHER lines and predict the held-out line, so it never sees its own treated cells."""
    pats = real_key["patient"].astype(str).to_numpy()
    d_blocks: list[pd.DataFrame] = []
    k_blocks: list[pd.DataFrame] = []
    for line in [str(i) for i in base.index]:
        tr = pats != line
        if not tr.any():
            continue
        rd = real_delta[tr].reset_index(drop=True)
        rk = real_key[tr].reset_index(drop=True)
        if kind == "additive":
            d, kk = build_additive_deltas(rd, rk, [line])
        elif kind == "knn":
            d, kk = build_knn_deltas(base.drop(index=line), rd, rk, base.loc[[line]], [line], k=k)
        else:
            raise ValueError(f"unknown baseline source {kind!r}")
        d_blocks.append(d)
        k_blocks.append(kk)
    if not d_blocks:
        raise ValueError(f"no held-out lines produced a {kind} delta")
    return pd.concat(d_blocks, ignore_index=True), pd.concat(k_blocks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", required=True, help="Tahoe context AnnData (build_tahoe_context)")
    ap.add_argument("--auc-tranche", default="gdscv2", help="measured-AUC cohort for check 2")
    ap.add_argument("--k", type=int, default=10, help="neighbors for the k-NN source")
    ap.add_argument("--n-hvg", type=int, default=2000, help="top HVGs for the generation metric")
    ap.add_argument("--n-permutations", type=int, default=1000)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    ctx = Path(args.context) if Path(args.context).is_absolute() else repo / args.context
    real_delta, real_key, base = build_tahoe_deltas(ad.read_h5ad(ctx))
    print(
        f"Tahoe: {len(real_key)} (line, drug) pairs over {base.shape[0]} lines, "
        f"{real_delta.shape[1]} genes"
    )

    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": _loo_baseline_source("additive", real_delta, real_key, base, k=args.k),
        "knn": _loo_baseline_source("knn", real_delta, real_key, base, k=args.k),
    }
    # Stack plugs in here once the Tahoe generation run exists:
    #   sources["stack"] = build_generated_deltas(generated_dir, baseline, pert_to_drug)

    # Check 1 -- generation quality vs the real Tahoe delta.
    fid_rows: list[dict[str, object]] = []
    for name, (d, kk) in sources.items():
        f = delta_fidelity(d, kk, real_delta, real_key, n_hvg=args.n_hvg)
        fid_rows.append(
            {
                "source": name,
                "r": round(float(f["r"].mean()), 3),
                "r_offdiag": round(float(f["r_offdiag"].mean()), 3),
                "rank": round(float(f["rank"].mean()), 3),
                "n_pairs": len(f),
                "n_genes": int(f["n_genes"].iloc[0]),
            }
        )
    print("\n=== check 1: generation quality (delta-Pearson vs real Tahoe) ===")
    print(pd.DataFrame(fid_rows).to_string(index=False))

    # Check 2 -- end-to-end through the fixed Hallmark readout vs measured AUC.
    _, design = build_sample_design(
        load_tranche(args.auc_tranche, repo), "all", "auc", drug_key="pubchem_cid"
    )
    adapters = build_adapters(
        ["hallmark"], signatures=load_hallmark(repo / "data/static/hallmark_signatures.gmt")
    )
    out: list[dict[str, object]] = []
    for name, (d, kk) in sources.items():
        for adapter in adapters:
            sens = adapter.predict(d)
            merged = pd.DataFrame(
                {"patient": kk["patient"].to_numpy(), "drug": kk["drug"].to_numpy(), "_s": sens}
            ).merge(design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
            if merged.empty:
                print(f"  [{name}/{adapter.name}] no (line, drug) overlap with {args.auc_tranche}")
                continue
            preds = pd.DataFrame(
                {
                    "patient": merged["patient"],
                    "drug": merged["drug"],
                    "y_true": merged["y_true"].to_numpy(),
                    "y_pred": -merged["_s"].to_numpy(),
                }
            )
            s = score_predictions(preds, n_perm=args.n_permutations)
            out.append(
                {
                    "source": name,
                    "method": adapter.name,
                    "global": s["global"],
                    "interaction": s["interaction"],
                    "p_label": s["p_label"],
                    "regret@1": s["regret@1"],
                    "regret@3": s["regret@3"],
                    "n": int(s["n"]),
                }
            )
    print(f"\n=== check 2: end-to-end vs {args.auc_tranche} AUC (leave-cell-line-out) ===")
    print(pd.DataFrame(out).to_string(index=False) if out else "(no scored pairs)")


if __name__ == "__main__":
    main()
