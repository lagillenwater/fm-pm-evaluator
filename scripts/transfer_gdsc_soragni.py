"""Transfer: train the head on GDSC2 cell lines, predict FROZEN on Soragni.

GDSC2 has the interaction power Soragni lacks (many cell lines per drug). We fit
the drug-mean + ridge head on GDSC2 over the genes and drugs shared with Soragni,
freeze it, and predict Soragni organoids -- never training on Soragni. This is
the powered test the within-Soragni benchmark cannot be: GDSC2 learns, for each
drug, the expression direction that predicts response, and we ask whether that
transfers to sarcoma organoids.

Trains on the sarcoma cell lines by default: the sarcoma-specific drug x organoid
interaction transfers only from in-domain training; the full pan-cancer panel
(--pan-cancer) washes it out. The general-sensitivity (substrate) gap is the
same either way.

The representation is swappable: log1p expression now, Stack embeddings via
--stack-gdsc/--stack-soragni (CSV indexed by cell-line / organoid id). Two heads
are scored: a shared slope (general sensitivity only) and a per-drug slope (which
can carry drug-specific / interaction signal). Predictions are frozen, so the
within-drug permutation null just reshuffles the Soragni labels -- no refit.

  uv run python scripts/transfer_gdsc_soragni.py
  uv run python scripts/transfer_gdsc_soragni.py --stack-gdsc g.csv --stack-soragni s.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from fmharness.controls import permute_within_drug
from fmharness.data.loaders import load_tranche
from fmharness.evaluation import (
    build_sample_design,
    cpm_bundle,
    interaction_rho,
    within_drug_rho,
)
from fmharness.probe import make_head
from fmharness.probe.heads import HEADS

SEED = 0
GDSC_SARCOMA = [
    "Alveolar Rhabdomyosarcoma",
    "Chondrosarcoma",
    "Ewing's Sarcoma",
    "Osteosarcoma",
    "Other Sarcomas",
    "Rhabdomyosarcoma",
]


def transfer_predict(factory, feat_tr, design_tr, feat_te, design_te) -> pd.DataFrame:
    """Fit a probe on the training cohort, predict the (frozen) test cohort."""
    probe = factory()
    probe.fit(
        feat_tr.loc[design_tr["patient"]].to_numpy(),
        list(design_tr["drug"]),
        design_tr["y"].to_numpy(),
        groups=list(design_tr["patient"]),
    )
    base, resid = probe.predict_parts(
        feat_te.loc[design_te["patient"]].to_numpy(), list(design_te["drug"])
    )
    return pd.DataFrame(
        {
            "patient": list(design_te["patient"]),
            "drug": list(design_te["drug"]),
            "y_true": design_te["y"].to_numpy(dtype=np.float64),
            "y_pred": base + resid,
            "y_resid": resid,
        }
    )


def score(preds: pd.DataFrame, n_perm: int) -> tuple[float, float, float, float, float]:
    gs = float(np.asarray(spearmanr(preds["y_true"], preds["y_pred"]))[0])
    gp = float(np.asarray(pearsonr(preds["y_true"], preds["y_pred"]))[0])
    wd = within_drug_rho(preds, "y_resid")
    it = interaction_rho(preds, "y_resid")
    null = np.empty(n_perm)
    for b in range(n_perm):
        rng = np.random.default_rng(SEED + 1 + b)
        p2 = preds.copy()
        p2["y_true"] = permute_within_drug(preds["drug"], preds["y_true"], rng)
        null[b] = interaction_rho(p2, "y_resid")
    return gs, gp, wd, it, float(np.mean(null >= it))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--std-floor", type=float, default=0.5)
    ap.add_argument("--n-permutations", type=int, default=1000)
    # Default to sarcoma-only GDSC2: training on the full pan-cancer panel washes
    # out the sarcoma-specific drug x organoid interaction signal (it survives the
    # permutation null only when trained in-domain). --pan-cancer opts back in.
    ap.add_argument("--pan-cancer", action="store_true", help="train on all GDSC2 lineages")
    ap.add_argument("--stack-gdsc", default=None)
    ap.add_argument("--stack-soragni", default=None)
    # Swap the predictive head to test whether the finding is head-invariant:
    # "linear" is the RidgeCV slope, "kernel" the RBF kernel ridge.
    ap.add_argument("--head", choices=list(HEADS), default="linear")
    ap.add_argument("--all-heads", action="store_true", help="run every head in turn")
    ap.add_argument(
        "--out",
        default=None,
        help="append (head, rep) metric rows to this CSV (e.g. results/head_invariance.csv)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    # Join GDSC2 <-> Soragni on PubChem CID: their native drug ids share no
    # namespace (Soragni uses drug names, GDSC2 numeric DRUG_IDs).
    # CPM-normalize both cohorts (length-free, per-million) so train and test
    # share one normalization -- the native loaders otherwise leave GDSC2 on
    # median-of-ratios and Soragni on CPM.
    xs, ds = build_sample_design(
        cpm_bundle(load_tranche("sarcoma", repo)), "organoid", "viability", drug_key="pubchem_cid"
    )
    ctf = None if args.pan_cancer else GDSC_SARCOMA
    gd = cpm_bundle(load_tranche("gdscv2", repo, cancer_type_filter=ctf))
    xg, dg = build_sample_design(gd, "all", "auc", drug_key="pubchem_cid")

    shared = sorted(set(ds["drug"].astype(str)) & set(dg["drug"].astype(str)))
    ds = ds[ds["drug"].astype(str).isin(shared)].copy()
    dg = dg[dg["drug"].astype(str).isin(shared)].copy()
    print(f"shared drugs {len(shared)} | gdsc rows {len(dg)} | soragni rows {len(ds)}")

    # Per-drug head only (a shared slope carries no interaction). Expression is
    # log1p(CPM) -- non-negative, so reducible by PCA or NMF; Stack embeddings can
    # be negative, so PCA only.
    genes = sorted(set(xs.columns) & set(xg.columns))
    ex_g, ex_s = np.log1p(xg[genes]), np.log1p(xs[genes])
    runs = [("expr/pca", ex_g, ex_s, "pca"), ("expr/nmf", ex_g, ex_s, "nmf")]
    if args.stack_gdsc and args.stack_soragni:
        eg = pd.read_csv(args.stack_gdsc, index_col=0)
        es = pd.read_csv(args.stack_soragni, index_col=0)
        eg.index, es.index = eg.index.astype(str), es.index.astype(str)
        runs.append(("stack/pca", eg, es, "pca"))

    heads = list(HEADS) if args.all_heads else [args.head]
    hdr = f"{'rep':18s}{'global_sp':>10}{'global_pe':>10}{'within':>9}{'interact':>10}{'p':>8}"
    rows: list[dict[str, object]] = []
    for head in heads:
        print(f"\n=== GDSC2 -> Soragni transfer (frozen, log1p CPM) | head={head} ===\n{hdr}")
        for label, fg, fs, reducer in runs:
            factory = make_head(
                head,
                n_components=args.n_components,
                std_floor=args.std_floor,
                per_drug=True,
                reducer=reducer,
            )
            preds = transfer_predict(factory, fg, dg, fs, ds)
            gs, gp, wd, it, pv = score(preds, args.n_permutations)
            print(f"{label:18s}{gs:>+10.3f}{gp:>+10.3f}{wd:>+9.3f}{it:>+10.3f}{pv:>8.3f}")
            rows.append(
                {
                    "head": head,
                    "rep": label,
                    "global_sp": gs,
                    "global_pe": gp,
                    "within": wd,
                    "interact": it,
                    "p": pv,
                }
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Append so the bilinear row (from transfer_pharmaformer_lite.py) and any
        # earlier head runs accumulate into one head-invariance table.
        df = pd.DataFrame(rows)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
