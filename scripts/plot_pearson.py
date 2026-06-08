"""Pearson (linear) correlation of predicted vs observed AUC for the baselines.

The benchmark headline, global rho, is Spearman -- a rank correlation, so a few
wild predictions barely move it. Pearson is different: it is the linear
correlation on the raw AUC magnitudes, so it asks whether predicted AUC tracks
observed AUC in value, and it is highly sensitive to outliers. The ridge-on-PCA
model occasionally sends a held-out organoid to AUC ~1400 (a low-variance PC
direction), which on its own can flatten Pearson. So we report Pearson on all
held-out pairs and, for context, on the in-window pairs (predictions that fall
inside the observed-AUC range), and plot raw predicted vs observed AUC.

Run:
  uv run python scripts/plot_pearson.py
  uv run python scripts/plot_pearson.py --n-splits 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from fmharness.data.loaders import load_coderdata_tranche
from fmharness.evaluation import build_sample_design, grouped_cv_predict
from fmharness.probe import SimpleProbe

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-splits", type=int, default=5, help="leave-organoid-out CV folds")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    bundle = load_coderdata_tranche("sarcoma", repo)
    x_df, design = build_sample_design(bundle, "organoid", "auc")
    x_expr = np.log1p(x_df)

    models = {
        "drug_mean": lambda: SimpleProbe(n_components=0),
        "linear (PCA)": lambda: SimpleProbe(n_components=10),
        "nmf": lambda: SimpleProbe(n_components=10, reducer="nmf"),
    }

    obs = design["y"].to_numpy(dtype=float)
    pad = 0.05 * float(obs.max() - obs.min())
    lim = (float(obs.min()) - pad, float(obs.max()) + pad)

    print(f"leave-organoid-out {args.n_splits}-fold CV, n={len(design)} pairs\n")
    print(f"{'model':14s}{'pearson(all)':>14}{'pearson(in-win)':>17}{'spearman':>11}{'off-scale':>11}")
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, (name, factory) in zip(axes, models.items(), strict=True):
        preds = grouped_cv_predict(factory, x_expr, design, n_splits=args.n_splits, seed=SEED)
        pred = preds["y_pred"].to_numpy(dtype=float)
        true = preds["y_true"].to_numpy(dtype=float)
        r_all = float(np.asarray(pearsonr(true, pred))[0])
        rho = float(np.asarray(spearmanr(true, pred))[0])
        m = (pred >= lim[0]) & (pred <= lim[1])
        off = int((~m).sum())
        r_win = float(np.asarray(pearsonr(true[m], pred[m]))[0]) if m.sum() > 2 else float("nan")
        print(f"{name:14s}{r_all:>+14.3f}{r_win:>+17.3f}{rho:>+11.3f}{off:>11d}")

        ax.plot(lim, lim, color="0.6", lw=1, ls="--", zorder=0, label="identity")
        if m.sum() > 2:
            slope, intercept = np.polyfit(pred[m], true[m], 1)
            xs = np.array(lim)
            ax.plot(xs, intercept + slope * xs, color="C3", lw=1.3, label="OLS fit (in-window)")
        ax.scatter(pred, true, s=20, alpha=0.5, edgecolor="none")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nPearson r = {r_all:+.3f}")
        ax.set_xlabel("predicted AUC (held out)")
        if off:
            ax.text(
                0.97, 0.04,
                f"{off} off-scale (to {pred.max():.0f})\nr in-window = {r_win:+.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="crimson",
            )
    axes[0].set_ylabel("observed AUC")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        f"Soragni organoids - predicted vs observed AUC (raw), leave-organoid-out "
        f"{args.n_splits}-fold CV",
        y=1.02,
    )
    fig.tight_layout()

    out = repo / "results" / f"pearson_scatter_{args.n_splits}fold.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
