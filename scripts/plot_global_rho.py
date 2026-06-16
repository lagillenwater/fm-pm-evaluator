"""Picture behind the global-rho number for the three expression baselines.

global rho is the Spearman rank correlation between the held-out predicted AUC
and the observed AUC, taken over every (organoid, drug) pair. Because that is a
rank correlation, this plots ranks: the rank of the predicted AUC on the x-axis
against the rank of the observed AUC on the y-axis, one panel per model. The
plotted cloud's agreement with the diagonal IS the reported rho, and the rank
scale is bounded so a ridge model's occasional off-scale prediction (a held-out
organoid sent to AUC ~1400 along a low-variance PC direction) cannot distort it.

For drug_mean the prediction is the drug's mean only, so all organoids of a drug
share one predicted value -- they take one tied rank and the points fall into
vertical stripes, the vertical spread inside a stripe being the organoid
variation the model cannot see. PCA and NMF tilt those stripes by nudging each
organoid off the drug mean.

Run:
  uv run python scripts/plot_global_rho.py
  uv run python scripts/plot_global_rho.py --n-splits 15   # leave-one-organoid-out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design, grouped_cv_predict
from fmharness.probe import SimpleProbe

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-splits", type=int, default=5, help="leave-organoid-out CV folds")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    bundle = load_tranche("sarcoma", repo)
    x_df, design = build_sample_design(bundle, "organoid", "viability")
    x_expr = np.log1p(x_df)

    models = {
        "drug_mean": lambda: SimpleProbe(n_components=0),
        "linear (PCA)": lambda: SimpleProbe(n_components=10),
        "nmf": lambda: SimpleProbe(n_components=10, reducer="nmf"),
    }

    # global rho IS the rank correlation between predicted and observed AUC, so
    # plotting ranks shows exactly what the number measures. It is also bounded
    # by construction, so the off-scale predictions ridge can make out-of-sample
    # (low-variance PC directions sending a held-out organoid to AUC ~1400) just
    # become the top rank instead of wrecking the axes.
    n = len(design)
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, (name, factory) in zip(axes, models.items(), strict=True):
        preds = grouped_cv_predict(factory, x_expr, design, n_splits=args.n_splits, seed=SEED)
        rho = float(np.asarray(spearmanr(preds["y_true"], preds["y_pred"]))[0])
        rx = preds["y_pred"].rank().to_numpy()
        ry = preds["y_true"].rank().to_numpy()
        ax.plot(
            [1, n], [1, n], color="0.6", lw=1, ls="--", zorder=0, label="perfect rank agreement"
        )
        ax.scatter(rx, ry, s=20, alpha=0.5, edgecolor="none")
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, n + 1)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nglobal $\\rho$ = {rho:+.3f}")
        ax.set_xlabel("predicted AUC rank")
    axes[0].set_ylabel("observed AUC rank")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        f"Soragni organoids - predicted vs observed AUC, leave-organoid-out "
        f"{args.n_splits}-fold CV (n={len(design)} organoid x drug pairs)",
        y=1.03,
    )
    fig.tight_layout()

    out = repo / "results" / f"global_rho_scatter_{args.n_splits}fold.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
