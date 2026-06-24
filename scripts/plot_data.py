"""Data-representation figures for the README -- the data, no conclusions.

These plots show the raw inputs and outputs of the benchmark with no model, rho,
or p-value annotations: cohort make-up, the response distributions, the Soragni
organoid x drug viability matrix, and the shared-drug panel side by side. They
exist so a reader can see what the evaluation is built on before any result.

Optionally renders the head-invariance figure from results/head_invariance.csv
(a grouped bar of interaction rho by head x representation) when that file exists;
that one summarizes a produced metric table, still without prose.

  uv run python scripts/plot_data.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from _plotting import plt, savefig

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design

GDSC_SARCOMA = ["sarcoma"]


def _subtype_counts(bundle) -> Counter:
    return Counter(p.subtype or "unknown" for p in bundle.patients)


def plot_cohort_composition(sb, gb, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, bundle) in zip(
        axes, (("GDSC2 sarcoma cell lines", gb), ("Soragni PDTOs", sb)), strict=True
    ):
        counts = _subtype_counts(bundle).most_common()
        labels = [c[0] for c in counts]
        vals = [c[1] for c in counts]
        ax.barh(range(len(labels)), vals, color="#4477aa")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("patients / lines")
        ax.set_title(f"{name} (n={sum(vals)})")
    fig.tight_layout()
    savefig(fig, out_dir / "cohort_composition.png")


def plot_response_distributions(ds, dg, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(dg["y"].to_numpy(float), bins=40, color="#ee6677")
    axes[0].set_title("GDSC2 dose-response AUC")
    axes[0].set_xlabel("AUC")
    axes[0].set_ylabel("cell-line x drug pairs")
    axes[1].hist(ds["y"].to_numpy(float), bins=40, color="#228833")
    axes[1].set_title("Soragni viability")
    axes[1].set_xlabel("Viability_Score (% of vehicle)")
    axes[1].set_ylabel("organoid x drug pairs")
    fig.tight_layout()
    savefig(fig, out_dir / "response_distributions.png")


def plot_soragni_heatmap(ds, out_dir: Path) -> None:
    mat = ds.pivot_table(index="patient", columns="drug", values="y", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(min(0.45 * mat.shape[1] + 3, 18), 0.5 * mat.shape[0] + 2))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title("Soragni organoid x drug viability")
    fig.colorbar(im, ax=ax, label="Viability_Score (% of vehicle)")
    fig.tight_layout()
    savefig(fig, out_dir / "soragni_response_heatmap.png")


def plot_shared_panel(ds_cid, dg_cid, sb, out_dir: Path) -> None:
    shared = sorted(set(ds_cid["drug"].astype(str)) & set(dg_cid["drug"].astype(str)))
    cid2name = {str(a.pubchem_cid): a.drug_name for a in sb.drug_assays if a.pubchem_cid}
    g_auc = dg_cid[dg_cid["drug"].astype(str).isin(shared)].groupby("drug")["y"].mean()
    s_via = ds_cid[ds_cid["drug"].astype(str).isin(shared)].groupby("drug")["y"].mean()
    labels = [cid2name.get(str(c), str(c)) for c in shared]
    yidx = np.arange(len(shared))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * len(shared) + 2), sharey=True)
    axes[0].barh(yidx, [g_auc.get(c, np.nan) for c in shared], color="#ee6677")
    axes[0].set_yticks(yidx)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("GDSC2 mean AUC")
    axes[1].barh(yidx, [s_via.get(c, np.nan) for c in shared], color="#228833")
    axes[1].set_xlabel("Soragni mean viability")
    fig.suptitle(f"Shared drug panel (n={len(shared)} PubChem CIDs)")
    fig.tight_layout()
    savefig(fig, out_dir / "shared_panel.png")


def plot_head_invariance(csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv)
    df["label"] = df["head"].astype(str) + " / " + df["rep"].astype(str)
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(df)), 4.6))
    ax.bar(range(len(df)), df["interact"].to_numpy(float), color="#4477aa")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("interaction rho")
    ax.set_title("Head-invariance: interaction rho by head / representation")
    fig.tight_layout()
    savefig(fig, out_dir / "head_invariance.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None, help="default docs/figures/ (tracked, for README)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    # docs/figures/ is tracked (results/ is gitignored), so the README can embed
    # these and they regenerate in place.
    out_dir = Path(args.out_dir) if args.out_dir else repo / "docs" / "figures"

    sb = load_tranche("sarcoma", repo)
    gb = load_tranche("gdscv2", repo, cancer_type_filter=GDSC_SARCOMA)
    _, ds = build_sample_design(sb, "organoid", "viability")  # drug = Soragni name
    _, dg = build_sample_design(gb, "all", "auc")
    _, ds_cid = build_sample_design(sb, "organoid", "viability", drug_key="pubchem_cid")
    _, dg_cid = build_sample_design(gb, "all", "auc", drug_key="pubchem_cid")

    plot_cohort_composition(sb, gb, out_dir)
    plot_response_distributions(ds, dg, out_dir)
    plot_soragni_heatmap(ds, out_dir)
    plot_shared_panel(ds_cid, dg_cid, sb, out_dir)

    hi = repo / "results" / "head_invariance.csv"
    if hi.exists():
        plot_head_invariance(hi, out_dir)
    else:
        print(f"(skipped head_invariance figure: {hi} not found -- run the transfer scripts)")


if __name__ == "__main__":
    main()
