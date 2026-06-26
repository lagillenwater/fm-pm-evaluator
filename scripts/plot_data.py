"""Data-representation figures for the README -- the data, no conclusions.

The raw inputs and outputs of the generation evaluation, no model / rho / p-value:
the Soragni cohort make-up, the response distributions (Soragni viability = the
target; GDSC2 AUC = the readout training labels), the Soragni organoid x drug
viability matrix, and L1000 drug coverage (which Soragni drugs have a real L1000
perturbation, so can be scored). GDSC2/DepMap RNA-seq is not a model input here, so
no GDSC2 cohort or shared-panel figure -- GDSC2 enters only as readout labels.

Optionally renders the head-invariance figure from results/head_invariance.csv when
that file exists; it summarizes a produced metric table, still without prose.

  uv run python scripts/plot_data.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from _plotting import plt, savefig

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design

GDSC_SARCOMA = ["sarcoma"]


def _subtype_counts(bundle) -> Counter:
    return Counter(p.subtype or "unknown" for p in bundle.patients)


def plot_cohort_composition(sb, out_dir: Path) -> None:
    counts = _subtype_counts(sb).most_common()
    labels = [c[0] for c in counts]
    vals = [c[1] for c in counts]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(labels) + 1.5))
    ax.barh(range(len(labels)), vals, color="#4477aa")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("patients")
    ax.set_title(f"Soragni PDTO cohort (n={sum(vals)} patients)")
    fig.tight_layout()
    savefig(fig, out_dir / "cohort_composition.png")


def plot_response_distributions(ds, dg, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(ds["y"].to_numpy(float), bins=40, color="#228833")
    axes[0].set_title("Soragni viability (prediction target)")
    axes[0].set_xlabel("Viability_Score (% of vehicle)")
    axes[0].set_ylabel("organoid x drug pairs")
    axes[1].hist(dg["y"].to_numpy(float), bins=40, color="#ee6677")
    axes[1].set_title("GDSC2 AUC (readout training labels)")
    axes[1].set_xlabel("dose-response AUC")
    axes[1].set_ylabel("cell-line x drug pairs")
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


def plot_l1000_coverage(ds_cid, sb, repo: Path, out_dir: Path) -> None:
    """Which Soragni drugs have a real L1000 perturbation (so can be scored)."""
    from fmharness.l1000 import soragni_pert_map

    covered = set(soragni_pert_map(repo).values())  # Soragni PubChem CIDs present in L1000
    cid2name = {str(a.pubchem_cid): a.drug_name for a in sb.drug_assays if a.pubchem_cid}
    cids = sorted(set(ds_cid["drug"].astype(str)))
    rows = sorted(
        ((cid2name.get(c, c), c in covered) for c in cids), key=lambda r: (not r[1], r[0])
    )
    names = [r[0] for r in rows]
    colors = ["#228833" if r[1] else "#cccccc" for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1.5))
    ax.barh(range(len(names)), [1] * len(names), color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xticks([])
    n_cov = sum(r[1] for r in rows)
    ax.set_title(f"Soragni drugs covered by L1000 ({n_cov}/{len(names)}); green = covered")
    fig.tight_layout()
    savefig(fig, out_dir / "l1000_coverage.png")


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
    _, ds = build_sample_design(sb, "tumor", "viability")  # drug = Soragni name
    _, dg = build_sample_design(gb, "all", "auc")  # GDSC2 AUC = readout training labels
    _, ds_cid = build_sample_design(sb, "tumor", "viability", drug_key="pubchem_cid")

    plot_cohort_composition(sb, out_dir)
    plot_response_distributions(ds, dg, out_dir)
    plot_soragni_heatmap(ds, out_dir)
    try:
        plot_l1000_coverage(ds_cid, sb, repo, out_dir)
    except Exception as e:
        print(f"(skipped l1000_coverage figure: {e})")

    hi = repo / "results" / "head_invariance.csv"
    if hi.exists():
        plot_head_invariance(hi, out_dir)
    else:
        print(f"(skipped head_invariance figure: {hi} not found -- run the transfer scripts)")


if __name__ == "__main__":
    main()
