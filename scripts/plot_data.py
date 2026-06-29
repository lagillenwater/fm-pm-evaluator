"""Data-representation figures for the README -- the data, no conclusions.

The raw inputs and outputs of the generation evaluation, no model / rho / p-value:
the Soragni cohort make-up, the Soragni viability distribution (the target), and the
Soragni organoid x drug viability matrix. GDSC2/DepMap is not a model input here (it
enters only as readout training labels), so it gets no figure. L1000 drug coverage is
printed as a text list rather than plotted.

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


def plot_soragni_viability(ds, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.hist(ds["y"].to_numpy(float), bins=40, color="#228833")
    ax.set_title("Soragni viability (prediction target)")
    ax.set_xlabel("Viability_Score (% of vehicle)")
    ax.set_ylabel("organoid x drug pairs")
    fig.tight_layout()
    savefig(fig, out_dir / "soragni_viability.png")


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


def report_l1000_coverage(ds_cid, sb, repo: Path) -> None:
    """Print which Soragni drugs have a real L1000 perturbation (a text list, no plot)."""
    from fmharness.l1000 import soragni_pert_map

    covered_cids = set(soragni_pert_map(repo).values())  # Soragni PubChem CIDs present in L1000
    cid2name = {str(a.pubchem_cid): a.drug_name for a in sb.drug_assays if a.pubchem_cid}
    cids = sorted(set(ds_cid["drug"].astype(str)))
    yes = sorted(cid2name.get(c, c) for c in cids if c in covered_cids)
    no = sorted(cid2name.get(c, c) for c in cids if c not in covered_cids)
    print(f"\nL1000 drug coverage: {len(yes)}/{len(cids)} Soragni drugs have an L1000 perturbation")
    print(f"  covered:     {', '.join(yes)}")
    print(f"  not in L1000: {', '.join(no)}")


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
    _, ds = build_sample_design(sb, "tumor", "viability")  # drug = Soragni name
    _, ds_cid = build_sample_design(sb, "tumor", "viability", drug_key="pubchem_cid")

    plot_cohort_composition(sb, out_dir)
    plot_soragni_viability(ds, out_dir)
    plot_soragni_heatmap(ds, out_dir)
    try:
        report_l1000_coverage(ds_cid, sb, repo)
    except Exception as e:
        print(f"(skipped l1000 coverage list: {e})")

    hi = repo / "results" / "head_invariance.csv"
    if hi.exists():
        plot_head_invariance(hi, out_dir)
    else:
        print(f"(skipped head_invariance figure: {hi} not found -- run the transfer scripts)")


if __name__ == "__main__":
    main()
