"""Per-patient, biomarker-anchored baseline for Soragni drug response.

Precision-oncology critique of the global transcriptome models: clinical drug
choice is biomarker-driven and per-patient (CDK4 amp -> palbociclib, NF1 loss ->
MEK inhibitor, ...), not a pooled cohort-wide expression model. This eval asks,
for the actionable drugs in the Soragni panel: does a single known biomarker --
a target gene's expression, or a sensitizing/resistance alteration from WES --
rank organoids by their response to the matched drug, and does the global
PCA-of-expression model beat that single biomarker?

Biomarker readouts (per organoid):
  - expr : the drug target's log1p(CPM) expression, z-scored (all 17 organoids)
  - mut  : a non-intronic SNV in the gene (WES, ~15 organoids)
  - amp  : focal amplification (CNV Pathologist_Call == 'Amplification')
  - del  : single/two-copy deletion

``direction`` = "sensitize" means biomarker-positive is expected to be MORE
sensitive (lower Viability_Score); "resist" the opposite. These are PRE-SPECIFIED
biological priors -- edit BIOMARKERS to vet the biology; nothing here is selected
on the response data.

  uv run python scripts/biomarker_anchored.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fmharness.controls import permute_within_drug
from fmharness.data.loaders import canonicalize_patient_id, load_tranche
from fmharness.evaluation import (
    build_sample_design,
    cpm_bundle,
    grouped_cv_predict,
    within_drug_rho,
)
from fmharness.probe import SimpleProbe

SEED = 0

# Curated biomarker -> drug links. PRE-SPECIFIED priors -- edit to vet the biology.
BIOMARKERS = [
    {"drug": "Everolimus", "gene": "PIK3CA", "kind": "mut", "direction": "sensitize"},
    {"drug": "Rapamycin", "gene": "PIK3CA", "kind": "mut", "direction": "sensitize"},
    {"drug": "Trametinib", "gene": "NF1", "kind": "mut", "direction": "sensitize"},
    {"drug": "Palbociclib", "gene": "CDK4", "kind": "amp", "direction": "sensitize"},
    {"drug": "Palbociclib", "gene": "RB1", "kind": "del", "direction": "resist"},
    {"drug": "Lenvatinib", "gene": "FGFR4", "kind": "amp", "direction": "sensitize"},
    {"drug": "Dovitinib", "gene": "FGFR4", "kind": "amp", "direction": "sensitize"},
    {"drug": "Olaparib", "gene": "FANCA", "kind": "mut", "direction": "sensitize"},
    {"drug": "Gefitinib", "gene": "EGFR", "kind": "expr", "direction": "sensitize"},
    {"drug": "Lapatinib", "gene": "ERBB2", "kind": "expr", "direction": "sensitize"},
    {"drug": "Crizotinib", "gene": "MET", "kind": "expr", "direction": "sensitize"},
    {"drug": "Cabozantinib", "gene": "MET", "kind": "expr", "direction": "sensitize"},
    {"drug": "Linsitinib", "gene": "IGF1R", "kind": "expr", "direction": "sensitize"},
    {"drug": "Pazopanib", "gene": "KDR", "kind": "expr", "direction": "sensitize"},
    {"drug": "Ruxolitinib", "gene": "JAK2", "kind": "expr", "direction": "sensitize"},
    {"drug": "Dasatinib", "gene": "SRC", "kind": "expr", "direction": "sensitize"},
]


def _sym2entrez(repo: Path) -> dict[str, int]:
    g = pd.read_csv(repo / "data/raw/coderdata/genes.csv.gz").dropna(subset=["gene_symbol"])
    g = g.drop_duplicates("gene_symbol")
    return dict(zip(g["gene_symbol"].astype(str), g["entrez_id"].astype(int), strict=True))


def _wes_alterations(repo: Path) -> tuple[dict[str, dict[str, set[str]]], set[str]]:
    """Return ({kind -> {gene -> set(patient)}}, set of WES-profiled patients)."""
    snv = pd.read_parquet(repo / "data/raw/soragni/tables/snv.parquet")
    cnv = pd.read_parquet(repo / "data/raw/soragni/tables/cnv.parquet")
    snv = snv[snv["BestEffect_Variant_Classification"].astype(str) != "intron"]
    alt: dict[str, dict[str, set[str]]] = {"mut": {}, "amp": {}, "del": {}}
    for gene, sid in zip(
        snv["BestEffect_Hugo_Symbol"].astype(str), snv["Sample_ID"].astype(str), strict=True
    ):
        alt["mut"].setdefault(gene, set()).add(canonicalize_patient_id(sid))
    for gene, sid, call in zip(
        cnv["Gene"].astype(str),
        cnv["Sample_ID"].astype(str),
        cnv["Pathologist_Call"].astype(str),
        strict=True,
    ):
        p = canonicalize_patient_id(sid)
        if call == "Amplification":
            alt["amp"].setdefault(gene, set()).add(p)
        elif "deletion" in call.lower():
            alt["del"].setdefault(gene, set()).add(p)
    wes = {
        canonicalize_patient_id(s)
        for s in pd.concat([snv["Sample_ID"], cnv["Sample_ID"]]).astype(str)
    }
    return alt, wes


def _biomarker_series(
    bm: dict, x_log: pd.DataFrame, alt: dict, wes: set[str], sym2ent: dict[str, int]
) -> pd.Series | None:
    """Per-organoid biomarker value (index = patient); None if unavailable."""
    if bm["kind"] == "expr":
        ent = sym2ent.get(bm["gene"])
        col = str(ent) if ent is not None else None
        if col is None or col not in x_log.columns:
            return None
        v = x_log[col]
        return (v - v.mean()) / (v.std() or 1.0)  # z-score across organoids
    positive = alt[bm["kind"]].get(bm["gene"], set())
    return pd.Series({p: float(p in positive) for p in sorted(wes)})


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    bundle = cpm_bundle(load_tranche("sarcoma", repo))
    x_df, design = build_sample_design(bundle, "tumor", "viability")  # drug = Soragni name
    x_log = np.log1p(x_df)
    sym2ent = _sym2entrez(repo)
    alt, wes = _wes_alterations(repo)

    rows: list[dict] = []
    bm_preds: list[pd.DataFrame] = []  # stacked (patient, drug, y_true, y_pred) for the biomarkers
    actionable: list[dict] = []  # per-patient genomic positives
    for bm in BIOMARKERS:
        b = _biomarker_series(bm, x_log, alt, wes, sym2ent)
        resp = design[design["drug"] == bm["drug"]].set_index("patient")["y"]
        if b is None or resp.empty:
            rows.append({**bm, "n": 0, "n_pos": "-", "rho": np.nan, "fits_prior": "-"})
            continue
        common = [p for p in resp.index if p in b.index]
        if len(common) < 3:
            rows.append({**bm, "n": len(common), "n_pos": "-", "rho": np.nan, "fits_prior": "-"})
            continue
        bv = b.loc[common].to_numpy(float)
        yv = resp.loc[common].to_numpy(float)
        rho = float(spearmanr(bv, yv).statistic) if np.std(bv) > 0 else np.nan
        # sensitize -> expect biomarker up, viability down (rho<0); resist -> rho>0
        want_sign = -1.0 if bm["direction"] == "sensitize" else 1.0
        fits = "-" if np.isnan(rho) else ("yes" if np.sign(rho) == want_sign else "no")
        n_pos = "-" if bm["kind"] == "expr" else int((bv > 0).sum())
        rows.append({**bm, "n": len(common), "n_pos": n_pos, "rho": rho, "fits_prior": fits})

        # oriented prediction: y_pred tracks viability when the biomarker fits its
        # prior (want_sign is the expected sign of rho(biomarker, viability)).
        y_pred = want_sign * bv
        bm_preds.append(
            pd.DataFrame({"patient": common, "drug": bm["drug"], "y_true": yv, "y_pred": y_pred})
        )

        # per-patient actionability for genomic positives: where does the matched drug
        # rank among that organoid's screened drugs (percentile; 0 = most sensitive)?
        if bm["kind"] != "expr":
            for p in common:
                if b.loc[p] <= 0:
                    continue
                pdrugs = design[design["patient"] == p]
                pct = float((pdrugs["y"] < resp.loc[p]).mean())  # frac of drugs MORE viable
                actionable.append(
                    {
                        "patient": p,
                        "alteration": f"{bm['gene']} {bm['kind']}",
                        "drug": bm["drug"],
                        "viability": round(float(resp.loc[p]), 1),
                        "sensitivity_pctile": round(pct, 2),
                        "n_drugs": len(pdrugs),
                    }
                )

    bm_tab = pd.DataFrame(rows)
    print("=== Biomarker -> drug, within-organoid signal (Soragni, per drug) ===")
    print(bm_tab.to_string(index=False))

    print("\n=== Per-patient actionability (genomic biomarker-positive organoids) ===")
    print("  sensitivity_pctile = fraction of this organoid's drugs MORE potent than the")
    print("  matched drug (0.0 = the matched drug is its single most effective).")
    act = pd.DataFrame(actionable)
    empty_msg = "  (no genomic-positive organoids with a matched-drug response)"
    print(act.to_string(index=False) if not act.empty else empty_msg)

    # ---- head-to-head: single biomarker vs global PCA-of-expression (within Soragni) ----
    bm_all = pd.concat(bm_preds, ignore_index=True) if bm_preds else pd.DataFrame()
    bm_drugs = sorted({b["drug"] for b in BIOMARKERS})
    d_bm = design[design["drug"].isin(bm_drugs)].copy()
    factory = partial(SimpleProbe, n_components=10, per_drug=True, reducer="pca")
    n_splits = min(5, int(d_bm["patient"].nunique()))
    glob = grouped_cv_predict(factory, x_log, d_bm, n_splits=n_splits, seed=SEED)

    def perm_p(preds: pd.DataFrame, pred_col: str, n: int = 2000) -> tuple[float, float]:
        obs = within_drug_rho(preds, pred_col)
        null = np.array(
            [
                within_drug_rho(
                    preds.assign(
                        y_true=permute_within_drug(
                            preds["drug"], preds["y_true"], np.random.default_rng(SEED + 1 + i)
                        )
                    ),
                    pred_col,
                )
                for i in range(n)
            ]
        )
        return obs, float(np.mean(null >= obs))

    bm_rho, bm_p = perm_p(bm_all, "y_pred")
    gl_rho, gl_p = perm_p(glob, "y_resid")
    print("\n=== Head-to-head: drug-specific signal on the biomarker-anchored drugs ===")
    print(f"  biomarker (single gene)   within-drug rho = {bm_rho:+.3f}  (p={bm_p:.3f})")
    print(f"  global PCA-of-expression  within-drug rho = {gl_rho:+.3f}  (p={gl_p:.3f})")
    print(f"  (within-Soragni grouped CV, {n_splits} folds, {len(bm_drugs)} actionable drugs)")


if __name__ == "__main__":
    main()
