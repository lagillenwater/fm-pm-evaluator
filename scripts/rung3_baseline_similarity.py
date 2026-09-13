"""How similar are the untreated profiles of the same cell line on Tahoe and on LINCS?

For every Tahoe line that also exists in LINCS (matched by name): the Tahoe baseline (mean
DESeq2 baseMean, log CPM) and the LINCS baseline (mean log2 DMSO, Level 3) over the shared
landmark genes, each platform standardised gene-wise within itself (z across its own lines).
Reported per matched line: raw log-expression correlation (dominated by gene abundance, the
platform-offset reference), standardised correlation with its own LINCS profile, and the rank
of that own profile among all LINCS lines (1 = the line finds itself), plus the same rank
from the LINCS side among all Tahoe lines. Real Tahoe DMSO cells (pseudobulk of the downloaded
cells) are used as a second Tahoe baseline where given.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensors", type=Path, required=True)
    ap.add_argument("--l1000", type=Path, required=True, help="rung3_l1000.npz")
    ap.add_argument("--crosswalk", type=Path, required=True)
    ap.add_argument("--real-cells", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    drop = {"NA", "ACH-000628", "ACH-000311"}

    t = np.load(args.tensors, allow_pickle=True)
    E, lines, genes = t["E"], [str(x) for x in t["lines"]], pd.Index(t["genes"])
    keep = np.array([ln not in drop for ln in lines])
    l1 = np.load(args.l1000, allow_pickle=True)
    B, cells, lm = l1["B"], [str(x) for x in l1["cells"]], [str(x) for x in l1["genes"]]
    gi = genes.get_indexer(lm)
    ok = (gi >= 0) & np.isfinite(B).all(0)
    lm_t = gi[ok]
    Bl = B[:, ok]
    Lt = np.log1p(E / np.maximum(E.sum(1, keepdims=True), 1e-12) * 1e6)[:, lm_t]
    sources = {"pseudobulk_baseMean": Lt}
    if args.real_cells and args.real_cells.exists():
        import anndata as ad

        real = ad.read_h5ad(args.real_cells)
        ctrl = real[real.obs["kind"] == "control"]
        prof = np.stack([np.asarray(ctrl[ctrl.obs["line"] == ln].X.sum(0)).ravel() for ln in lines])
        sources["real_dmso_pseudobulk"] = np.log1p(prof / np.maximum(prof.sum(1, keepdims=True), 1e-12) * 1e6)[:, lm_t]

    cw = pd.read_csv(args.crosswalk)
    name = dict(zip(cw["line"].astype(str), cw["cell_name"].astype(str)))
    ckey = {norm(c): i for i, c in enumerate(cells)}
    matched = {ln: ckey[norm(name[ln])] for ln in lines if ln in name and norm(name[ln]) in ckey and ln not in drop}
    print(f"{int(ok.sum())} shared landmark genes; {len(matched)} matched lines: {sorted(name[l] for l in matched)}")

    Zl = zscore(Bl)
    rows, summaries = [], {}
    for src, Lt_src in sources.items():
        Zt = zscore(Lt_src[keep])
        Zt_full = np.full_like(Lt_src, np.nan)
        Zt_full[keep] = Zt
        C = np.corrcoef(np.vstack([Zt, Zl]))[: Zt.shape[0], Zt.shape[0]:]  # Tahoe lines x LINCS lines, standardised
        Craw = np.corrcoef(np.vstack([Lt_src[keep], Bl]))[: Zt.shape[0], Zt.shape[0]:]
        kept_lines = [ln for ln, k in zip(lines, keep) if k]
        for ln, ci in matched.items():
            i = kept_lines.index(ln)
            r_own, r_raw = C[i, ci], Craw[i, ci]
            rank_lincs = 1 + int((C[i] > r_own).sum())  # among LINCS lines, from the Tahoe side
            rank_tahoe = 1 + int((C[:, ci] > r_own).sum())  # among Tahoe lines, from the LINCS side
            rows.append({"source": src, "line": ln, "name": name[ln], "r_raw_log_expression": r_raw, "r_standardised_own": r_own,
                         "r_standardised_other_lincs_median": float(np.median(np.delete(C[i], ci))),
                         "rank_among_lincs_lines": rank_lincs, "n_lincs_lines": len(cells),
                         "rank_among_tahoe_lines": rank_tahoe, "n_tahoe_lines": len(kept_lines)})
        sub = pd.DataFrame([r for r in rows if r["source"] == src])
        summaries[src] = {
            "n_matched": len(sub), "r_raw_median": float(sub.r_raw_log_expression.median()),
            "r_standardised_own_median": float(sub.r_standardised_own.median()),
            "r_standardised_other_median": float(sub.r_standardised_other_lincs_median.median()),
            "rank1_fraction_among_lincs": float((sub.rank_among_lincs_lines == 1).mean()),
            "rank_median_among_lincs": float(sub.rank_among_lincs_lines.median()), "chance_rank_lincs": (len(cells) + 1) / 2,
            "rank1_fraction_among_tahoe": float((sub.rank_among_tahoe_lines == 1).mean()),
            "rank_median_among_tahoe": float(sub.rank_among_tahoe_lines.median()), "chance_rank_tahoe": (len(kept_lines) + 1) / 2,
            "all_pairs_standardised_r_median": float(np.median(C)),
        }
    per = pd.DataFrame(rows).round(4)
    per.to_csv(args.out / "rung3_baseline_similarity_per_line.csv", index=False)

    # --- the response itself: same line, same drug, both platforms -- the expected-transfer ceiling
    Y, D, H0, H1 = t["Y"], t["D"], t["H0"], t["H1"]
    drugs = [str(x) for x in t["drugs"]]
    R, l_drugs = l1["R"], [str(x) for x in l1["drugs"]]
    in_lm = np.zeros(len(genes), dtype=bool)
    in_lm[lm_t] = True
    rng = np.random.default_rng(0)
    resp = []
    for ln, ci in matched.items():
        i = lines.index(ln)
        others_t = [k for k, l2 in enumerate(lines) if keep[k] and l2 != ln]
        for jl, d in enumerate(l_drugs):
            if d not in drugs or not np.isfinite(R[ci, jl]).any():
                continue
            j = drugs.index(d)
            panel = D[i, j] & in_lm
            if panel.sum() < 10:
                continue
            y = Y[i, j][panel]
            same = np.full(len(genes), np.nan); same[lm_t] = R[ci, jl][ok]
            lincs_others = np.delete(R[:, jl, :], ci, axis=0)
            with np.errstate(invalid="ignore"):
                lmean = np.full(len(genes), np.nan); lmean[lm_t] = np.nanmean(lincs_others, axis=0)[ok]
                tmean = np.nanmean(Y[others_t, j], axis=0)
                jshuf = rng.choice([k for k in range(len(l_drugs)) if k != jl])
                shuf = np.full(len(genes), np.nan); shuf[lm_t] = R[ci, jshuf][ok]
            def r_(p):
                m = np.isfinite(p[panel]) & np.isfinite(y)
                return float(np.corrcoef(p[panel][m], y[m])[0, 1]) if m.sum() >= 10 else np.nan
            resp.append({"line": ln, "name": name[ln], "drug": d, "n_panel": int(panel.sum()),
                         "r_split_half": r_(H0[i, j]) if False else float(np.corrcoef(H0[i, j][panel], H1[i, j][panel])[0, 1]),
                         "r_same_line_lincs": r_(same), "r_same_line_lincs_wrong_drug": r_(shuf),
                         "r_lincs_mean_other_lines": r_(lmean), "r_tahoe_mean_other_lines": r_(tmean)})
    rp = pd.DataFrame(resp).round(4)
    rp.to_csv(args.out / "rung3_same_line_same_drug_response.csv", index=False)
    rs = rp.drop(columns=["line", "name", "drug"]).agg(["mean", "median", "count"]).round(4)
    rs.to_csv(args.out / "rung3_same_line_same_drug_summary.csv")
    print("\n=== same line, same drug, both platforms ===")
    print(rs.to_string())
    print("pairs:", len(rp), "| lines:", rp.line.nunique(), "| drugs:", rp.drug.nunique(),
          "| same-line LINCS beats wrong-drug in %.2f of pairs" % (rp.r_same_line_lincs > rp.r_same_line_lincs_wrong_drug).mean())
    summ = pd.DataFrame(summaries).round(4)
    summ.to_csv(args.out / "rung3_baseline_similarity_summary.csv")
    print(per.to_string(index=False))
    print(summ.to_string())


if __name__ == "__main__":
    main()
