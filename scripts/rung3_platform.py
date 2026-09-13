"""Rung 3: what does crossing the measurement platform cost?

Rung 1's question with the training data on the other platform. Rung 1 showed that the best
predictor of a new line's response to a drug is the drug's average response over other lines,
at ~0.9 of the attainable correlation on the Tahoe screen. Rung 3 asks how much of that
survives when the average is taken on LINCS L1000 (bulk, 978 measured landmark genes,
different labs, different cells) and used to predict the response in a Tahoe line.

  L1000 side   LINCS 2020 Level 5 signatures (replicate-collapsed z-scores against plate
               vehicle) for the Tahoe drugs found by name, at 24 h and the dose nearest 10 uM
               per (cell line, drug); one mean signature per (L1000 line, drug) over the 978
               landmark genes. LINCS lines are matched to Tahoe lines by name so that when a
               Tahoe line also exists in LINCS it is LEFT OUT of the L1000 average -- the
               prediction is for a line the predictor has never seen on either platform.
  Tahoe side   the rung 1 block: Y[line, drug, gene] at 5 uM, own-DE panels, split-half halves;
               the two lines with empty DE tables (ACH-000628, ACH-000311) dropped.
  Panel        the held-out condition's own DE genes within the 978 landmarks -- the genes both
               platforms measure, where rung 0 put the reliable signal.
  Columns      tahoe_mean      drug mean over the other Tahoe lines (rung 1's predictor, the
                               within-platform reference)
               l1000_mean      drug mean over the L1000 lines (the held-out Tahoe line excluded
                               if present) -- the cross-platform prediction
               l1000_shuffled  the L1000 mean of a different drug (a fixed derangement) -- the
                               null: a bulk signature with the right platform but the wrong drug
               ceiling_panel   split-half r of the held-out condition on the same panel
  Statistic    Pearson r per held-out condition, mean per drug; retained fraction =
               r(l1000_mean) / r(tahoe_mean).

Stages: l1000 (download if absent, select, read, average -> l1000_response.npz + census),
score (the tables).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020"
FILES = {
    "gctx": "level5/level5_beta_trt_cp_n720216x12328.gctx",
    "siginfo": "siginfo_beta.txt",
    "geneinfo": "geneinfo_beta.txt",
    "cellinfo": "cellinfo_beta.txt",
}
TIME_H = 24.0
DOSE_UM = 10.0
DROP_LINES = ("NA", "ACH-000628", "ACH-000311")
MIN_PANEL = 10
CEILING = "ceiling_panel"
SEED = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def base_name(drug: str) -> str:
    """'Neratinib (maleate)' -> 'neratinib': the salt in parentheses is dropped before matching."""
    return norm(str(drug).split(" (")[0])


# ----------------------------------------------------------------------------- l1000


def fetch(l1000_dir: Path) -> dict[str, Path]:
    l1000_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, rel in FILES.items():
        path = l1000_dir / Path(rel).name
        if not path.exists():
            log(f"downloading {rel}")
            subprocess.run(["curl", "-sSfL", "-C", "-", "-o", str(path), f"{BASE}/{rel}"], check=True)
        out[key] = path
    return out


def l1000(args: argparse.Namespace) -> None:
    import h5py

    out = args.cache / "l1000_response.npz"
    if out.exists():
        log("l1000_response.npz present, skipping")
        return
    paths = fetch(args.l1000_dir)
    t = np.load(args.tensors, allow_pickle=True)
    drugs = [str(x) for x in t["drugs"]]

    gi = pd.read_csv(paths["geneinfo"], sep="\t", low_memory=False)
    lm = gi[gi["feature_space"].astype(str).str.lower() == "landmark"]
    lm_ids = lm["gene_id"].astype(str).to_numpy()
    lm_symbols = lm["gene_symbol"].astype(str).to_numpy()
    log(f"{len(lm_ids)} landmark genes")

    si = pd.read_csv(paths["siginfo"], sep="\t", low_memory=False)
    si = si[si["pert_type"].astype(str) == "trt_cp"].copy()
    si["name_key"] = si["cmap_name"].astype(str).map(norm) if "cmap_name" in si.columns else si["pert_iname"].astype(str).map(norm)
    key_of = {base_name(d): d for d in drugs}
    si = si[si["name_key"].isin(key_of)].copy()
    si["drug"] = si["name_key"].map(key_of)
    si["time_h"] = pd.to_numeric(si["pert_time"], errors="coerce")
    si["dose_um"] = pd.to_numeric(si["pert_dose"], errors="coerce")
    unit = si["pert_dose_unit"].astype(str).str.lower() if "pert_dose_unit" in si.columns else pd.Series("um", index=si.index)
    si = si[(si["time_h"] == TIME_H) & unit.str.contains("um|µm|uM".lower(), regex=True)]
    log(f"{si['drug'].nunique()} of {len(drugs)} Tahoe drugs found by name; {len(si)} signatures at {TIME_H:.0f} h")
    # the dose nearest 10 uM per (line, drug); every signature at that dose is averaged
    si["dose_gap"] = (si["dose_um"] - DOSE_UM).abs()
    best = si.groupby(["cell_iname", "drug"])["dose_gap"].transform("min")
    si = si[si["dose_gap"] == best].copy()
    census = si.groupby(["cell_iname", "drug"]).agg(n_sigs=("sig_id", "size"), dose_um=("dose_um", "first")).reset_index()
    log(f"{len(census)} (line, drug) signatures groups over {census['cell_iname'].nunique()} lines; "
        f"dose exactly {DOSE_UM} uM in {int(np.isclose(census['dose_um'], DOSE_UM).sum())} of them")

    with h5py.File(paths["gctx"], "r") as h:
        col_ids = h["0/META/COL/id"][:].astype(str)
        row_ids = h["0/META/ROW/id"][:].astype(str)
        want = si["sig_id"].astype(str).to_numpy()
        col_pos = pd.Index(col_ids).get_indexer(want)
        keep = col_pos >= 0
        si, col_pos = si[keep].reset_index(drop=True), col_pos[keep]
        row_pos = pd.Index(row_ids).get_indexer(lm_ids)
        assert (row_pos >= 0).all(), "landmark ids not all in the matrix"
        mat = h["0/DATA/0/matrix"]
        sigs_first = mat.shape[0] == len(col_ids)  # GCTX stores signatures x genes; guard the other orientation
        log(f"reading {len(col_pos)} signatures x {len(row_pos)} landmark genes from the matrix {mat.shape} "
            f"({'signatures x genes' if sigs_first else 'genes x signatures'})")
        order = np.argsort(col_pos)
        sorted_cols = col_pos[order]
        Z = np.empty((len(col_pos), len(row_pos)), dtype=np.float32)
        chunk = 500
        for s in range(0, len(sorted_cols), chunk):
            idx = sorted_cols[s:s + chunk]  # increasing, as h5py fancy indexing requires
            block = mat[idx, :] if sigs_first else mat[:, idx].T
            Z[order[s:s + chunk]] = block[:, row_pos]
            if (s // chunk) % 10 == 0:
                log(f"  {min(s + chunk, len(sorted_cols))}/{len(sorted_cols)} signatures read")
    si["row"] = np.arange(len(si))
    groups = si.groupby(["cell_iname", "drug"])["row"].apply(list)
    cells = sorted({c for c, _ in groups.index})
    drug_list = sorted({d for _, d in groups.index})
    R = np.full((len(cells), len(drug_list), len(row_pos)), np.nan, dtype=np.float32)
    ci, di = pd.Index(cells), pd.Index(drug_list)
    for (c, d), rows in groups.items():
        R[ci.get_loc(c), di.get_loc(d)] = Z[rows].mean(0)
    np.savez(out, R=R, cells=np.array(cells), drugs=np.array(drug_list), genes=lm_symbols)
    census.to_csv(args.out / "rung3_l1000_census.csv", index=False)
    log(f"wrote l1000_response.npz: {len(cells)} lines x {len(drug_list)} drugs x {len(row_pos)} genes; "
        f"lines per drug median {int(census.groupby('drug').size().median())}")


# ----------------------------------------------------------------------------- score


def masked_pearson(P: np.ndarray, O: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = M & np.isfinite(P) & np.isfinite(O)
    n = M.sum(1)
    P = np.where(M, P, 0.0)
    O = np.where(M, O, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = P.sum(1) / n
        mo = O.sum(1) / n
        pc = np.where(M, P - mp[:, None], 0.0)
        oc = np.where(M, O - mo[:, None], 0.0)
        r = (pc * oc).sum(1) / np.sqrt((pc**2).sum(1) * (oc**2).sum(1))
    r[n < MIN_PANEL] = np.nan
    return r, n


def score(args: argparse.Namespace) -> None:
    t = np.load(args.tensors, allow_pickle=True)
    Y, D, H0, H1 = t["Y"], t["D"], t["H0"], t["H1"]
    lines, drugs, genes = [str(x) for x in t["lines"]], [str(x) for x in t["drugs"]], pd.Index(t["genes"])
    keep_l = np.array([ln not in DROP_LINES for ln in lines])
    l1 = np.load(args.cache / "l1000_response.npz", allow_pickle=True)
    R, cells, l_drugs, l_genes = l1["R"], [str(x) for x in l1["cells"]], [str(x) for x in l1["drugs"]], [str(x) for x in l1["genes"]]

    gi = genes.get_indexer(l_genes)
    in_lm = np.zeros(len(genes), dtype=bool)
    in_lm[gi[gi >= 0]] = True
    lm_to_tahoe = gi  # position of each landmark gene in the Tahoe table (-1 if absent)
    log(f"{int(in_lm.sum())} of {len(l_genes)} landmark genes are in the Tahoe table")

    cw = pd.read_csv(args.crosswalk)
    tahoe_name = dict(zip(cw["line"].astype(str), cw["cell_name"].astype(str)))
    cell_key = {norm(c): i for i, c in enumerate(cells)}
    matched = {ln: cell_key[norm(tahoe_name[ln])] for ln in lines if ln in tahoe_name and norm(tahoe_name[ln]) in cell_key}
    log(f"{len(matched)} Tahoe lines also in L1000 (left out of the L1000 mean for their own predictions): "
        f"{sorted(tahoe_name[l] for l in matched)}")

    d_idx = [drugs.index(d) for d in l_drugs if d in drugs]
    l_pos = {d: j for j, d in enumerate(l_drugs)}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(d_idx))
    while (perm == np.arange(len(d_idx))).any():
        perm = rng.permutation(len(d_idx))
    log(f"{len(d_idx)} drugs on both platforms")

    def l1000_pred(j_l: int, exclude_cell: int | None) -> np.ndarray:
        block = R[:, j_l, :]
        if exclude_cell is not None:
            block = np.delete(block, exclude_cell, axis=0)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(block, axis=0)
        full = np.full(len(genes), np.nan, dtype=np.float32)
        full[lm_to_tahoe[lm_to_tahoe >= 0]] = m[lm_to_tahoe >= 0]
        return full

    rows = []
    for i, ln in enumerate(lines):
        if not keep_l[i] or not np.isfinite(Y[i]).any():
            continue
        train = np.flatnonzero(keep_l & (np.arange(len(lines)) != i))
        with np.errstate(invalid="ignore"):
            tahoe_mean = np.nanmean(Y[train], axis=0)
        ex = matched.get(ln)
        for k, j_t in enumerate(d_idx):
            d = drugs[j_t]
            panel = D[i, j_t] & in_lm
            preds = {
                "tahoe_mean": tahoe_mean[j_t],
                "l1000_mean": l1000_pred(l_pos[d], ex),
                "l1000_shuffled": l1000_pred(l_pos[drugs[d_idx[perm[k]]]], ex),
            }
            for m, p in preds.items():
                r, n = masked_pearson(p[None, :], Y[i, j_t][None, :], panel[None, :])
                rows.append({"line": ln, "drug": d, "model": m, "r": float(r[0]), "n_panel": int(n[0])})
            r, n = masked_pearson(H0[i, j_t][None, :], H1[i, j_t][None, :], panel[None, :])
            rows.append({"line": ln, "drug": d, "model": CEILING, "r": float(r[0]), "n_panel": int(n[0])})
        log(f"fold {i + 1}/{len(lines)} {ln}")
    per = pd.DataFrame(rows).dropna(subset=["r"])
    per.to_csv(args.out / "rung3_per_condition.csv", index=False)

    census = pd.read_csv(args.out / "rung3_l1000_census.csv") if (args.out / "rung3_l1000_census.csv").exists() else None
    wide = per.pivot_table(index="drug", columns="model", values="r", aggfunc="mean")
    table = per[per["model"] == "tahoe_mean"].groupby("drug").agg(n_lines=("line", "nunique"), n_panel_median=("n_panel", "median"))
    if census is not None:
        table = table.join(census.groupby("drug").agg(n_l1000_lines=("cell_iname", "nunique"), l1000_dose_um=("dose_um", "median")))
    table = table.join(wide[[CEILING, "tahoe_mean", "l1000_mean", "l1000_shuffled"]].rename(columns=lambda c: c if c == CEILING else f"r_{c}"))
    table["retained"] = table["r_l1000_mean"] / table["r_tahoe_mean"]
    table.sort_values("r_tahoe_mean", ascending=False).round(4).to_csv(args.out / "rung3_per_drug.csv")

    summary = per.groupby("model")["r"].agg(["mean", "median", "count"])
    split_half = float(summary.loc[CEILING, "mean"])
    attainable = np.sqrt(2 * split_half / (1 + split_half))
    summary["attainable_bound"] = attainable
    summary["frac_attainable"] = summary["mean"] / attainable
    summary = summary.reindex([CEILING, "tahoe_mean", "l1000_mean", "l1000_shuffled"]).round(4)
    summary.loc["retained_fraction", "mean"] = round(summary.loc["l1000_mean", "mean"] / summary.loc["tahoe_mean", "mean"], 4)
    w = per.pivot_table(index=["line", "drug"], columns="model", values="r")
    summary.loc["l1000_beats_shuffled_fraction", "mean"] = round(float((w["l1000_mean"] > w["l1000_shuffled"]).mean()), 4)
    summary.to_csv(args.out / "rung3_summary.csv")
    log("summary:\n" + summary.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("l1000", "score", "all"), default="all")
    ap.add_argument("--tensors", type=Path, required=True)
    ap.add_argument("--crosswalk", type=Path, default=Path("docs/tasks/rung1-held-out-prediction/rung1_line_crosswalk.csv"))
    ap.add_argument("--l1000-dir", type=Path, required=True, help="where the LINCS 2020 files are (downloaded if absent)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("l1000", "all"):
        l1000(args)
    if args.stage in ("score", "all"):
        score(args)
    (args.out / "run.json").write_text(json.dumps({"stage": args.stage, "time_h": TIME_H, "dose_um": DOSE_UM, "seed": SEED,
                                                   "drop_lines": DROP_LINES, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
