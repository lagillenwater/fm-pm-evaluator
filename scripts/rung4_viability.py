"""Rung 4: does the representation predict how much a drug kills a line, beyond the drug's average?

The readout changes from expression to viability. Viability is PRISM's primary repurposing
screen (DepMap 19Q4, public): replicate-collapsed log fold change in cell number at 2.5 uM,
5 days, for 568 DepMap lines x 4,686 compounds; here the Tahoe block's lines and the 108 drugs
found by name.

  Target      V[line, drug] minus the drug's mean over the training lines: the line-specific
              part of the viability response. A held-out line's viability is never seen.
  Ceiling     split-half over PRISM's replicates: V from half the replicates against V from
              the other half, residualised the same way, correlated across lines per drug.
  Predictors  from the line's baseline only (as in rung 1):
                mean            predicts 0 (the drug's average and nothing else)
                knn             mean residual of the 5 training lines nearest in baseline
                                expression
                pca_adj, nmf_adj  OLS of the training lines' residuals on 10 components of
                                baseline expression, applied to the held-out line
                stack_base      the same OLS on 10 PCs of the base Stack embedding of each
                                line's real DMSO cells (rung 2's block embeddings)
              and from the line's measured Tahoe expression response to the drug -- the
              readout crossing itself, does the transcriptional response predict killing:
                tahoe_prolif    mean log2FC over a fixed proliferation gene set, residualised by
                                the drug's mean over training lines, sign-flipped (less
                                proliferation signal -> more killing)
                tahoe_n_de      log number of DE genes, residualised likewise
              and from Stack's GENERATED response for the held-out line (rung 1's real-cell
              generation arrays: prompt/context/query all real cells):
                stack_gen_prolif_<ckpt>  the same proliferation summary on the predicted
                                log2FC -- the model's own cross-modality prediction
              The August lineage's readouts and models, carried over (Check 2):
                hallmark_<src>  the fixed signature readout: z-score each gene across the
                                response profiles, signed mean over four MSigDB Hallmark sets
                                (p53 pathway and apoptosis +1, E2F targets and G2-M checkpoint
                                -1), averaged; src = tahoe (measured) or gen_<ckpt> (Stack)
                szalai_<src>    the supervised L2 readout (Szalai et al., NAR 2019): ridge from
                                the z-scored response profile to the line-specific viability,
                                fit on the training lines' (line, drug) pairs, applied to the
                                held-out line's
                ridge_expr, lasso_expr, ridge_stack, lasso_stack  per-drug L2 / L1 regressions
                                of the line-specific viability on the full representation
                                (standardised baseline expression; the 1,600-d Stack embedding),
                                penalty chosen by inner cross-validation on the training lines
  Score       per drug, Pearson r across the held-out lines between predicted and observed
              residual; mean over drugs (each drug weighted once), beside the ceiling; a
              line-shuffled null for every column (the same predictions assigned to the wrong
              lines). A second table removes each line's mean over its drugs from observed and
              predicted first (the August lineage's "interaction" score), so a line's general
              sensitivity cannot carry a column.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rung1_simple_tables import LatentShift, N_COMPONENTS, log_cpm  # noqa: E402

FIGSHARE = "https://api.figshare.com/v2/articles/9393293"
FILES = ("primary-screen-replicate-collapsed-logfold-change.csv", "primary-screen-logfold-change.csv",
         "primary-screen-replicate-collapsed-treatment-info.csv", "primary-screen-replicate-treatment-info.csv",
         "primary-screen-cell-line-info.csv")
DROP_LINES = ("NA", "ACH-000628", "ACH-000311")
K_NEIGHBOURS = 5
SEED = 0
MIN_LINES = 8
PROLIFERATION = ("MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "CCNB1", "CCNA2", "CDK1",
                 "BUB1", "AURKA", "AURKB", "PLK1", "CENPF", "TYMS", "RRM2", "E2F1", "CCNE1", "FOXM1", "KIF11", "BIRC5")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def fetch(prism_dir: Path) -> dict[str, Path]:
    prism_dir.mkdir(parents=True, exist_ok=True)
    art = json.loads(subprocess.run(["curl", "-sSf", FIGSHARE], check=True, capture_output=True, text=True).stdout)
    urls = {f["name"]: f["download_url"] for f in art["files"]}
    out = {}
    for name in FILES:
        path = prism_dir / name
        if not path.exists():
            log(f"downloading {name}")
            subprocess.run(["curl", "-sSfL", "-o", str(path), urls[name]], check=True)
        out[name] = path
    return out


def prism_block(paths: dict[str, Path], lines: list[str], drugs: list[str]):
    """V[line, drug] (collapsed) and two replicate halves VA, VB (means over alternate replicates),
    over the Tahoe lines and drugs present in the primary screen at the standard dose."""
    cl = pd.read_csv(paths["primary-screen-cell-line-info.csv"])
    row_of = dict(zip(cl["row_name"], cl["depmap_id"]))
    ti = pd.read_csv(paths["primary-screen-replicate-collapsed-treatment-info.csv"])
    key_of = {norm(d.split(" (")[0]): d for d in drugs}
    ti["drug"] = ti["name"].astype(str).map(norm).map(key_of)
    ti = ti[ti["drug"].notna() & np.isclose(pd.to_numeric(ti["dose"], errors="coerce"), 2.5, atol=0.3)]
    ti = ti.sort_values("column_name").drop_duplicates("drug")  # one treatment column per drug
    col_of = dict(zip(ti["column_name"], ti["drug"]))
    V = pd.read_csv(paths["primary-screen-replicate-collapsed-logfold-change.csv"], index_col=0)
    V.index = V.index.map(row_of)
    V = V.loc[V.index.isin(lines), V.columns.isin(col_of)].rename(columns=col_of)
    V = V.groupby(level=0).mean()
    # replicate halves from the uncollapsed file: replicates of a treatment share broad_id, dose and screen
    ri = pd.read_csv(paths["primary-screen-replicate-treatment-info.csv"])
    ri = ri.merge(ti[["broad_id", "dose", "screen_id", "drug"]].drop_duplicates(), on=["broad_id", "dose", "screen_id"], how="inner")
    R = pd.read_csv(paths["primary-screen-logfold-change.csv"], index_col=0)
    R.index = R.index.map(row_of)
    R = R.loc[R.index.isin(lines)].groupby(level=0).mean()
    ri = ri[ri["column_name"].isin(R.columns)]
    ri["rep"] = ri.groupby("drug").cumcount() % 2
    halves = {}
    for h in (0, 1):
        cols = ri[ri["rep"] == h]
        M = pd.DataFrame({d: R[g["column_name"]].mean(1) for d, g in cols.groupby("drug")})
        halves[h] = M
    common = sorted(set(V.columns) & set(halves[0].columns) & set(halves[1].columns))
    V = V[common]
    VA, VB = halves[0][common].reindex(V.index), halves[1][common].reindex(V.index)
    n_rep = ri.groupby("drug").size().reindex(common)
    log(f"PRISM primary: {V.shape[0]} Tahoe lines x {V.shape[1]} drugs; replicates per drug median {int(n_rep.median())}; "
        f"missing entries {int(V.isna().sum().sum())} of {V.size}")
    return V, VA, VB, n_rep


def masked_corr_cols(P: np.ndarray, O: np.ndarray) -> np.ndarray:
    """Pearson per column (drug) across rows (lines), ignoring NaN pairs; NaN below MIN_LINES."""
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        m = np.isfinite(P[:, j]) & np.isfinite(O[:, j])
        if m.sum() >= MIN_LINES and np.std(P[m, j]) > 0 and np.std(O[m, j]) > 0:
            out[j] = np.corrcoef(P[m, j], O[m, j])[0, 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True, help="rung 2 bridge_block_embeddings.npz (real DMSO cells, base encoder)")
    ap.add_argument("--prism-dir", type=Path, required=True)
    ap.add_argument("--gen-dir", type=Path, default=None, help="rung 1 cache with real-cell gen_stack_*.npy")
    ap.add_argument("--genelist", type=Path, default=Path("stack-large/basecount_1000per_15000max.pkl"))
    ap.add_argument("--hallmark-gmt", type=Path, default=Path("data/static/hallmark_signatures.gmt"))
    ap.add_argument("--n-lasso-genes", type=int, default=2000, help="most variable baseline genes given to the lasso")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t = np.load(args.tensors, allow_pickle=True)
    Y, D, E = t["Y"], t["D"], t["E"]
    lines, drugs, genes = [str(x) for x in t["lines"]], [str(x) for x in t["drugs"]], pd.Index(t["genes"])
    keep = [ln for ln in lines if ln not in DROP_LINES]
    paths = fetch(args.prism_dir)
    V, VA, VB, n_rep = prism_block(paths, keep, drugs)
    vl = list(V.index)  # lines with viability
    vd = list(V.columns)
    li = [lines.index(ln) for ln in vl]
    dj = [drugs.index(d) for d in vd]
    n_l, n_d = len(vl), len(vd)

    # --- representations of the lines with viability
    L = log_cpm(E)[li]
    keep_g = L.std(0) > 0
    Z = (L[:, keep_g] - L[:, keep_g].mean(0)) / L[:, keep_g].std(0)
    latent = {k: LatentShift(k, L, keep_g) for k in ("pca", "nmf")}
    z = {k: v.encode(L) for k, v in latent.items()}
    z_emb = np.load(args.embeddings, allow_pickle=True)
    emb_line = {}
    for ln in set(z_emb["line"]):
        m = (z_emb["line"] == ln) & np.isin(z_emb["kind"], ["real_a", "real_b"])
        if m.any():
            emb_line[ln] = z_emb["emb"][m].mean(0)
    have_emb = np.array([ln in emb_line for ln in vl])
    if have_emb.any():
        from sklearn.decomposition import PCA

        Emb = np.stack([emb_line[ln] for ln in vl if ln in emb_line])
        z_stack = np.full((n_l, N_COMPONENTS), np.nan)
        z_stack[have_emb] = PCA(N_COMPONENTS, random_state=0).fit_transform(Emb)
        z["stack_base"] = z_stack
    log(f"{n_l} lines with viability, {int(have_emb.sum())} with a Stack embedding; {n_d} drugs")

    # --- the Tahoe expression response, summarised per (line, drug)
    prolif = np.flatnonzero(genes.isin(PROLIFERATION))
    Ysub = Y[np.ix_(li, dj)]
    tahoe_prolif = -np.nanmean(Ysub[:, :, prolif], axis=2)  # less proliferation signal -> more killing, so flip the sign
    tahoe_n_de = np.log1p(D[np.ix_(li, dj)].sum(2).astype(float))
    log(f"proliferation set: {len(prolif)} of {len(PROLIFERATION)} genes in the table")
    gen_prolif = {}
    if args.gen_dir is not None:
        from rung1_simple_tables import load_stack_genes

        stack_genes = load_stack_genes(args.genelist)
        if stack_genes is not None:
            sp = [k for k, g in enumerate(stack_genes) if g in set(PROLIFERATION)]
            for ck in ("stack_cytokine", "stack_sciplex"):
                p_ = args.gen_dir / f"gen_{ck}.npy"
                if p_.exists():
                    G = np.load(p_)[np.ix_(li, dj)]  # (line, drug, stack gene) predicted log2FC
                    with np.errstate(invalid="ignore"):
                        gen_prolif[f"stack_gen_prolif_{ck.split('_')[1]}"] = -np.nanmean(G[:, :, sp], axis=2)
            log(f"Stack generated responses: {sorted(gen_prolif)} over {len(sp)} proliferation genes in Stack's list")

    # --- response profiles over Stack's gene list (the space the generated arrays live in)
    from rung1_simple_tables import load_stack_genes as _lsg

    stack_genes = _lsg(args.genelist) or []
    sg_in = genes.get_indexer(stack_genes)
    resp_genes = [g for g, k in zip(stack_genes, sg_in) if k >= 0]
    responses = {"tahoe": Y[np.ix_(li, dj)][:, :, sg_in[sg_in >= 0]]}
    if args.gen_dir is not None:
        for ck in ("stack_cytokine", "stack_sciplex"):
            p_ = args.gen_dir / f"gen_{ck}.npy"
            if p_.exists():
                responses[f"gen_{ck.split('_')[1]}"] = np.load(p_)[np.ix_(li, dj)][:, :, sg_in >= 0]
    hallmark = {}
    if args.hallmark_gmt.exists():
        direction = {"HALLMARK_P53_PATHWAY": 1, "HALLMARK_APOPTOSIS": 1, "HALLMARK_E2F_TARGETS": -1, "HALLMARK_G2M_CHECKPOINT": -1}
        gpos = {g: k for k, g in enumerate(resp_genes)}
        for line_ in args.hallmark_gmt.read_text().splitlines():
            parts = line_.split("\t")
            if parts[0] in direction:
                hallmark[parts[0]] = (direction[parts[0]], np.array([gpos[g] for g in parts[2:] if g in gpos]))
        log(f"Hallmark sets over the response genes: { {k: len(v[1]) for k, v in hallmark.items()} }")

    def zscore_profiles(Rz: np.ndarray) -> np.ndarray:
        flat = Rz.reshape(-1, Rz.shape[-1])
        mu, sd = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0)
        return np.nan_to_num((Rz - mu) / np.where(sd > 0, sd, 1.0), nan=0.0)

    readout_hallmark, resp_z = {}, {}
    for src, Rr in responses.items():
        Zr = zscore_profiles(Rr)
        resp_z[src] = Zr
        if hallmark:
            readout_hallmark[f"hallmark_{src}"] = np.mean([d * Zr[:, :, idx].mean(2) for d, idx in hallmark.values()], axis=0)

    # --- full representations for the L1 / L2 per-drug models
    from sklearn.linear_model import LassoCV, RidgeCV

    X_expr = Z  # standardised log CPM, all genes with variance
    top = np.argsort(-L[:, keep_g].var(0))[: args.n_lasso_genes]
    X_expr_lasso = Z[:, top]
    X_stack = None
    if have_emb.all():
        Emb_all = np.stack([emb_line[ln] for ln in vl])
        X_stack = (Emb_all - Emb_all.mean(0)) / np.where(Emb_all.std(0) > 0, Emb_all.std(0), 1.0)
    reps_full = {"expr": (X_expr, X_expr_lasso)}
    if X_stack is not None:
        reps_full["stack"] = (X_stack, X_stack)
    alphas = np.logspace(-2, 5, 15)

    Vn = V.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    preds = {m: np.full((n_l, n_d), np.nan) for m in ("mean", "knn", "pca_adj", "nmf_adj", "stack_base", "tahoe_prolif", "tahoe_n_de", *gen_prolif,
                                                       *readout_hallmark, *[f"szalai_{s_}" for s_ in resp_z],
                                                       *[f"{pen}_{r_}" for r_ in reps_full for pen in ("ridge", "lasso")])}
    obs = np.full((n_l, n_d), np.nan)
    ceil_a, ceil_b = np.full((n_l, n_d), np.nan), np.full((n_l, n_d), np.nan)
    for i in range(n_l):
        train = np.array([k for k in range(n_l) if k != i])
        with np.errstate(invalid="ignore"):
            dmean = np.nanmean(Vn[train], axis=0)
            resid_tr = Vn[train] - dmean  # (train, drug)
        obs[i] = Vn[i] - dmean
        # the same residualisation of the replicate halves gives the ceiling's two sides
        ceil_a[i] = VA.to_numpy()[i] - np.nanmean(VA.to_numpy()[train], axis=0)
        ceil_b[i] = VB.to_numpy()[i] - np.nanmean(VB.to_numpy()[train], axis=0)
        preds["mean"][i] = 0.0
        c = np.corrcoef(Z[[i], :], Z[train])[0, 1:]
        preds["knn"][i] = np.nanmean(resid_tr[np.argsort(-c)[:K_NEIGHBOURS]], axis=0)
        for k, name in (("pca", "pca_adj"), ("nmf", "nmf_adj"), ("stack_base", "stack_base")):
            if k not in z or not np.isfinite(z[k][i]).all():
                continue
            ok = train[np.isfinite(z[k][train]).all(1)]
            X = np.column_stack([np.ones(len(ok)), z[k][ok]])
            R_ = np.nan_to_num(resid_tr[[np.where(train == o)[0][0] for o in ok]])
            beta = np.linalg.pinv(X) @ R_
            preds[name][i] = np.concatenate([[1.0], z[k][i]]) @ beta
        for name, S in (("tahoe_prolif", tahoe_prolif), ("tahoe_n_de", tahoe_n_de), *gen_prolif.items(), *readout_hallmark.items()):
            with np.errstate(invalid="ignore"):
                preds[name][i] = S[i] - np.nanmean(S[train], axis=0)
        # supervised L2 readout (szalai): response profile -> line-specific viability, fit on training pairs
        for src, Zr in resp_z.items():
            tr_pairs = np.isfinite(resid_tr)
            Xtr = Zr[train][tr_pairs]
            if len(Xtr) >= 20:
                from sklearn.linear_model import Ridge

                model = Ridge(alpha=1.0).fit(Xtr, resid_tr[tr_pairs])
                preds[f"szalai_{src}"][i] = model.predict(Zr[i])
        # per-drug L1 / L2 on the full representations
        for r_, (Xr, Xl) in reps_full.items():
            for j in range(n_d):
                ok = train[np.isfinite(resid_tr[:, j])]
                if len(ok) < MIN_LINES:
                    continue
                yj = Vn[ok, j] - dmean[j]
                preds[f"ridge_{r_}"][i, j] = RidgeCV(alphas=alphas).fit(Xr[ok], yj).predict(Xr[[i]])[0]
                try:
                    preds[f"lasso_{r_}"][i, j] = LassoCV(cv=3, alphas=np.logspace(-3, 1, 15), max_iter=3000, random_state=SEED).fit(Xl[ok], yj).predict(Xl[[i]])[0]
                except Exception:  # noqa: BLE001 -- a degenerate fold leaves the cell empty
                    pass
        log(f"fold {i + 1}/{n_l} {vl[i]}")

    rows = []
    r_ceil = masked_corr_cols(ceil_a, ceil_b)
    for m, P in preds.items():
        r = masked_corr_cols(P, obs) if m != "mean" else np.full(n_d, np.nan)
        # line-shuffled null: the same predictions on the wrong lines
        Pshuf = P[rng.permutation(n_l)]
        r_null = masked_corr_cols(Pshuf, obs) if m != "mean" else np.full(n_d, np.nan)
        for j, d in enumerate(vd):
            rows.append({"drug": d, "model": m, "r": r[j], "r_line_shuffled": r_null[j]})
    per = pd.DataFrame(rows)
    per["ceiling_split_half"] = per["drug"].map(dict(zip(vd, r_ceil)))
    per["n_lines"] = per["drug"].map(dict(zip(vd, np.isfinite(obs).sum(0))))
    per["n_replicates"] = per["drug"].map(n_rep)
    per.round(4).to_csv(args.out / "rung4_per_drug.csv", index=False)

    # the drug-mean's own worth, as global correlation with V (it cannot have a per-drug r)
    with np.errstate(invalid="ignore"):
        dm_pred = np.tile(np.nanmean(Vn, axis=0), (n_l, 1))
    m = np.isfinite(Vn)
    r_global_mean = float(np.corrcoef(dm_pred[m], Vn[m])[0, 1])

    # the August lineage's interaction score: each line's mean over its drugs removed from observed and predicted
    with np.errstate(invalid="ignore"):
        obs_int = obs - np.nanmean(obs, axis=1, keepdims=True)
    rows_int = []
    for m, P in preds.items():
        with np.errstate(invalid="ignore"):
            Pi = P - np.nanmean(P, axis=1, keepdims=True)
        r = masked_corr_cols(Pi, obs_int) if m != "mean" else np.full(n_d, np.nan)
        for j, d in enumerate(vd):
            rows_int.append({"drug": d, "model": m, "r_interaction": r[j]})
    ci_a = ceil_a - np.nanmean(ceil_a, axis=1, keepdims=True)
    ci_b = ceil_b - np.nanmean(ceil_b, axis=1, keepdims=True)
    r_ceil_int = masked_corr_cols(ci_a, ci_b)
    per_int = pd.DataFrame(rows_int)
    summ_int = per_int.groupby("model").agg(mean_r_interaction=("r_interaction", "mean"), median=("r_interaction", "median"), n_drugs=("r_interaction", "count"))
    summ_int.loc["ceiling_split_half", ["mean_r_interaction", "median", "n_drugs"]] = [np.nanmean(r_ceil_int), np.nanmedian(r_ceil_int), int(np.isfinite(r_ceil_int).sum())]
    summ_int["frac_of_ceiling"] = summ_int["mean_r_interaction"] / np.nanmean(r_ceil_int)
    summ_int.round(4).to_csv(args.out / "rung4_summary_interaction.csv")

    summ = per.groupby("model").agg(mean_r=("r", "mean"), median_r=("r", "median"), n_drugs=("r", "count"),
                                    frac_drugs_r_gt_0=("r", lambda s: float((s > 0).mean())),
                                    mean_r_line_shuffled=("r_line_shuffled", "mean"))
    summ.loc["ceiling_split_half", ["mean_r", "median_r", "n_drugs"]] = [np.nanmean(r_ceil), np.nanmedian(r_ceil), int(np.isfinite(r_ceil).sum())]
    summ["frac_of_ceiling"] = summ["mean_r"] / np.nanmean(r_ceil)
    summ.loc["drug_mean_global_r_with_V", "mean_r"] = r_global_mean
    order = ["ceiling_split_half", "mean", "knn", "pca_adj", "nmf_adj", "stack_base", "tahoe_prolif", "tahoe_n_de", *sorted(gen_prolif)]
    order += [m for m in preds if m not in order] + ["drug_mean_global_r_with_V"]
    summ = summ.reindex(order).round(4)
    summ.to_csv(args.out / "rung4_summary.csv")
    log("summary (per-drug r across held-out lines, mean over drugs):\n" + summ.to_string())
    log("interaction score (line mean removed):\n" + summ_int.reindex([o for o in order if o in summ_int.index]).round(4).to_string())
    (args.out / "run.json").write_text(json.dumps({"lines": vl, "n_drugs": n_d, "k": K_NEIGHBOURS, "n_components": N_COMPONENTS,
                                                   "proliferation_genes": list(PROLIFERATION), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
