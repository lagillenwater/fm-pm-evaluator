"""Rung 4: does a predicted expression response tell how much a drug kills a line?

The readout changes from expression to viability. Viability is PRISM's primary repurposing
screen (DepMap 19Q4, public): replicate-collapsed log fold change in cell number at 2.5 uM,
5 days, for 568 DepMap lines x 4,686 compounds; here the Tahoe block's lines and the 108 drugs
found by name (37 lines x 69 drugs).

Structure -- rows are WHAT PREDICTS THE EXPRESSION RESPONSE, tables are HOW THAT RESPONSE IS READ
INTO VIABILITY:

  Rows (each yields a predicted log2FC profile for the held-out line, exactly as in rung 1,
  fit on the other lines of the full Tahoe block, viability never seen):
    mean            the drug's mean response over the block -- a per-drug constant, no line
                    information (a leave-one-out mean would leak the held-out line's own
                    response, so every row's additive base is this constant and only the
                    departures come from the other lines)
    knn             mean response of the 5 lines nearest in baseline expression
    pca, nmf        rung 1's encode -> shift -> decode departure added to the mean
    stack_base      the base Stack encoder's embedding of the line's real DMSO cells (rung 2),
                    10 PCs, gene-wise OLS of the response departure on them, added to the mean
                    (the base checkpoint has no decoder, so this is its only form)
    stack_cytokine, stack_sciplex   in-context generation on real cells (rung 1's arrays)
    measured        the line's actual Tahoe response -- the reference: what the readout can do
                    when the response is known

  Tables (the readout, applied identically to every row's predicted response):
    proliferation   minus the mean z-scored log2FC over MSigDB Hallmark E2F targets and G2-M
                    checkpoint (proliferation programme down -> killing up); no fitting
    ridge, lasso    per drug, L2 / L1 regression of the line-specific viability on the predicted
                    response over the drug's DE genes (union of the training lines' padj < 0.05
                    calls, capped at the 3,000 most often called), penalty tuned by inner
                    cross-validation, no intercept (targets centred on the training lines)

  Target      V[line, drug] minus the drug's mean over the training lines (line-specific part).
  Scores      overall: per drug, Pearson r across the held-out lines, mean over drugs.
              interaction: the same after removing each line's mean over its drugs from
              observed and predicted (a line's general sensitivity cannot carry it).
  Ceiling     replicate split-half of the target, per drug across lines.
  Null        matched: viability shuffled across lines within each drug and the whole procedure
              rerun (``--permute-seed``), because leave-one-line-out predictions from a
              many-feature regression are anti-correlated with the held-out value by
              construction. ``rung4_combine.py`` reports, per table, each row's overall and
              interaction r, their empirical p, and each minus its null mean.
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
from rung1_simple_tables import LatentShift, N_COMPONENTS, load_stack_genes, log_cpm  # noqa: E402

FIGSHARE = "https://api.figshare.com/v2/articles/9393293"
FILES = ("primary-screen-replicate-collapsed-logfold-change.csv", "primary-screen-logfold-change.csv",
         "primary-screen-replicate-collapsed-treatment-info.csv", "primary-screen-replicate-treatment-info.csv",
         "primary-screen-cell-line-info.csv")
DROP_LINES = ("NA", "ACH-000628", "ACH-000311")
K_NEIGHBOURS = 5
SEED = 0
MIN_LINES = 8
ROWS = ("mean", "knn", "pca", "nmf", "stack_base", "stack_cytokine", "stack_sciplex", "measured")
READOUTS = ("proliferation", "ridge", "lasso")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ----------------------------------------------------------------------------- PRISM


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
    ti = ti.sort_values("column_name").drop_duplicates("drug")
    col_of = dict(zip(ti["column_name"], ti["drug"]))
    V = pd.read_csv(paths["primary-screen-replicate-collapsed-logfold-change.csv"], index_col=0)
    V.index = V.index.map(row_of)
    V = V.loc[V.index.isin(lines), V.columns.isin(col_of)].rename(columns=col_of)
    V = V.groupby(level=0).mean()
    ri = pd.read_csv(paths["primary-screen-replicate-treatment-info.csv"])
    ri = ri.merge(ti[["broad_id", "dose", "screen_id", "drug"]].drop_duplicates(), on=["broad_id", "dose", "screen_id"], how="inner")
    R = pd.read_csv(paths["primary-screen-logfold-change.csv"], index_col=0)
    R.index = R.index.map(row_of)
    R = R.loc[R.index.isin(lines)].groupby(level=0).mean()
    ri = ri[ri["column_name"].isin(R.columns)]
    ri["rep"] = ri.groupby("drug").cumcount() % 2
    halves = {h: pd.DataFrame({d: R[g["column_name"]].mean(1) for d, g in ri[ri["rep"] == h].groupby("drug")}) for h in (0, 1)}
    common = sorted(set(V.columns) & set(halves[0].columns) & set(halves[1].columns))
    V = V[common]
    VA, VB = halves[0][common].reindex(V.index), halves[1][common].reindex(V.index)
    n_rep = ri.groupby("drug").size().reindex(common)
    log(f"PRISM primary: {V.shape[0]} Tahoe lines x {V.shape[1]} drugs; replicates per drug median {int(n_rep.median())}; "
        f"missing entries {int(V.isna().sum().sum())} of {V.size}")
    return V, VA, VB, n_rep


# ----------------------------------------------------------------------------- predicted responses


def predicted_responses(Y: np.ndarray, E: np.ndarray, lines: list[str], emb_line: dict[str, np.ndarray], gen: dict[str, np.ndarray],
                        keep_lines: np.ndarray) -> dict[str, np.ndarray]:
    """Rung 1's predicted log2FC for every (held-out line, drug, gene), each fit on the other lines
    of the block. Returns arrays (line, drug, gene) over the lines in ``keep_lines`` order."""
    n_l, n_d, n_g = Y.shape
    L = log_cpm(E)
    keep_g = L.std(0) > 0
    Z = (L[:, keep_g] - L[:, keep_g].mean(0)) / L[:, keep_g].std(0)
    L_T = log_cpm(E[:, None, :] * np.exp2(np.nan_to_num(Y, nan=0.0)))
    latent = {k: LatentShift(k, L, keep_g) for k in ("pca", "nmf")}
    z = {k: v.encode(L) for k, v in latent.items()}
    t_lat = {k: v.encode(L_T) for k, v in latent.items()}
    z_stack = None
    have = np.array([ln in emb_line for ln in lines])
    if have.all():
        from sklearn.decomposition import PCA

        z_stack = PCA(N_COMPONENTS, random_state=0).fit_transform(np.stack([emb_line[ln] for ln in lines]))
    out = {r: np.full((len(keep_lines), n_d, n_g), np.nan, dtype=np.float32) for r in ("mean", "knn", "pca", "nmf", "stack_base")}
    valid = np.array([ln not in DROP_LINES and np.isfinite(Y[k]).any() for k, ln in enumerate(lines)])
    with np.errstate(invalid="ignore"):
        full_mean = np.nanmean(Y[valid], axis=0)  # a per-drug constant across lines: carries no line information
    for o, i in enumerate(keep_lines):
        train = np.flatnonzero(valid & (np.arange(n_l) != i))
        with np.errstate(invalid="ignore"):
            mean_d = np.nanmean(Y[train], axis=0)  # the training-line mean the departures are fit against
        out["mean"][o] = full_mean
        c = np.corrcoef(Z[[i], :], Z[train])[0, 1:]
        out["knn"][o] = full_mean + (np.nanmean(Y[train[np.argsort(-c)[:K_NEIGHBOURS]]], axis=0) - mean_d)
        for k in ("pca", "nmf"):
            X = np.column_stack([np.ones(len(train)), z[k][train]])
            beta = np.linalg.pinv(X) @ t_lat[k][train].reshape(len(train), -1)
            t_hat = (np.concatenate([[1.0], z[k][i]]) @ beta).reshape(n_d, N_COMPONENTS)
            fitted = (X @ beta).reshape(len(train), n_d, N_COMPONENTS) - z[k][train][:, None, :]
            out[k][o] = full_mean + latent[k].decode_shift((t_hat - z[k][i]) - fitted.mean(0), n_g)
        if z_stack is not None:
            dep = np.nan_to_num(Y[train] - mean_d)  # (train, drug, gene) departures
            Xc = z_stack[train] - z_stack[train].mean(0)
            beta = np.linalg.pinv(Xc) @ dep.reshape(len(train), -1)
            out["stack_base"][o] = full_mean + ((z_stack[i] - z_stack[train].mean(0)) @ beta).reshape(n_d, n_g)
        log(f"  predicted responses {o + 1}/{len(keep_lines)} {lines[i]}")
    for name, arr in gen.items():
        out[name] = arr[keep_lines]
    out["measured"] = Y[keep_lines]
    return out


# ----------------------------------------------------------------------------- scoring


def masked_corr_cols(P: np.ndarray, O: np.ndarray) -> np.ndarray:
    out = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        m = np.isfinite(P[:, j]) & np.isfinite(O[:, j])
        if m.sum() >= MIN_LINES and np.std(P[m, j]) > 0 and np.std(O[m, j]) > 0:
            out[j] = np.corrcoef(P[m, j], O[m, j])[0, 1]
    return out


def zscore_profiles(Rz: np.ndarray) -> np.ndarray:
    flat = Rz.reshape(-1, Rz.shape[-1])
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0)
    return np.nan_to_num((Rz - mu) / np.where(sd > 0, sd, 1.0), nan=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True, help="rung 2 bridge_block_embeddings.npz (real DMSO cells, base encoder)")
    ap.add_argument("--prism-dir", type=Path, required=True)
    ap.add_argument("--gen-dir", type=Path, required=True, help="rung 1 cache with real-cell gen_stack_*.npy")
    ap.add_argument("--genelist", type=Path, default=Path("stack-large/basecount_1000per_15000max.pkl"))
    ap.add_argument("--hallmark-gmt", type=Path, default=Path("data/static/hallmark_signatures.gmt"))
    ap.add_argument("--max-de-genes", type=int, default=3000)
    ap.add_argument("--permute-seed", type=int, default=None, help="matched null: shuffle viability across lines within each drug")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t = np.load(args.tensors, allow_pickle=True)
    Y, D, E = t["Y"], t["D"], t["E"]
    lines, drugs, genes = [str(x) for x in t["lines"]], [str(x) for x in t["drugs"]], pd.Index(t["genes"])
    stack_genes = load_stack_genes(args.genelist) or []
    sg_in = genes.get_indexer(stack_genes)
    resp_cols = sg_in[sg_in >= 0]
    resp_genes = [g for g, k in zip(stack_genes, sg_in) if k >= 0]
    Yr, Dr = Y[:, :, resp_cols], D[:, :, resp_cols]  # the response space every row shares
    gen = {}
    for ck in ("stack_cytokine", "stack_sciplex"):
        p_ = args.gen_dir / f"gen_{ck}.npy"
        if p_.exists():
            gen[ck] = np.load(p_)[:, :, sg_in >= 0]
    z_emb = np.load(args.embeddings, allow_pickle=True)
    emb_line = {ln: z_emb["emb"][(z_emb["line"] == ln) & np.isin(z_emb["kind"], ["real_a", "real_b"])].mean(0)
                for ln in set(z_emb["line"]) if ((z_emb["line"] == ln) & np.isin(z_emb["kind"], ["real_a", "real_b"])).any()}

    paths = fetch(args.prism_dir)
    keep = [ln for ln in lines if ln not in DROP_LINES]
    V, VA, VB, n_rep = prism_block(paths, keep, drugs)
    if args.permute_seed is not None:
        prng = np.random.default_rng(args.permute_seed)
        for c in V.columns:
            V[c] = prng.permutation(V[c].to_numpy())
        log(f"matched null: viability shuffled across lines within each drug (seed {args.permute_seed})")
    vl, vd = list(V.index), list(V.columns)
    li = np.array([lines.index(ln) for ln in vl])
    dj = np.array([drugs.index(d) for d in vd])
    n_l, n_d = len(vl), len(vd)
    log(f"{n_l} lines with viability, {sum(ln in emb_line for ln in vl)} with a Stack embedding; {n_d} drugs; "
        f"{len(resp_genes)} response genes; generated arrays {sorted(gen)}")

    resp = predicted_responses(Yr, E[:, resp_cols], lines, emb_line, gen, li)  # baselines over the same response genes
    resp = {r: arr[:, dj, :] for r, arr in resp.items()}  # (viability line, viability drug, gene)
    rows_present = [r for r in ROWS if r in resp]
    Dsub = Dr[np.ix_(li, dj)]

    gpos = {g: k for k, g in enumerate(resp_genes)}
    prolif_idx = np.array(sorted({gpos[g] for line_ in args.hallmark_gmt.read_text().splitlines()
                                  for g in (line_.split("\t")[2:] if line_.split("\t")[0] in ("HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT") else [])
                                  if g in gpos}))
    log(f"proliferation gene sets (E2F targets + G2-M checkpoint): {len(prolif_idx)} genes in the response space")
    resp_z = {r: zscore_profiles(arr) for r, arr in resp.items()}
    prolif = {r: -Zr[:, :, prolif_idx].mean(2) for r, Zr in resp_z.items()}

    from sklearn.linear_model import LassoCV, RidgeCV

    alphas_ridge, alphas_lasso = np.logspace(-2, 5, 15), np.logspace(-3, 1, 15)
    Vn = V.to_numpy(dtype=float)
    preds = {(ro, r): np.full((n_l, n_d), np.nan) for ro in READOUTS for r in rows_present}
    obs = np.full((n_l, n_d), np.nan)
    ceil_a, ceil_b = np.full((n_l, n_d), np.nan), np.full((n_l, n_d), np.nan)
    for i in range(n_l):
        train = np.array([k for k in range(n_l) if k != i])
        with np.errstate(invalid="ignore"):
            dmean = np.nanmean(Vn[train], axis=0)
            resid_tr = Vn[train] - dmean
        obs[i] = Vn[i] - dmean
        ceil_a[i] = VA.to_numpy()[i] - np.nanmean(VA.to_numpy()[train], axis=0)
        ceil_b[i] = VB.to_numpy()[i] - np.nanmean(VB.to_numpy()[train], axis=0)
        for r in rows_present:
            with np.errstate(invalid="ignore"):
                preds[("proliferation", r)][i] = prolif[r][i] - np.nanmean(prolif[r][train], axis=0)
        for j in range(n_d):
            ok = train[np.isfinite(resid_tr[:, j])]
            if len(ok) < MIN_LINES:
                continue
            calls = Dsub[ok, j].sum(0)
            de = np.flatnonzero(calls > 0)
            if len(de) < 5:
                continue
            if len(de) > args.max_de_genes:
                de = de[np.argsort(-calls[de])[: args.max_de_genes]]
            yj = Vn[ok, j] - dmean[j]
            for r in rows_present:
                Zr = resp_z[r]
                Xtr, Xte = Zr[ok, j][:, de], Zr[i, j][de][None, :]
                if Xtr.std(0).max() < 1e-4:  # z-scored features; float32 rounding leaves ~1e-7
                    continue  # the mean row: identical features for every line, nothing to fit
                preds[("ridge", r)][i, j] = RidgeCV(alphas=alphas_ridge, fit_intercept=False).fit(Xtr, yj).predict(Xte)[0]
                try:
                    preds[("lasso", r)][i, j] = LassoCV(cv=3, alphas=alphas_lasso, max_iter=3000, random_state=SEED, fit_intercept=False).fit(Xtr, yj).predict(Xte)[0]
                except Exception:  # noqa: BLE001
                    pass
        log(f"fold {i + 1}/{n_l} {vl[i]}")

    with np.errstate(invalid="ignore"):
        obs_int = obs - np.nanmean(obs, axis=1, keepdims=True)
    r_ceil = masked_corr_cols(ceil_a, ceil_b)
    r_ceil_int = masked_corr_cols(ceil_a - np.nanmean(ceil_a, axis=1, keepdims=True), ceil_b - np.nanmean(ceil_b, axis=1, keepdims=True))
    per_rows, summ_rows = [], []
    for (ro, r), P in preds.items():
        overall = masked_corr_cols(P, obs)
        with np.errstate(invalid="ignore"):
            Pi = P - np.nanmean(P, axis=1, keepdims=True)
        inter = masked_corr_cols(Pi, obs_int)
        for j, d in enumerate(vd):
            per_rows.append({"readout": ro, "row": r, "drug": d, "overall_r": overall[j], "interaction_r": inter[j],
                             "ceiling_overall": r_ceil[j], "ceiling_interaction": r_ceil_int[j]})
        summ_rows.append({"readout": ro, "row": r, "overall_r": np.nanmean(overall), "interaction_r": np.nanmean(inter),
                          "n_drugs": int(np.isfinite(overall).sum())})
    per = pd.DataFrame(per_rows)
    per.round(4).to_csv(args.out / "rung4_per_drug.csv", index=False)
    summ = pd.DataFrame(summ_rows)
    summ.loc[len(summ)] = {"readout": "ceiling", "row": "split_half", "overall_r": np.nanmean(r_ceil), "interaction_r": np.nanmean(r_ceil_int),
                           "n_drugs": int(np.isfinite(r_ceil).sum())}
    summ.round(4).to_csv(args.out / "rung4_summary.csv", index=False)
    log("summary:\n" + summ.round(4).to_string(index=False))
    (args.out / "run.json").write_text(json.dumps({"lines": vl, "n_drugs": n_d, "rows": rows_present, "readouts": list(READOUTS),
                                                   "proliferation_genes": int(len(prolif_idx)), "max_de_genes": args.max_de_genes,
                                                   "permute_seed": args.permute_seed, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
