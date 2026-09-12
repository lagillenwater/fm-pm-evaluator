"""Rung 0 noise per drug and per line, and rung 1 held-out prediction, as five tables.

Everything runs on Alpine from what is already on scratch; nothing is downloaded.

Rung 0 -- two group-bys of the promoted per-pair split-half table at 5 uM:
  rung0_noise_per_drug.csv   one row per drug, split-half r averaged over its lines
  rung0_noise_per_line.csv   one row per line, split-half r averaged over its drugs
``noise_de`` is the r over the DE genes and is the ceiling every rung 1 row reads against.

Rung 1 -- Y[line, drug, gene] is the 5 uM log2 fold change averaged over plates (the two
half means of the rung 0 cache averaged), D the padj < 0.05 call on any plate. Three
holdout schemes, each a fold loop with vectorised array algebra inside:

  line      line l* held out of everything. Baseline = drug's mean over the other lines.
            Panel = union of D for that drug over the other lines. Read against noise per drug.
  drugcell  the single cell (l, d*) held out; d* still seen in the other lines. Baseline =
            l's mean over its other drugs. Panel = union of D for l over its other drugs.
            Read against noise per line.
  drugwhole d* held out of every line. Baseline only (no line representation can speak to
            a drug no line has seen); by construction the same numbers as drugcell's mean.

Models predict the departure from the baseline (the interaction) and share one equation:
  mean         departure 0
  knn          average departure of the 5 nearest training lines in baseline expression
  pca / nmf    OLS of the training lines' departure on 10 components of baseline expression
  stack_*      the same OLS on 10 PCs of a Stack checkpoint's embedding of the baseline
Score = Pearson r between predicted and observed Y on the panel, per held-out condition,
then averaged per drug (scheme line) or per line (the drug schemes).

Baseline expression per line is the mean DESeq2 ``baseMean`` over that line's contrasts in a
strided subset of the raw shards -- treated and vehicle pseudobulks averaged over many drugs, which
is the line's expression level on the screen's own platform.

Stages: build (duckdb -> tensors + a baseline h5ad), embed (three Stack checkpoints, each
skipped with a note if it fails), score (the tables). ``--stage all`` runs them in order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

DE_SUBSTRING = "pseudobulk_differential_expression"
DOSE = 5.0
ALPHA = 0.05
K_NEIGHBOURS = 5
N_COMPONENTS = 10
MIN_PANEL = 10
MODELS = ("mean", "knn", "pca", "nmf", "stack_base", "stack_cytokine", "stack_sciplex")
STACK_VERSIONS = {"stack_base": "CKPT_BASE", "stack_cytokine": "CKPT_CYTOKINE", "stack_sciplex": "CKPT_DRUG"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- rung 0


def rung0_tables(per_pair_csv: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(per_pair_csv)
    d = d[d["dose"].astype(float) == DOSE]
    agg = dict(noise_all=("r", "mean"), noise_de=("r_responder", "mean"), n_de_median=("n_responders", "median"))
    per_drug = d.groupby("drug").agg(n_lines=("patient", "nunique"), **agg).sort_values("noise_de", ascending=False)
    per_line = d.groupby("patient").agg(n_drugs=("drug", "nunique"), **agg).sort_values("noise_de", ascending=False)
    per_line.index.name = "line"
    per_drug.round(4).to_csv(out / "rung0_noise_per_drug.csv")
    per_line.round(4).to_csv(out / "rung0_noise_per_line.csv")
    log(f"rung 0: {len(per_drug)} drugs, {len(per_line)} lines at {DOSE} uM")
    return per_drug, per_line


# ----------------------------------------------------------------------------- build


def _dense(df: pd.DataFrame, value: str, lines: pd.Index, drugs: pd.Index, genes: pd.Index, fill) -> np.ndarray:
    li = lines.get_indexer(df["patient"])
    di = drugs.get_indexer(df["drug"])
    gi = genes.get_indexer(df["gene_name"])
    arr = np.full((len(lines), len(drugs), len(genes)), fill, dtype=np.float32 if fill is np.nan else bool)
    arr[li, di, gi] = df[value].to_numpy()
    return arr


def build(args: argparse.Namespace) -> None:
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.duckdb_memory}'; SET threads={args.duckdb_threads};")
    con.execute(f"SET temp_directory='{args.cache / 'duckdb_tmp'}';")

    frames = sorted(str(p) for p in args.rung0_cache.glob("frame_*.parquet"))
    if not frames:
        raise FileNotFoundError(f"no frame_*.parquet under {args.rung0_cache}")
    log(f"reading {len(frames)} rung 0 frame slices at {DOSE} uM")
    de = con.execute(
        f"""SELECT patient, drug, gene_name,
                   (lfc0 + lfc1) / 2 AS y,
                   coalesce(least(padj0, padj1) < {ALPHA}, false) AS de
            FROM read_parquet(?)
            WHERE TRY_CAST(dose AS DOUBLE) = {DOSE} AND patient IS NOT NULL
              AND lfc0 IS NOT NULL AND lfc1 IS NOT NULL""",
        [frames],
    ).df()
    lines = pd.Index(sorted(de["patient"].unique()), name="line")
    drugs = pd.Index(sorted(de["drug"].unique()), name="drug")
    genes = pd.Index(sorted(de["gene_name"].unique()), name="gene")
    log(f"block: {len(lines)} lines x {len(drugs)} drugs x {len(genes)} genes, {len(de):,} gene-conditions")
    Y = _dense(de, "y", lines, drugs, genes, np.nan)
    D = _dense(de, "de", lines, drugs, genes, False)
    del de

    shards = sorted(str(p) for p in args.tahoe_dir.rglob("*.parquet") if DE_SUBSTRING in str(p))
    # The shards are laid out by line (30 consecutive shards held 2 lines, job 32435735), so
    # the subset is strided across the whole set rather than taken from the front.
    shards = shards[:: max(1, len(shards) // args.n_shards)][: args.n_shards]
    if not shards:
        raise FileNotFoundError(f"no DE shards under {args.tahoe_dir}")
    log(f"baseline expression: avg(baseMean) per (line, gene) over {len(shards)} shards")
    con.register("keep_lines", pd.DataFrame({"p": lines}))
    con.register("keep_genes", pd.DataFrame({"g": genes}))
    base = con.execute(
        """SELECT Cell_ID_DepMap AS patient, gene_name, avg(baseMean) AS base_mean, count(*) AS n
           FROM read_parquet(?)
           WHERE Cell_ID_DepMap IN (SELECT p FROM keep_lines) AND gene_name IN (SELECT g FROM keep_genes)
           GROUP BY 1, 2""",
        [shards],
    ).df()
    E = np.zeros((len(lines), len(genes)), dtype=np.float32)
    E[lines.get_indexer(base["patient"]), genes.get_indexer(base["gene_name"])] = base["base_mean"].to_numpy()
    n_obs = base.groupby("patient")["n"].median().reindex(lines).fillna(0)
    log(f"baseline: median observations per (line, gene) = {float(base['n'].median()):.0f}; "
        f"lines with no rows: {int((n_obs == 0).sum())}")

    np.savez(args.cache / "tensors.npz", Y=Y, D=D, E=E, lines=np.asarray(lines), drugs=np.asarray(drugs), genes=np.asarray(genes))

    import anndata as ad

    cpm = E / np.maximum(E.sum(1, keepdims=True), 1) * 1e6
    adata = ad.AnnData(X=np.rint(cpm).astype(np.float32))
    adata.obs_names = list(lines)
    adata.var_names = list(genes)
    adata.var["feature_name"] = list(genes)
    adata.write_h5ad(args.cache / "baseline.h5ad")
    log("wrote tensors.npz and baseline.h5ad")


# ----------------------------------------------------------------------------- embed


def embed(args: argparse.Namespace) -> None:
    import os

    from heldout_embed import load_stack_model  # rung 1 branch: strict load, head-stripped on retry

    h5ad = args.cache / "baseline.h5ad"
    for model_name, env_key in STACK_VERSIONS.items():
        out = args.cache / f"emb_{model_name}.npy"
        if out.exists():
            log(f"{model_name}: embedding present, skipping")
            continue
        ckpt = os.environ.get(env_key)
        if not ckpt:
            log(f"{model_name}: {env_key} unset, skipping")
            continue
        try:
            model = load_stack_model(Path(ckpt), args.cache)
            emb, _ = model.get_latent_representation(
                adata_path=str(h5ad), genelist_path=str(args.genelist), gene_name_col="feature_name",
                batch_size=8, show_progress=False, num_workers=0, random_state=0,
            )
            emb = np.asarray(emb, dtype=np.float32)
            np.save(out, emb)
            log(f"{model_name}: embedded {emb.shape[0]} lines x {emb.shape[1]}")
        except Exception as exc:  # a failed checkpoint costs its column, not the run
            log(f"{model_name}: FAILED ({type(exc).__name__}: {exc}); column will be absent")
        finally:
            import gc, torch  # noqa: E401

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- score


def masked_pearson(P: np.ndarray, O: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise Pearson r of P against O over the columns M marks; NaN below MIN_PANEL."""
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


def representations(E: np.ndarray, cache: Path) -> dict[str, np.ndarray]:
    """Line features: log1p CPM for knn, ten components for each linear model."""
    from sklearn.decomposition import NMF, PCA

    logcpm = np.log1p(E / np.maximum(E.sum(1, keepdims=True), 1) * 1e6)
    keep = logcpm.std(0) > 0
    Z = (logcpm[:, keep] - logcpm[:, keep].mean(0)) / logcpm[:, keep].std(0)
    reps: dict[str, np.ndarray] = {"knn": Z}
    reps["pca"] = PCA(N_COMPONENTS, random_state=0).fit_transform(Z)
    reps["nmf"] = NMF(N_COMPONENTS, init="nndsvda", max_iter=1000, random_state=0).fit_transform(logcpm[:, keep])
    for name in STACK_VERSIONS:
        path = cache / f"emb_{name}.npy"
        if path.exists():
            emb = np.load(path)
            reps[name] = PCA(min(N_COMPONENTS, emb.shape[1]), random_state=0).fit_transform(emb) if emb.shape[1] > N_COMPONENTS else emb
    return reps


def neighbours(Z: np.ndarray, i: int, train: np.ndarray) -> np.ndarray:
    c = np.corrcoef(Z[[i], :], Z[train])[0, 1:]
    return train[np.argsort(-c)[:K_NEIGHBOURS]]


def ols_predict(X: np.ndarray, i: int, train: np.ndarray, B_tr: np.ndarray) -> np.ndarray:
    """Departure at line i from OLS of the training departures on centred features."""
    mu = X[train].mean(0)
    Xc = X[train] - mu
    beta = np.linalg.pinv(Xc) @ np.nan_to_num(B_tr.reshape(len(train), -1))
    return ((X[i] - mu) @ beta).reshape(B_tr.shape[1:])


def score(args: argparse.Namespace, noise_drug: pd.DataFrame, noise_line: pd.DataFrame) -> None:
    t = np.load(args.cache / "tensors.npz", allow_pickle=True)
    Y, D, E = t["Y"], t["D"], t["E"]
    lines, drugs = pd.Index(t["lines"]), pd.Index(t["drugs"])
    n_l = len(lines)
    reps = representations(E, args.cache)
    models = [m for m in MODELS if m == "mean" or m in reps]
    log(f"scoring {len(models)} models: {models}")

    rows: list[pd.DataFrame] = []
    with np.errstate(invalid="ignore", divide="ignore"):
        Y_line_centred = Y - np.nanmean(Y, axis=1, keepdims=True)  # each line about its own drug mean
        D_line_sum = D.sum(1)  # (line, gene): drugs calling each gene DE in that line
        Y_line_sum = np.nansum(Y, axis=1)
        Y_line_cnt = np.isfinite(Y).sum(1)

    for i in range(n_l):
        train = np.setdiff1d(np.arange(n_l), [i])
        obs = Y[i]  # (drug, gene)
        nbrs = neighbours(reps["knn"], i, train)

        # --- scheme line: l* out, baseline = drug mean over training lines
        with np.errstate(invalid="ignore"):
            base_line = np.nanmean(Y[train], axis=0)
        B_tr = Y[train] - base_line
        panel = D[train].any(0)
        preds = {"mean": np.zeros_like(base_line)}
        preds["knn"] = np.nanmean(B_tr[np.searchsorted(train, nbrs)], axis=0)
        for m in models:
            if m not in ("mean", "knn"):
                preds[m] = ols_predict(reps[m], i, train, B_tr)
        for m in models:
            r, n = masked_pearson(base_line + preds[m], obs, panel)
            rows.append(pd.DataFrame({"scheme": "line", "line": lines[i], "drug": drugs, "model": m, "r": r, "n_panel": n}))

        # --- scheme drugcell: (l, d*) out, baseline = l's mean over its other drugs
        with np.errstate(invalid="ignore"):
            base_cell = (Y_line_sum[i][None, :] - np.nan_to_num(obs)) / (Y_line_cnt[i][None, :] - np.isfinite(obs))
        B_tr = Y_line_centred[train]
        panel = (D_line_sum[i][None, :] - D[i]) >= 1
        preds = {"mean": np.zeros_like(base_cell)}
        preds["knn"] = np.nanmean(B_tr[np.searchsorted(train, nbrs)], axis=0)
        for m in models:
            if m not in ("mean", "knn"):
                preds[m] = ols_predict(reps[m], i, train, B_tr)
        for m in models:
            r, n = masked_pearson(base_cell + preds[m], obs, panel)
            rows.append(pd.DataFrame({"scheme": "drugcell", "line": lines[i], "drug": drugs, "model": m, "r": r, "n_panel": n}))
        log(f"fold {i + 1}/{n_l} {lines[i]}")

    per = pd.concat(rows, ignore_index=True).dropna(subset=["r"])
    per.to_csv(args.out / "rung1_per_condition.csv", index=False)

    def table(scheme: str, by: str, noise: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        sub = per[per["scheme"] == scheme]
        other = "drug" if by == "line" else "line"
        wide = sub.pivot_table(index=by, columns="model", values="r", aggfunc="mean")[cols]
        wide.columns = [f"r_{c}" for c in wide.columns]
        meta = sub[sub["model"] == "mean"].groupby(by).agg(**{f"n_{other}s": (other, "nunique")}, n_panel_median=("n_panel", "median"))
        out = meta.join(noise[["noise_de"]]).join(wide)
        return out.sort_values("noise_de", ascending=False).round(4)

    table("line", "drug", noise_drug, models).to_csv(args.out / "rung1_line_per_drug.csv")
    table("drugcell", "line", noise_line, models).to_csv(args.out / "rung1_drugcell_per_line.csv")
    table("drugcell", "line", noise_line, ["mean"]).to_csv(args.out / "rung1_drugwhole_per_line.csv")

    summary = per.groupby(["scheme", "model"])["r"].agg(["mean", "median", "count"]).round(4)
    summary.to_csv(args.out / "rung1_summary.csv")
    log("summary (mean r per scheme x model):\n" + summary.to_string())


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("build", "embed", "score", "all"), default="all")
    ap.add_argument("--rung0-cache", type=Path, required=True, help="rung 0 cache with frame_*.parquet")
    ap.add_argument("--tahoe-dir", type=Path, required=True, help="Tahoe DE shards on scratch")
    ap.add_argument("--per-pair", type=Path, default=Path("results/rung0-assay-reliability/rung0_per_pair_r.csv"))
    ap.add_argument("--genelist", type=Path, default=Path("stack-large/basecount_1000per_15000max.pkl"))
    ap.add_argument("--cache", type=Path, required=True, help="where tensors, h5ad and embeddings go")
    ap.add_argument("--out", type=Path, required=True, help="where the tables go")
    ap.add_argument("--n-shards", type=int, default=100)
    ap.add_argument("--duckdb-memory", default="40GB")
    ap.add_argument("--duckdb-threads", type=int, default=8)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    noise_drug, noise_line = rung0_tables(args.per_pair, args.out)
    if args.stage in ("build", "all"):
        build(args)
    if args.stage in ("embed", "all"):
        embed(args)
    if args.stage in ("score", "all"):
        score(args, noise_drug, noise_line)
    (args.out / "run.json").write_text(json.dumps({"stage": args.stage, "dose": DOSE, "alpha": ALPHA, "k": K_NEIGHBOURS,
                                                   "n_components": N_COMPONENTS, "n_shards": args.n_shards,
                                                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
