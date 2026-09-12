"""Rung 0 noise per drug and per line, and rung 1 held-out-line prediction, as three tables.

Everything runs on Alpine from what is already on scratch; nothing is downloaded.

Rung 0 -- two group-bys of the promoted per-pair split-half table at 5 uM:
  rung0_noise_per_drug.csv   one row per drug, split-half r averaged over its lines
  rung0_noise_per_line.csv   one row per line, split-half r averaged over its drugs
``noise_de`` is the r over the DE genes and says where the reliable signal is.

Rung 1 -- the question: given a cell line never seen before, and only its baseline expression,
can a model predict how its expression changes under a drug better than assuming it responds
like the average line? Y[line, drug, gene] is the 5 uM log2 fold change averaged over plates
(the two half means of the rung 0 cache averaged), D the padj < 0.05 call on any plate.

  Holdout   line l* is held out of everything; the other 48 lines are the training lines.
  Panel     the held-out condition's OWN DE genes, D[l*, d], intersected with Stack's gene
            list so every model is scored on one panel. Selected by the target's response,
            never shown to a predictor, identical for every model. Rung 0 put the reliable
            signal there (split-half r ~0.6, against ~0.1 on any wide panel).
  Ceiling   ``ceiling_panel``: the split-half r of that condition's two plate halves on the
            same panel. The bound a predictor of the two-plate mean can attain is
            sqrt(2r/(1+r)), the square root of Y's Spearman-Brown reliability.
  Score     Pearson r between the predicted and the observed log fold change on the panel,
            per held-out condition, averaged per drug.

Every model predicts the held-out line's treated expression from its baseline and the other
lines' responses to the same drug, so every column is the same kind of object:

  mean            apply the drug's average fold change over the training lines
  knn             apply the average fold change of the 5 training lines nearest in baseline
                  expression
  pca, nmf        encode -> shift -> decode. Fit 10 components on the 49 baselines (unsupervised,
                  no response data). Encode the training lines' baselines (z) and their
                  treated profiles for drug d (t); fit the latent shift t = A z + b on the 48
                  training lines; for the held-out line predict t* = A z* + b and decode the
                  shift (t* - z*) back to genes. The decoded shift lives in the span of the
                  baseline components, so a drug effect that is not a baseline program is
                  invisible to the direct column; ``pca_adj`` / ``nmf_adj`` use the model only
                  for the departure: drug mean + decode(l*'s latent shift minus the training
                  lines' mean fitted shift). The mean model is the departure set to zero.
  stack_*         Stack in in-context generation mode (the two checkpoints with a generation
                  head: the cytokine-aligned model and the sci-Plex fine-tune). One 512-cell
                  set per (drug, held-out line), assembled the way the model was trained:
                  128 baseline cells of the training lines (the prompt, attending only to
                  itself), 256 treated cells of the training lines for this drug (the context,
                  visible to all), and 128 baseline cells of the held-out line (the query,
                  marked by the model's query embedding). The model returns each query cell's
                  expected treated counts; the predicted fold change is the query block's
                  mean over its own baseline mean. ``stack_*_adj`` is the same prediction used
                  only for the departure: drug mean + (Stack's prediction for l* minus Stack's
                  mean prediction over the training lines).

With ``--real-cells`` (real_cells_5um.h5ad from scripts/tahoe_real_cells.py) every cell in the
set is a real Tahoe cell: the prompt and query are real DMSO cells, the context real treated
cells at 5 uM. That is the rung 1 Stack column proper; the synthetic-cell run is kept as the
bridge variant rung 2 evaluates. Without it, cells are synthesised:

How a line's expression reaches Stack. Stack is a single-cell model trained on raw UMI counts
(~10^4 per cell, most genes at zero). Each line's baseline profile (mean DESeq2 baseMean over
its contrasts in every third raw shard) is normalised to gene probabilities and cells are drawn
as multinomial(10^4, p): count vectors with a cell's sparsity and magnitude whose expectation
is the line's profile. Treated cells are drawn the same way from baseline x 2^lfc. Untested
genes (no fold change) are left at baseline.

The drug axis (a line's mean over its other drugs predicting a new drug) was dropped on
2026-09-12: on this platform every drug in line l on plate p is contrasted against the same
DMSO pseudobulk, so that baseline shares control noise with its target and scored above the
attainable bound (1.15x). The earlier embedding -> 10 PCs -> OLS framing was dropped the same
day: it tests our regression on Stack's features, not Stack's own prediction.

Stages: build (duckdb -> tensors), generate (Stack, GPU; one checkpoint per array task), score
(the tables, CPU). ``--stage all`` runs them in order.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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
CEILING = "ceiling_panel"  # split-half r of the held-out condition on the scored panel
STACK_LIBRARY = 10_000  # counts per synthetic cell: the library size of a typical Tahoe cell
STACK_SEED = 0
EPS_COUNTS = 0.05  # pseudocount, in counts per 10^4, for the generated fold change
DROP_LINES = ("NA",)  # a line whose DepMap id is the literal string NA; no baseline can be joined to it
GEN_MODELS = {"stack_cytokine": "CKPT_CYTOKINE", "stack_sciplex": "CKPT_DRUG"}  # checkpoints with a generation head
MODELS = ("mean", "knn", "pca", "pca_adj", "nmf", "nmf_adj",
          "stack_cytokine", "stack_cytokine_adj", "stack_sciplex", "stack_sciplex_adj")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- rung 0


def rung0_tables(per_pair_csv: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(per_pair_csv, keep_default_na=False, na_values=[""])
    d = d[(d["dose"].astype(float) == DOSE) & ~d["patient"].isin(DROP_LINES)]
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

    if (args.cache / "tensors.npz").exists():
        log("tensors.npz present, skipping build")
        return
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.duckdb_memory}'; SET threads={args.duckdb_threads};")
    con.execute(f"SET temp_directory='{args.cache / 'duckdb_tmp'}';")

    frames = sorted(str(p) for p in args.rung0_cache.glob("frame_*.parquet"))
    if not frames:
        raise FileNotFoundError(f"no frame_*.parquet under {args.rung0_cache}")
    log(f"reading {len(frames)} rung 0 frame slices at {DOSE} uM")
    de = con.execute(
        f"""SELECT patient, drug, gene_name, lfc0, lfc1,
                   (lfc0 + lfc1) / 2 AS y,
                   coalesce(least(padj0, padj1) < {ALPHA}, false) AS de
            FROM read_parquet(?)
            WHERE TRY_CAST(dose AS DOUBLE) = {DOSE} AND patient IS NOT NULL
              AND patient NOT IN ({",".join("'" + x + "'" for x in DROP_LINES)})
              AND lfc0 IS NOT NULL AND lfc1 IS NOT NULL""",
        [frames],
    ).df()
    lines = pd.Index(sorted(de["patient"].unique()), name="line")
    drugs = pd.Index(sorted(de["drug"].unique()), name="drug")
    genes = pd.Index(sorted(de["gene_name"].unique()), name="gene")
    log(f"block: {len(lines)} lines x {len(drugs)} drugs x {len(genes)} genes, {len(de):,} gene-conditions")
    Y = _dense(de, "y", lines, drugs, genes, np.nan)
    D = _dense(de, "de", lines, drugs, genes, False)
    H0 = _dense(de, "lfc0", lines, drugs, genes, np.nan)  # the two plate halves, for the
    H1 = _dense(de, "lfc1", lines, drugs, genes, np.nan)  # split-half ceiling on each panel
    del de

    shards = sorted(str(p) for p in args.tahoe_dir.rglob("*.parquet") if DE_SUBSTRING in str(p))
    # The shards are laid out by line (30 consecutive shards held 2 lines, job 32435735; every
    # tenth missed ACH-000750, job 32435791), so take every third across the whole set.
    shards = shards[:: args.shard_stride]
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
    missing = list(lines[n_obs.to_numpy() == 0])
    log(f"baseline: median observations per (line, gene) = {float(base['n'].median()):.0f}; lines with no rows: {missing}")
    if missing:
        raise RuntimeError(f"no baseline expression for {missing}: lower --shard-stride")

    np.savez(args.cache / "tensors.npz", Y=Y, D=D, H0=H0, H1=H1, E=E, lines=np.asarray(lines), drugs=np.asarray(drugs), genes=np.asarray(genes))
    log("wrote tensors.npz")


# ----------------------------------------------------------------------------- synthetic cells


def profile_probabilities(E: np.ndarray, lfc: np.ndarray | None = None) -> np.ndarray:
    """Gene probabilities of each line's baseline, or of its treated state baseline x 2^lfc
    (untested genes, NaN lfc, stay at baseline)."""
    prof = E if lfc is None else E * np.exp2(np.nan_to_num(lfc, nan=0.0))
    return prof / np.maximum(prof.sum(-1, keepdims=True), 1e-12)


def draw_cells(p: np.ndarray, k: int, rng: np.random.Generator):
    """k multinomial cells of STACK_LIBRARY counts from each row of p; CSR of shape (rows*k, genes),
    row-major (row 0's k cells, then row 1's ...)."""
    from scipy import sparse

    blocks = [sparse.csr_matrix(rng.multinomial(STACK_LIBRARY, p[i], size=k).astype(np.float32)) for i in range(len(p))]
    return sparse.vstack(blocks).tocsr()


def load_stack_genes(genelist: Path | None) -> list[str] | None:
    if genelist is None or not genelist.exists():
        return None
    with genelist.open("rb") as fh:
        return [str(g) for g in pickle.load(fh)]


# ----------------------------------------------------------------------------- generate (Stack)


def assemble_sets(n_set: int, baseline_pool, treated_pool, line_of_base: np.ndarray, line_of_treated: np.ndarray,
                  held_out: np.ndarray, rng: np.random.Generator):
    """One in-context set per held-out line, in the model's training order:
    prompt = n_set/4 baseline cells of the training lines, context = n_set/2 treated cells of the
    training lines, query = n_set/4 baseline cells of the held-out line. Returns the stacked CSR
    (len(held_out) * n_set rows), the query rows of each set (indices into baseline_pool), and
    the three block sizes."""
    from scipy import sparse

    n_query = n_set // 4
    n_prompt = n_set // 4
    n_context = n_set - n_query - n_prompt
    blocks, query_rows = [], []
    for i in held_out:
        own = np.flatnonzero(line_of_base == i)[:n_query]
        prompt = rng.choice(np.flatnonzero(line_of_base != i), n_prompt, replace=False)
        context = rng.choice(np.flatnonzero(line_of_treated != i), n_context, replace=False)
        blocks.append(sparse.vstack([baseline_pool[prompt], treated_pool[context], baseline_pool[own]]))
        query_rows.append(own)
    return sparse.vstack(blocks).tocsr(), np.stack(query_rows), n_query, n_prompt, n_context


def generate(args: argparse.Namespace) -> None:
    import anndata as ad
    from stack.model_loading import load_model_from_checkpoint

    from scipy import sparse

    t = np.load(args.tensors or (args.cache / "tensors.npz"), allow_pickle=True)
    E, Y, lines, drugs, genes = t["E"], t["Y"], [str(x) for x in t["lines"]], [str(x) for x in t["drugs"]], pd.Index(t["genes"])
    n_l, n_d = len(lines), len(drugs)
    stack_genes = load_stack_genes(args.genelist)
    if stack_genes is None:
        raise FileNotFoundError(f"gene list {args.genelist} is required to generate")
    stack_in_ours = genes.get_indexer(stack_genes)  # -1 where Stack's gene is not in the table
    real = None
    if args.real_cells:
        import anndata as ad

        real = ad.read_h5ad(args.real_cells)
        assert list(real.var_names) == list(genes), "real cells must be over the rung 1 gene table"
        real_ctrl = {ln: real[(real.obs["line"] == ln) & (real.obs["kind"] == "control")].X.tocsr() for ln in lines}
        log(f"real cells: {real.n_obs}; control cells per line min {min(m.shape[0] for m in real_ctrl.values())}")

    def per_1e4(M) -> np.ndarray:
        """Mean profile of a cell block, in counts per 10^4 (so real and synthetic depths compare)."""
        M = sparse.csr_matrix(M)
        lib = np.asarray(M.sum(1)).ravel()
        return np.asarray(M.multiply(1e4 / np.maximum(lib, 1)[:, None]).mean(0)).ravel()
    todo = [m for m in GEN_MODELS if args.gen_model in (None, m)]
    for model_name in todo:
        out = args.cache / f"gen_{model_name}.npy"
        if out.exists():
            log(f"{model_name}: generation present, skipping")
            continue
        ckpt = os.environ.get(GEN_MODELS[model_name])
        if not ckpt:
            log(f"{model_name}: {GEN_MODELS[model_name]} unset, skipping")
            continue
        model = load_model_from_checkpoint(ckpt, model_class="ICL_FinetunedModel")
        n_set = int(model.n_cells)
        rng = np.random.default_rng(STACK_SEED)
        if real is None:
            base_pool = draw_cells(profile_probabilities(E), n_set // 4, rng)  # n_set/4 baseline cells per line
        else:
            base_pool = sparse.vstack([real_ctrl[ln][: n_set // 4] for ln in lines]).tocsr()  # the first n_set/4 real DMSO cells
        line_of_base = np.repeat(np.arange(n_l), n_set // 4)
        base_mean = np.stack([per_1e4(base_pool[line_of_base == i]) for i in range(n_l)])  # (line, gene), per 10^4
        k_treated = int(np.ceil((n_set // 2) / (n_l - 1))) + 1
        pred_lfc = np.full((n_l, n_d, len(stack_genes)), np.nan, dtype=np.float32)
        log(f"{model_name}: set size {n_set}; prompt {n_set // 4} / context {n_set - 2 * (n_set // 4)} / query {n_set // 4}; "
            f"{'real' if real is not None else 'synthetic'} cells; {k_treated} treated cells per training line per drug when drawn")
        for j in range(n_d):
            t0 = time.time()
            if real is None:
                treated_pool = draw_cells(profile_probabilities(E, Y[:, j, :]), k_treated, rng)
                line_of_treated = np.repeat(np.arange(n_l), k_treated)
                held_out = np.arange(n_l)
            else:
                blocks, owners = [], []
                for i, ln in enumerate(lines):
                    M = real[(real.obs["line"] == ln) & (real.obs["drug"] == drugs[j])].X
                    if M.shape[0]:
                        blocks.append(sparse.csr_matrix(M))
                        owners.append(np.full(M.shape[0], i))
                if not blocks:
                    log(f"{model_name}: drug {j + 1}/{n_d} {drugs[j]}: no real treated cells, skipped")
                    continue
                treated_pool = sparse.vstack(blocks).tocsr()
                line_of_treated = np.concatenate(owners)
                # a line can be held out only if the other lines still supply a full context block
                n_context_needed = n_set - 2 * (n_set // 4)
                held_out = np.array([i for i in range(n_l) if (line_of_treated != i).sum() >= n_context_needed])
                if len(held_out) == 0:
                    log(f"{model_name}: drug {j + 1}/{n_d} {drugs[j]}: {treated_pool.shape[0]} treated cells, too few for a context, skipped")
                    continue
            X, query_rows, n_query, n_prompt, n_context = assemble_sets(
                n_set, base_pool, treated_pool, line_of_base, line_of_treated, held_out, rng)
            adata = ad.AnnData(X=X)
            adata.var_names = list(genes)
            adata.var["feature_name"] = list(genes)
            mean_pred, _, _, _ = model.get_prediction(
                adata, str(args.genelist), gene_name_col="feature_name", mask_rate=0.0,
                cell_ratio=n_prompt / n_set, context_ratio=n_context / n_set,
                batch_size=args.gen_batch, num_workers=0, random_seed=STACK_SEED, show_progress=False,
            )
            mean_pred = np.asarray(mean_pred, dtype=np.float32).reshape(len(held_out), n_set, -1)
            q = mean_pred[:, n_set - n_query:, :]  # expected treated counts of each query cell, at its own library
            lib_q = np.asarray(base_pool[query_rows.ravel()].sum(1)).ravel().reshape(len(held_out), n_query)  # query libraries
            gen = (q * (1e4 / np.maximum(lib_q, 1))[:, :, None]).mean(1)  # (held-out line, stack gene), per 10^4
            base = np.where(stack_in_ours >= 0, base_mean[held_out][:, np.maximum(stack_in_ours, 0)], np.nan)
            pred_lfc[held_out, j, :] = np.log2((gen + EPS_COUNTS) / (base + EPS_COUNTS))
            log(f"{model_name}: drug {j + 1}/{n_d} {drugs[j]}: {len(held_out)} held-out lines, "
                f"{treated_pool.shape[0]} treated cells in the pool ({time.time() - t0:.0f}s)")
        np.save(out, pred_lfc)
        log(f"{model_name}: wrote {out.name} {pred_lfc.shape}")
        del model


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


def log_cpm(profiles: np.ndarray) -> np.ndarray:
    return np.log1p(profiles / np.maximum(profiles.sum(-1, keepdims=True), 1e-12) * 1e6)


class LatentShift:
    """Encode -> shift -> decode with a linear decomposition of the baselines (PCA or NMF).

    ``encode`` maps log-expression profiles to k coordinates; ``decode_shift`` maps a latent
    shift back to a per-gene change in log expression. Both are linear in the components."""

    def __init__(self, kind: str, L: np.ndarray, keep: np.ndarray):
        from sklearn.decomposition import NMF, PCA

        self.kind, self.keep = kind, keep
        X = L[:, keep]
        if kind == "pca":
            self.mu, self.sd = X.mean(0), X.std(0)
            self.model = PCA(N_COMPONENTS, random_state=0).fit((X - self.mu) / self.sd)
            self.components = self.model.components_ * self.sd  # back to log units
        else:
            self.scale = X.max(0)  # per-gene scaling, the non-negative analogue of standardising
            self.model = NMF(N_COMPONENTS, init="nndsvda", max_iter=1000, random_state=0).fit(X / self.scale)
            self.components = self.model.components_ * self.scale

    def encode(self, L: np.ndarray) -> np.ndarray:
        X = L.reshape(-1, L.shape[-1])[:, self.keep]
        if self.kind == "pca":
            z = self.model.transform((X - self.mu) / self.sd)
        else:
            z = self.model.transform(np.clip(X / self.scale, 0, None))
        return z.reshape(*L.shape[:-1], N_COMPONENTS)

    def decode_shift(self, dz: np.ndarray, n_genes: int) -> np.ndarray:
        out = np.full((*dz.shape[:-1], n_genes), np.nan, dtype=np.float32)
        out[..., self.keep] = dz @ self.components
        return out


def score(args: argparse.Namespace, noise_drug: pd.DataFrame) -> None:
    t = np.load(args.tensors or (args.cache / "tensors.npz"), allow_pickle=True)
    Y, D, E, H0, H1 = t["Y"], t["D"], t["E"], t["H0"], t["H1"]
    lines, drugs, genes = pd.Index(t["lines"]), pd.Index(t["drugs"]), pd.Index(t["genes"])
    n_l, n_d, n_g = Y.shape

    stack_genes = load_stack_genes(args.genelist)
    in_stack = np.ones(n_g, dtype=bool) if stack_genes is None else np.isin(genes, stack_genes)
    log(f"panel restricted to {int(in_stack.sum())} of {n_g} genes present in Stack's list")

    # Stack's generated fold changes, mapped from its gene order into the table's
    gen: dict[str, np.ndarray] = {}
    for m in GEN_MODELS:
        path = args.cache / f"gen_{m}.npy"
        if path.exists() and stack_genes is not None:
            arr = np.load(path)
            full = np.full((n_l, n_d, n_g), np.nan, dtype=np.float32)
            idx = genes.get_indexer(stack_genes)
            full[:, :, idx[idx >= 0]] = arr[:, :, idx >= 0]
            gen[m] = full
    models = [m for m in MODELS if not m.startswith("stack") or m.replace("_adj", "") in gen]
    log(f"scoring {len(models)} models: {models}")

    # baselines and treated profiles in log CPM; latent coordinates of both, once
    L = log_cpm(E)
    keep = L.std(0) > 0
    Z = (L[:, keep] - L[:, keep].mean(0)) / L[:, keep].std(0)  # for knn
    L_T = log_cpm(E[:, None, :] * np.exp2(np.nan_to_num(Y, nan=0.0)))  # (line, drug, gene)
    latent = {k: LatentShift(k, L, keep) for k in ("pca", "nmf")}
    z = {k: v.encode(L) for k, v in latent.items()}  # (line, k)
    t_lat = {k: v.encode(L_T) for k, v in latent.items()}  # (line, drug, k)
    log("latent coordinates encoded")

    rows: list[pd.DataFrame] = []
    for i in range(n_l):
        if not np.isfinite(Y[i]).any():
            continue
        train = np.setdiff1d(np.arange(n_l), [i])
        obs = Y[i]
        panel = D[i] & in_stack[None, :]
        with np.errstate(invalid="ignore"):
            mean_d = np.nanmean(Y[train], axis=0)
        preds = {"mean": mean_d}
        c = np.corrcoef(Z[[i], :], Z[train])[0, 1:]
        preds["knn"] = np.nanmean(Y[train[np.argsort(-c)[:K_NEIGHBOURS]]], axis=0)
        for k in ("pca", "nmf"):
            X = np.column_stack([np.ones(len(train)), z[k][train]])  # (48, 1+k), the same for every drug
            beta = np.linalg.pinv(X) @ t_lat[k][train].reshape(len(train), -1)  # -> (1+k, drug*k)
            t_hat = (np.concatenate([[1.0], z[k][i]]) @ beta).reshape(n_d, N_COMPONENTS)
            shift = t_hat - z[k][i]  # (drug, k): the held-out line's predicted latent shift
            fitted_shift = (X @ beta).reshape(len(train), n_d, N_COMPONENTS) - z[k][train][:, None, :]
            preds[k] = latent[k].decode_shift(shift, n_g)
            preds[k + "_adj"] = mean_d + latent[k].decode_shift(shift - fitted_shift.mean(0), n_g)
        for m, arr in gen.items():
            preds[m] = arr[i]
            with np.errstate(invalid="ignore"):
                preds[m + "_adj"] = mean_d + (arr[i] - np.nanmean(arr[train], axis=0))
        for m in models:
            r, n = masked_pearson(preds[m], obs, panel)
            rows.append(pd.DataFrame({"line": lines[i], "drug": drugs, "model": m, "r": r, "n_panel": n}))
        r, n = masked_pearson(H0[i], H1[i], panel)
        rows.append(pd.DataFrame({"line": lines[i], "drug": drugs, "model": CEILING, "r": r, "n_panel": n}))
        log(f"fold {i + 1}/{n_l} {lines[i]}")

    per = pd.concat(rows, ignore_index=True).dropna(subset=["r"])
    per.to_csv(args.out / "rung1_per_condition.csv", index=False)

    wide = per.pivot_table(index="drug", columns="model", values="r", aggfunc="mean")
    table = per[per["model"] == "mean"].groupby("drug").agg(n_lines=("line", "nunique"), n_panel_median=("n_panel", "median"))
    table = table.join(noise_drug[["noise_de"]]).join(wide[[CEILING]])
    table = table.join(wide[models].rename(columns=lambda m: f"r_{m}"))
    table.sort_values("noise_de", ascending=False).round(4).to_csv(args.out / "rung1_line_per_drug.csv")

    summary = per.groupby("model")["r"].agg(["mean", "median", "count"])
    split_half = float(summary.loc[CEILING, "mean"])
    attainable = np.sqrt(2 * split_half / (1 + split_half))  # sqrt of Y's Spearman-Brown reliability
    summary["attainable_bound"] = attainable
    summary["frac_attainable"] = summary["mean"] / attainable
    summary = summary.reindex([CEILING] + models).round(4)
    summary.to_csv(args.out / "rung1_summary.csv")
    log("summary (mean r over held-out conditions):\n" + summary.to_string())


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("build", "generate", "score", "all"), default="all")
    ap.add_argument("--rung0-cache", type=Path, required=True, help="rung 0 cache with frame_*.parquet")
    ap.add_argument("--tahoe-dir", type=Path, required=True, help="Tahoe DE shards on scratch")
    ap.add_argument("--per-pair", type=Path, default=Path("results/rung0-assay-reliability/rung0_per_pair_r.csv"))
    ap.add_argument("--genelist", type=Path, default=Path("stack-large/basecount_1000per_15000max.pkl"))
    ap.add_argument("--cache", type=Path, required=True, help="where tensors and generations go")
    ap.add_argument("--out", type=Path, required=True, help="where the tables go")
    ap.add_argument("--gen-model", choices=tuple(GEN_MODELS), default=None, help="generate for one checkpoint only")
    ap.add_argument("--real-cells", type=Path, default=None, help="real_cells_5um.h5ad: real cells for prompt, context and query")
    ap.add_argument("--tensors", type=Path, default=None, help="tensors.npz to reuse (default: <cache>/tensors.npz)")
    ap.add_argument("--gen-batch", type=int, default=2, help="in-context sets per forward pass")
    ap.add_argument("--shard-stride", type=int, default=3)
    ap.add_argument("--duckdb-memory", default="40GB")
    ap.add_argument("--duckdb-threads", type=int, default=8)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    noise_drug, _ = rung0_tables(args.per_pair, args.out)
    if args.stage in ("build", "all") and not args.tensors:
        build(args)
    if args.stage in ("generate", "all"):
        generate(args)
    if args.stage in ("score", "all"):
        score(args, noise_drug)
    (args.out / "run.json").write_text(json.dumps({
        "stage": args.stage, "dose": DOSE, "alpha": ALPHA, "k": K_NEIGHBOURS, "n_components": N_COMPONENTS,
        "shard_stride": args.shard_stride, "stack_library": STACK_LIBRARY, "eps_counts": EPS_COUNTS,
        "real_cells": str(args.real_cells) if args.real_cells else None,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
