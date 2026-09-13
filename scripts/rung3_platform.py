"""Rung 3: what does crossing the measurement platform cost?

Rung 1's question with the training data on the other platform. Rung 1 showed that the best
predictor of a new line's response to a drug is the drug's average response over other lines,
at ~0.9 of the attainable correlation on the Tahoe screen. Rung 3 fits every rung 1 predictor
on LINCS L1000 instead (bulk, 978 measured landmark genes, other labs, other cells) and applies
it to a new Tahoe line, beside the same predictor fit on Tahoe, on one panel and one ceiling.
The ratio, per method, is the platform cost.

LINCS side (Level 3, per-instance log2 expression; LINCS 2020, public). For each Tahoe drug
found by name: instances at 24 h and the dose nearest 10 uM in every LINCS line, and the DMSO
instances on the same plates. Per (LINCS line, drug): log2FC = mean log2 treated - mean log2
plate-matched DMSO over the 978 landmarks; per LINCS line: baseline = mean log2 DMSO. A LINCS
line that matches a Tahoe line by name is left out of everything used to predict that Tahoe
line, so the prediction is for a line never seen on either platform.

Tahoe side: the rung 1 block (Y at 5 uM, own-DE panels, split-half halves), the two lines with
empty DE tables dropped. Panel: the held-out condition's own DE genes within the landmarks.

Columns (cross-platform, fit on LINCS -> applied to the Tahoe line):
  l1000_mean       the drug's mean log2FC over LINCS lines
  l1000_knn        mean log2FC of the 5 LINCS lines nearest the Tahoe line by baseline
                   expression on the landmarks (each platform standardised gene-wise within
                   itself first, so the platform offset does not choose the neighbours)
  l1000_pca_adj,   components on LINCS baselines, latent shift t = A z + b fit from LINCS
  l1000_nmf_adj    baseline -> LINCS treated; the Tahoe baseline mapped into LINCS units
                   gene-wise, encoded, its departure decoded and added to l1000_mean
  l1000_shuffled   another drug's l1000_mean (fixed derangement): the null
  tahoe_shuffled   another drug's tahoe_mean (same derangement): how much of the within-platform
                   reference is a response every drug provokes on this screen rather than
                   the drug's own effect
  stack_l1000ctx   Stack in-context generation on the pure LINCS set: prompt = 128 L1000
  (+ _adj)         control pseudo-cells, context = 256 L1000 treated pseudo-cells of the drug,
                   query = 128 real Tahoe DMSO cells of the held-out line. Pseudo-cells:
                   2^log2 expression over the genes both tables share, multinomial draws at
                   real Tahoe library sizes. Nothing Tahoe enters except the line predicted.
Within-platform references (rung 1's predictors, rescored on this panel): tahoe_mean,
tahoe_knn, tahoe_pca_adj, tahoe_nmf_adj, stack_tahoectx (+ _adj; rung 1's real-cell run).
retained = r(cross-platform) / r(within-platform), per method, per drug and overall.

Stages: level3 (download if absent, select, read, summarise), stack (one checkpoint per array
task), score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rung1_simple_tables import LatentShift, N_COMPONENTS, load_stack_genes, log_cpm, masked_pearson  # noqa: E402

BASE = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020"
FILES = {
    "trt": "level3/level3_beta_trt_cp_n1805898x12328.gctx",
    "ctl": "level3/level3_beta_ctl_n188708x12328.gctx",
    "instinfo": "instinfo_beta.txt",
    "geneinfo": "geneinfo_beta.txt",
}
TIME_H = 24.0
DOSE_UM = 10.0
DROP_LINES = ("NA", "ACH-000628", "ACH-000311")
K_NEIGHBOURS = 5
CEILING = "ceiling_panel"
SEED = 0
EPS_COUNTS = 0.05
GEN_MODELS = {"stack_cytokine": "CKPT_CYTOKINE", "stack_sciplex": "CKPT_DRUG"}
CROSS = ("l1000_mean", "l1000_knn", "l1000_pca_adj", "l1000_nmf_adj", "l1000_shuffled")
WITHIN = ("tahoe_mean", "tahoe_knn", "tahoe_pca_adj", "tahoe_nmf_adj")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def base_name(drug: str) -> str:
    """'Neratinib (maleate)' -> 'neratinib': the salt in parentheses is dropped before matching."""
    return norm(str(drug).split(" (")[0])


def first_col(df: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in {list(df.columns)[:20]}")


# ----------------------------------------------------------------------------- level 3


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


def read_rows(gctx: Path, inst_ids: np.ndarray, gene_ids: np.ndarray | None):
    """Rows of a GCTX (instances x genes) for the given instance ids, over the given gene ids
    (all genes when None). Returns (matrix, kept instance ids, gene ids in matrix order)."""
    import h5py

    with h5py.File(gctx, "r") as h:
        col_ids = h["0/META/COL/id"][:].astype(str)
        row_ids = h["0/META/ROW/id"][:].astype(str)
        pos = pd.Index(col_ids).get_indexer(inst_ids)
        keep = pos >= 0
        pos = pos[keep]
        gpos = np.arange(len(row_ids)) if gene_ids is None else pd.Index(row_ids).get_indexer(gene_ids)
        assert (gpos >= 0).all()
        mat = h["0/DATA/0/matrix"]
        sigs_first = mat.shape[0] == len(col_ids)
        order = np.argsort(pos)
        sorted_pos = pos[order]
        Z = np.empty((len(pos), len(gpos)), dtype=np.float32)
        chunk = 500
        for s in range(0, len(sorted_pos), chunk):
            idx = sorted_pos[s:s + chunk]
            block = mat[idx, :] if sigs_first else mat[:, idx].T
            Z[order[s:s + chunk]] = block[:, gpos]
            if (s // chunk) % 20 == 0:
                log(f"  {gctx.name}: {min(s + chunk, len(sorted_pos))}/{len(sorted_pos)} instances read")
    return Z, inst_ids[keep], row_ids[gpos]


def level3(args: argparse.Namespace) -> None:
    out = args.cache / "rung3_l1000.npz"
    if out.exists():
        log("rung3_l1000.npz present, skipping level3")
        return
    paths = fetch(args.l1000_dir)
    t = np.load(args.tensors, allow_pickle=True)
    drugs = [str(x) for x in t["drugs"]]
    gi = pd.read_csv(paths["geneinfo"], sep="\t", low_memory=False)
    lm = gi[gi["feature_space"].astype(str).str.lower() == "landmark"]
    id_to_symbol = dict(zip(gi["gene_id"].astype(str), gi["gene_symbol"].astype(str)))

    ii = pd.read_csv(paths["instinfo"], sep="\t", low_memory=False)
    name_c, cell_c, type_c = first_col(ii, "cmap_name", "pert_iname"), first_col(ii, "cell_iname", "cell_id"), "pert_type"
    plate_c = first_col(ii, "rna_plate", "det_plate", "plate")
    inst_c = first_col(ii, "inst_id", "sample_id")
    ii["time_h"] = pd.to_numeric(ii["pert_time"], errors="coerce")
    ii["dose_um"] = pd.to_numeric(ii["pert_dose"], errors="coerce")
    key_of = {base_name(d): d for d in drugs}
    trt = ii[(ii[type_c].astype(str) == "trt_cp") & (ii["time_h"] == TIME_H)].copy()
    trt["drug"] = trt[name_c].astype(str).map(norm).map(key_of)
    trt = trt[trt["drug"].notna()].copy()
    trt["dose_gap"] = (trt["dose_um"] - DOSE_UM).abs()
    trt = trt[trt["dose_gap"] == trt.groupby([cell_c, "drug"])["dose_gap"].transform("min")]
    ctl = ii[(ii[type_c].astype(str) == "ctl_vehicle") & (ii["time_h"] == TIME_H) & ii[plate_c].isin(set(trt[plate_c]))].copy()
    log(f"{trt['drug'].nunique()} of {len(drugs)} drugs found; {len(trt)} treated instances over {trt[cell_c].nunique()} lines; "
        f"{len(ctl)} vehicle instances on their {ctl[plate_c].nunique()} plates")

    T, t_ids, genes12k = read_rows(paths["trt"], trt[inst_c].astype(str).to_numpy(), None)
    C, c_ids, _ = read_rows(paths["ctl"], ctl[inst_c].astype(str).to_numpy(), None)
    trt = trt.set_index(inst_c).loc[t_ids].reset_index()
    ctl = ctl.set_index(inst_c).loc[c_ids].reset_index()
    symbols = np.array([id_to_symbol.get(g, g) for g in genes12k])
    lm_mask = np.isin(genes12k, lm["gene_id"].astype(str).to_numpy())

    # plate-matched log2 fold change per treated instance, then per (line, drug); baseline per line
    ctl_plate_mean = pd.DataFrame(C).groupby(ctl[plate_c].to_numpy()).mean()
    has_vehicle = trt[plate_c].isin(ctl_plate_mean.index).to_numpy()
    if (~has_vehicle).any():
        log(f"{int((~has_vehicle).sum())} treated instances on {trt.loc[~has_vehicle, plate_c].nunique()} plates with no vehicle "
            f"rows in the control matrix are dropped")
        trt, T = trt[has_vehicle].reset_index(drop=True), T[has_vehicle]
    lfc_inst = T - ctl_plate_mean.loc[trt[plate_c].to_numpy()].to_numpy()
    cells = sorted(trt[cell_c].astype(str).unique())
    l_drugs = sorted(trt["drug"].unique())
    ci, di = pd.Index(cells), pd.Index(l_drugs)
    R = np.full((len(cells), len(l_drugs), int(lm_mask.sum())), np.nan, dtype=np.float32)
    for (c, d), g in trt.groupby([cell_c, "drug"]).indices.items():
        R[ci.get_loc(c), di.get_loc(d)] = lfc_inst[g][:, lm_mask].mean(0)
    B = np.full((len(cells), int(lm_mask.sum())), np.nan, dtype=np.float32)
    for c, g in ctl.groupby(cell_c).indices.items():
        if c in ci:
            B[ci.get_loc(c)] = C[g][:, lm_mask].mean(0)
    census = trt.groupby([cell_c, "drug"]).agg(n_inst=(inst_c, "size"), dose_um=("dose_um", "first")).reset_index().rename(columns={cell_c: "cell"})
    census.to_csv(args.out / "rung3_l1000_census.csv", index=False)
    np.savez(out, R=R, B=B, cells=np.array(cells), drugs=np.array(l_drugs), genes=symbols[lm_mask])
    np.savez(args.cache / "rung3_l1000_instances.npz", T=T, C=C, genes=symbols,
             trt_cell=trt[cell_c].astype(str).to_numpy(), trt_drug=trt["drug"].to_numpy(), trt_plate=trt[plate_c].astype(str).to_numpy(),
             ctl_cell=ctl[cell_c].astype(str).to_numpy(), ctl_plate=ctl[plate_c].astype(str).to_numpy())
    log(f"wrote rung3_l1000.npz: {len(cells)} lines x {len(l_drugs)} drugs x {int(lm_mask.sum())} landmarks; "
        f"lines per drug median {int(census.groupby('drug').size().median())}; baseline for {int(np.isfinite(B).all(1).sum())} lines")


# ----------------------------------------------------------------------------- Stack, pure LINCS set


def matched_cells(lines: list[str], cells: list[str], crosswalk: Path) -> dict[str, int]:
    cw = pd.read_csv(crosswalk)
    name = dict(zip(cw["line"].astype(str), cw["cell_name"].astype(str)))
    key = {norm(c): i for i, c in enumerate(cells)}
    return {ln: key[norm(name[ln])] for ln in lines if ln in name and norm(name[ln]) in key}


def pseudo_cells(log2_profiles: np.ndarray, libs: np.ndarray, rng: np.random.Generator):
    """One multinomial cell per (profile, library): 2^log2 expression -> probabilities -> counts."""
    from scipy import sparse

    P = np.exp2(log2_profiles.astype(np.float64))
    P /= P.sum(1, keepdims=True)
    return sparse.csr_matrix(np.stack([rng.multinomial(int(round(n)), p) for p, n in zip(P, libs)]).astype(np.float32))


def stack(args: argparse.Namespace) -> None:
    import anndata as ad
    from scipy import sparse
    from stack.model_loading import load_model_from_checkpoint

    inst = np.load(args.cache / "rung3_l1000_instances.npz", allow_pickle=True)
    T, C, genes12k = inst["T"], inst["C"], [str(g) for g in inst["genes"]]
    trt_cell, trt_drug, trt_plate = inst["trt_cell"].astype(str), inst["trt_drug"].astype(str), inst["trt_plate"].astype(str)
    ctl_cell, ctl_plate = inst["ctl_cell"].astype(str), inst["ctl_plate"].astype(str)
    t = np.load(args.tensors, allow_pickle=True)
    lines, drugs, genes = [str(x) for x in t["lines"]], [str(x) for x in t["drugs"]], pd.Index(t["genes"])
    real = ad.read_h5ad(args.real_cells)
    assert list(real.var_names) == list(genes)
    shared = np.flatnonzero(genes.get_indexer(genes12k) >= 0)  # Tahoe-table positions of the LINCS genes
    l_pos = genes.get_indexer(genes12k)
    l_keep = (l_pos >= 0) & ~pd.Series(genes12k).duplicated().to_numpy()  # a few LINCS ids share a symbol; keep the first
    shared_names = [genes12k[k] for k in np.flatnonzero(l_keep)]
    stack_genes = load_stack_genes(args.genelist)
    log(f"{len(shared_names)} genes shared by the LINCS matrix and the Tahoe table")
    cells = sorted(set(trt_cell))
    matched = matched_cells(lines, cells, args.crosswalk)
    real_ctrl = {ln: real[(real.obs["line"] == ln) & (real.obs["kind"] == "control")].X.tocsr()[:, l_pos[l_keep]] for ln in lines}
    libs_pool = np.concatenate([np.asarray(m.sum(1)).ravel() for m in real_ctrl.values()])
    rng = np.random.default_rng(SEED)

    for model_name in [m for m in GEN_MODELS if args.gen_model in (None, m)]:
        out = args.cache / f"gen_l1000ctx_{model_name}.npy"
        if out.exists():
            log(f"{model_name}: present, skipping")
            continue
        model = load_model_from_checkpoint(os.environ[GEN_MODELS[model_name]], model_class="ICL_FinetunedModel")
        n_set = int(model.n_cells)
        n_query, n_prompt = n_set // 4, n_set // 4
        n_context = n_set - n_query - n_prompt
        pred = np.full((len(lines), len(drugs), len(stack_genes)), np.nan, dtype=np.float32)
        stack_in_ours = genes.get_indexer(stack_genes)
        for j, d in enumerate(drugs):
            t0 = time.time()
            ti = np.flatnonzero(trt_drug == d)
            if len(ti) == 0:
                continue
            plates = set(trt_plate[ti])
            ci_ = np.flatnonzero(np.isin(ctl_plate, list(plates)))
            k_t = int(np.ceil(n_context / max(1, len(ti)))) + 1
            k_c = int(np.ceil(n_prompt / max(1, len(ci_)))) + 1
            trt_pool = pseudo_cells(np.repeat(T[ti][:, l_keep], k_t, axis=0), rng.choice(libs_pool, len(ti) * k_t), rng)
            ctl_pool = pseudo_cells(np.repeat(C[ci_][:, l_keep], k_c, axis=0), rng.choice(libs_pool, len(ci_) * k_c), rng)
            trt_owner = np.repeat(trt_cell[ti], k_t)
            ctl_owner = np.repeat(ctl_cell[ci_], k_c)
            blocks, held = [], []
            for i, ln in enumerate(lines):
                if ln in DROP_LINES:
                    continue
                ex = cells[matched[ln]] if ln in matched else None
                t_ok = np.flatnonzero(trt_owner != ex) if ex else np.arange(len(trt_owner))
                c_ok = np.flatnonzero(ctl_owner != ex) if ex else np.arange(len(ctl_owner))
                if len(t_ok) < n_context or len(c_ok) < n_prompt:
                    continue
                q = real_ctrl[ln][:n_query]
                blocks.append(sparse.vstack([ctl_pool[rng.choice(c_ok, n_prompt, replace=False)],
                                             trt_pool[rng.choice(t_ok, n_context, replace=False)], q]))
                held.append(i)
            if not blocks:
                continue
            adata = ad.AnnData(X=sparse.vstack(blocks).tocsr())
            adata.var_names = shared_names
            adata.var["feature_name"] = shared_names
            mean_pred, _, _, _ = model.get_prediction(
                adata, str(args.genelist), gene_name_col="feature_name", mask_rate=0.0,
                cell_ratio=n_prompt / n_set, context_ratio=n_context / n_set, batch_size=2, num_workers=0,
                random_seed=SEED, show_progress=False)
            mp = np.asarray(mean_pred, dtype=np.float32).reshape(len(held), n_set, -1)[:, n_set - n_query:, :]
            for h, i in enumerate(held):
                q = real_ctrl[lines[i]][:n_query]
                lib = np.asarray(q.sum(1)).ravel()
                gen = (mp[h] * (1e4 / np.maximum(lib, 1))[:, None]).mean(0)
                base_full = np.asarray(q.multiply(1e4 / np.maximum(lib, 1)[:, None]).mean(0)).ravel()  # over shared genes
                base = np.full(len(stack_genes), np.nan, dtype=np.float32)
                pos_in_shared = pd.Index(shared_names).get_indexer(stack_genes)
                base[pos_in_shared >= 0] = base_full[pos_in_shared[pos_in_shared >= 0]]
                pred[i, j] = np.log2((gen + EPS_COUNTS) / (base + EPS_COUNTS))
            log(f"{model_name}: drug {j + 1}/{len(drugs)} {d}: {len(held)} held-out lines, {len(ti)} LINCS treated instances, "
                f"{len(ci_)} vehicle instances ({time.time() - t0:.0f}s)")
        np.save(out, pred)
        log(f"{model_name}: wrote {out.name} {pred.shape}")
        del model


# ----------------------------------------------------------------------------- score


def score(args: argparse.Namespace) -> None:
    t = np.load(args.tensors, allow_pickle=True)
    Y, D, E, H0, H1 = t["Y"], t["D"], t["E"], t["H0"], t["H1"]
    lines, drugs, genes = [str(x) for x in t["lines"]], [str(x) for x in t["drugs"]], pd.Index(t["genes"])
    n_l, n_d, n_g = Y.shape
    keep_l = np.array([ln not in DROP_LINES for ln in lines])
    l1 = np.load(args.cache / "rung3_l1000.npz", allow_pickle=True)
    R, Bl, cells, l_drugs, lm = l1["R"], l1["B"], [str(x) for x in l1["cells"]], [str(x) for x in l1["drugs"]], [str(x) for x in l1["genes"]]
    gi = genes.get_indexer(lm)
    lm_ok = gi >= 0
    lm_tahoe = gi[lm_ok]  # Tahoe positions of the landmark genes, in LINCS order
    in_lm = np.zeros(n_g, dtype=bool)
    in_lm[lm_tahoe] = True
    stack_genes = load_stack_genes(args.genelist)
    log(f"{int(lm_ok.sum())} of {len(lm)} landmark genes in the Tahoe table")
    matched = matched_cells(lines, cells, args.crosswalk)
    log(f"{len(matched)} Tahoe lines also in LINCS, excluded from their own LINCS-side predictions")
    d_idx = [drugs.index(d) for d in l_drugs if d in drugs]
    l_pos = {d: j for j, d in enumerate(l_drugs)}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(d_idx))
    while (perm == np.arange(len(d_idx))).any():
        perm = rng.permutation(len(d_idx))

    # --- LINCS-side representations on the landmarks (lines with a baseline)
    have_b = np.isfinite(Bl).all(1)
    Lb = Bl[have_b][:, lm_ok]  # LINCS baselines, log2, landmark genes in Tahoe order
    L_T = Lb[:, None, :] + np.nan_to_num(R[have_b][:, :, lm_ok], nan=0.0)  # LINCS treated profiles
    keep_g = Lb.std(0) > 0
    latent = {k: LatentShift(k, Lb, keep_g) for k in ("pca", "nmf")}
    z_l = {k: v.encode(Lb) for k, v in latent.items()}
    t_l = {k: v.encode(L_T) for k, v in latent.items()}
    Zl = (Lb[:, keep_g] - Lb[:, keep_g].mean(0)) / Lb[:, keep_g].std(0)
    # the Tahoe baselines on the same genes, standardised within Tahoe, then mapped into LINCS units gene-wise
    Ltahoe = log_cpm(E)[:, lm_tahoe]
    mu_t, sd_t = Ltahoe[keep_l].mean(0), np.where(Ltahoe[keep_l].std(0) > 0, Ltahoe[keep_l].std(0), 1.0)
    Zt = (Ltahoe - mu_t) / sd_t
    L_tahoe_in_lincs = Zt * Lb.std(0) + Lb.mean(0)
    z_t = {k: v.encode(L_tahoe_in_lincs) for k, v in latent.items()}
    cell_rows = np.flatnonzero(have_b)  # LINCS cell index of each row of Lb

    # --- Tahoe-side (within-platform) representations, as in rung 1
    Lt_all = log_cpm(E)
    keep_t = Lt_all.std(0) > 0
    Zt_all = (Lt_all[:, keep_t] - Lt_all[:, keep_t].mean(0)) / Lt_all[:, keep_t].std(0)
    L_T_t = log_cpm(E[:, None, :] * np.exp2(np.nan_to_num(Y, nan=0.0)))
    latent_t = {k: LatentShift(k, Lt_all, keep_t) for k in ("pca", "nmf")}
    zt_all = {k: v.encode(Lt_all) for k, v in latent_t.items()}
    tt_all = {k: v.encode(L_T_t) for k, v in latent_t.items()}
    gen_t = {}
    for m in GEN_MODELS:
        p = args.rung1_cache / f"gen_{m}.npy"
        if p.exists() and stack_genes is not None:
            arr = np.load(p)
            full = np.full((n_l, n_d, n_g), np.nan, dtype=np.float32)
            idx = genes.get_indexer(stack_genes)
            full[:, :, idx[idx >= 0]] = arr[:, :, idx >= 0]
            gen_t[m] = full
    gen_x = {}
    for m in GEN_MODELS:
        p = args.cache / f"gen_l1000ctx_{m}.npy"
        if p.exists() and stack_genes is not None:
            arr = np.load(p)
            full = np.full((n_l, n_d, n_g), np.nan, dtype=np.float32)
            idx = genes.get_indexer(stack_genes)
            full[:, :, idx[idx >= 0]] = arr[:, :, idx >= 0]
            gen_x[m] = full
    log(f"within-platform Stack arrays: {sorted(gen_t)}; cross-platform Stack arrays: {sorted(gen_x)}")

    def to_full(v_lm: np.ndarray) -> np.ndarray:
        full = np.full(n_g, np.nan, dtype=np.float32)
        full[lm_tahoe] = v_lm
        return full

    rows = []
    for i, ln in enumerate(lines):
        if not keep_l[i] or not np.isfinite(Y[i]).any():
            continue
        train = np.flatnonzero(keep_l & (np.arange(n_l) != i))
        ex = matched.get(ln)
        rows_ok = np.flatnonzero(cell_rows != ex) if ex is not None else np.arange(len(cell_rows))
        # --- LINCS side, once per held-out line
        c = np.corrcoef(Zt[i][keep_g][None, :], Zl[rows_ok])[0, 1:]
        nbr = rows_ok[np.argsort(-c)[:K_NEIGHBOURS]]
        lin_dep = {}
        for k in ("pca", "nmf"):
            X = np.column_stack([np.ones(len(rows_ok)), z_l[k][rows_ok]])
            beta = np.linalg.pinv(X) @ t_l[k][rows_ok].reshape(len(rows_ok), -1)
            t_hat = (np.concatenate([[1.0], z_t[k][i]]) @ beta).reshape(len(l_drugs), N_COMPONENTS)
            fitted = (X @ beta).reshape(len(rows_ok), len(l_drugs), N_COMPONENTS) - z_l[k][rows_ok][:, None, :]
            lin_dep[k] = latent[k].decode_shift((t_hat - z_t[k][i]) - fitted.mean(0), Lb.shape[1])  # (l_drug, landmark)
        # --- Tahoe side, once per held-out line (rung 1's predictors)
        with np.errstate(invalid="ignore"):
            tahoe_mean = np.nanmean(Y[train], axis=0)
        ct = np.corrcoef(Zt_all[[i], :], Zt_all[train])[0, 1:]
        tahoe_knn = np.nanmean(Y[train[np.argsort(-ct)[:K_NEIGHBOURS]]], axis=0)
        tahoe_dep = {}
        for k in ("pca", "nmf"):
            X = np.column_stack([np.ones(len(train)), zt_all[k][train]])
            beta = np.linalg.pinv(X) @ tt_all[k][train].reshape(len(train), -1)
            t_hat = (np.concatenate([[1.0], zt_all[k][i]]) @ beta).reshape(n_d, N_COMPONENTS)
            fitted = (X @ beta).reshape(len(train), n_d, N_COMPONENTS) - zt_all[k][train][:, None, :]
            tahoe_dep[k] = latent_t[k].decode_shift((t_hat - zt_all[k][i]) - fitted.mean(0), n_g)

        for k_d, j_t in enumerate(d_idx):
            d = drugs[j_t]
            jl = l_pos[d]
            panel = D[i, j_t] & in_lm
            with np.errstate(invalid="ignore"):
                l_mean = np.nanmean(R[cell_rows[rows_ok], jl][:, lm_ok], axis=0)
                l_knn = np.nanmean(R[cell_rows[nbr], jl][:, lm_ok], axis=0)
                l_shuf = np.nanmean(R[cell_rows[rows_ok], l_pos[drugs[d_idx[perm[k_d]]]]][:, lm_ok], axis=0)
            preds = {
                "tahoe_mean": tahoe_mean[j_t], "tahoe_shuffled": tahoe_mean[d_idx[perm[k_d]]], "tahoe_knn": tahoe_knn[j_t],
                "tahoe_pca_adj": tahoe_mean[j_t] + tahoe_dep["pca"][j_t], "tahoe_nmf_adj": tahoe_mean[j_t] + tahoe_dep["nmf"][j_t],
                "l1000_mean": to_full(l_mean), "l1000_knn": to_full(l_knn),
                "l1000_pca_adj": to_full(l_mean + lin_dep["pca"][jl]), "l1000_nmf_adj": to_full(l_mean + lin_dep["nmf"][jl]),
                "l1000_shuffled": to_full(l_shuf),
            }
            for m, arr in gen_t.items():
                preds[f"{m}_tahoectx"] = arr[i, j_t]
                with np.errstate(invalid="ignore"):
                    preds[f"{m}_tahoectx_adj"] = tahoe_mean[j_t] + (arr[i, j_t] - np.nanmean(arr[train, j_t], axis=0))
            for m, arr in gen_x.items():
                preds[f"{m}_l1000ctx"] = arr[i, j_t]
                with np.errstate(invalid="ignore"):
                    preds[f"{m}_l1000ctx_adj"] = to_full(l_mean) + (arr[i, j_t] - np.nanmean(arr[train, j_t], axis=0))
            for m, p in preds.items():
                r, n = masked_pearson(p[None, :], Y[i, j_t][None, :], panel[None, :])
                rows.append({"line": ln, "drug": d, "model": m, "r": float(r[0]), "n_panel": int(n[0])})
            r, n = masked_pearson(H0[i, j_t][None, :], H1[i, j_t][None, :], panel[None, :])
            rows.append({"line": ln, "drug": d, "model": CEILING, "r": float(r[0]), "n_panel": int(n[0])})
        log(f"fold {i + 1}/{n_l} {ln}")
    per = pd.DataFrame(rows).dropna(subset=["r"])
    per.to_csv(args.out / "rung3_per_condition.csv", index=False)

    models = [m for m in per["model"].unique() if m != CEILING]
    wide = per.pivot_table(index="drug", columns="model", values="r", aggfunc="mean")
    table = per[per["model"] == "tahoe_mean"].groupby("drug").agg(n_lines=("line", "nunique"), n_panel_median=("n_panel", "median"))
    census_p = args.out / "rung3_l1000_census.csv"
    if census_p.exists():
        cen = pd.read_csv(census_p)
        table = table.join(cen.groupby("drug").agg(n_l1000_lines=("cell", "nunique"), l1000_dose_um=("dose_um", "median")))
    table = table.join(wide[[CEILING]]).join(wide[models].rename(columns=lambda m: f"r_{m}"))
    pairs = [("mean", "l1000_mean", "tahoe_mean"), ("knn", "l1000_knn", "tahoe_knn"), ("pca_adj", "l1000_pca_adj", "tahoe_pca_adj"),
             ("nmf_adj", "l1000_nmf_adj", "tahoe_nmf_adj")]
    pairs += [(m, f"{m}_l1000ctx", f"{m}_tahoectx") for m in GEN_MODELS if f"{m}_l1000ctx" in models and f"{m}_tahoectx" in models]
    pairs += [(f"{m}_adj", f"{m}_l1000ctx_adj", f"{m}_tahoectx_adj") for m in GEN_MODELS if f"{m}_l1000ctx_adj" in models and f"{m}_tahoectx_adj" in models]
    for name, x, w in pairs:
        table[f"retained_{name}"] = table[f"r_{x}"] / table[f"r_{w}"]
    table.sort_values("r_tahoe_mean", ascending=False).round(4).to_csv(args.out / "rung3_per_drug.csv")

    summary = per.groupby("model")["r"].agg(["mean", "median", "count"])
    split_half = float(summary.loc[CEILING, "mean"])
    attainable = np.sqrt(2 * split_half / (1 + split_half))
    summary["attainable_bound"] = attainable
    summary["frac_attainable"] = summary["mean"] / attainable
    order = [CEILING] + [m for m in ("tahoe_mean", "tahoe_shuffled", "l1000_mean", "l1000_shuffled", "tahoe_knn", "l1000_knn", "tahoe_pca_adj", "l1000_pca_adj",
                                     "tahoe_nmf_adj", "l1000_nmf_adj") if m in summary.index]
    order += [m for m in models if m not in order]
    summary = summary.reindex(order)
    for name, x, w in pairs:
        summary.loc[f"retained_{name}", "mean"] = summary.loc[x, "mean"] / summary.loc[w, "mean"]
    wide_c = per.pivot_table(index=["line", "drug"], columns="model", values="r")
    summary.loc["l1000_mean_beats_shuffled_fraction", "mean"] = float((wide_c["l1000_mean"] > wide_c["l1000_shuffled"]).mean())
    summary.loc["tahoe_mean_beats_shuffled_fraction", "mean"] = float((wide_c["tahoe_mean"] > wide_c["tahoe_shuffled"]).mean())
    # drug-specific transfer: what crosses above the wrong-drug floor on each platform
    summary.loc["retained_mean_above_shuffled", "mean"] = ((summary.loc["l1000_mean", "mean"] - summary.loc["l1000_shuffled", "mean"])
                                                           / (summary.loc["tahoe_mean", "mean"] - summary.loc["tahoe_shuffled", "mean"]))
    summary.round(4).to_csv(args.out / "rung3_summary.csv")
    log("summary:\n" + summary.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("level3", "stack", "score", "all"), default="all")
    ap.add_argument("--tensors", type=Path, required=True)
    ap.add_argument("--crosswalk", type=Path, default=Path("docs/tasks/rung1-held-out-prediction/rung1_line_crosswalk.csv"))
    ap.add_argument("--l1000-dir", type=Path, required=True, help="where the LINCS 2020 files are (downloaded if absent)")
    ap.add_argument("--real-cells", type=Path, default=None, help="real_cells_5um.h5ad (query cells for the Stack stage)")
    ap.add_argument("--rung1-cache", type=Path, default=None, help="rung 1 cache holding the real-cell gen_*.npy (within-platform Stack)")
    ap.add_argument("--genelist", type=Path, default=Path("stack-large/basecount_1000per_15000max.pkl"))
    ap.add_argument("--gen-model", choices=tuple(GEN_MODELS), default=None)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rung1_cache is None:
        args.rung1_cache = args.cache
    if args.stage in ("level3", "all"):
        level3(args)
    if args.stage in ("stack", "all"):
        stack(args)
    if args.stage in ("score", "all"):
        score(args)
    (args.out / "run.json").write_text(json.dumps({"stage": args.stage, "time_h": TIME_H, "dose_um": DOSE_UM, "seed": SEED,
                                                   "drop_lines": DROP_LINES, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
