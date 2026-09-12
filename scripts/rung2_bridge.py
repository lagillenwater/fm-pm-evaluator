"""Rung 2: does a bulk profile, turned into cells, land where the same line's real cells land?

Rung 1's Stack columns rest on a bridge: a line's pseudobulk baseline is normalised to gene
probabilities and cells are drawn as multinomial(library, p). This script tests the bridge by
putting real Tahoe cells of the same lines through the same model and asking whether each
line's synthetic population sits with its own real population.

  cells   Real DMSO_TF (vehicle) cells of each rung 1 line, 256 per line, pooled from the
          per-drug context shards the 2026-08 generation work built from Tahoe's raw cells
          (context_by_drug/*.h5ad, already on the Alpine checkout; over Stack's gene panel).
          Written as real_dmso.h5ad. No download.
  embed   For each line: real half A (128 cells, even positions), real half B (128, odd), and
          128 synthetic cells drawn from the line's baseline profile with the library sizes of
          its real cells, so library size cannot separate them. Written line-major in blocks
          of 128 so each block is one Stack set of one population, and embedded with the base
          Stack encoder (set size 128).
  score   Mean embedding per block. For each line, cosine distance from its synthetic
          population to every line's real population (halves pooled):
            rank_synthetic   rank of the matched line among the 49 (1 = lands on its own line)
            ratio            d(synthetic, own real) / d(real half A, real half B) -- how far the
                             synthetic population sits from its real one, in units of the
                             distance between two real halves of the same line
            rank_real_half   rank of real half A's own line among all lines' real half B -- the
                             metric's positive control; 1 for every line if the model resolves
                             lines at all
          Chance rank is (n + 1) / 2.

Passing, in the spec's words: the synthesised population lands near the real cells by the
representation the rungs above consume, clearing a mismatched-line null -- here, rank 1 for
most lines with a ratio not far above 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

TAHOE_REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"  # the revision the context shards were built from
N_REAL = 256  # per line: two halves of one Stack set each
SET = 128  # base Stack's set size
SEED = 0
KINDS = ("real_a", "real_b", "synthetic")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- cells


def cells(args: argparse.Namespace) -> None:
    """Real DMSO cells from the context shards the 2026-08 generation work built from Tahoe's
    raw cells (``context_by_drug/*.h5ad``: treated + plate-matched DMSO_TF cells of the 50
    lines, capped per condition, over Stack's gene panel, DepMap id in ``cell_id``). Pooled
    across shards and de-duplicated by cell name until each rung 1 line has N_REAL cells."""
    import anndata as ad
    from scipy import sparse

    out = args.cache / "real_dmso.h5ad"
    if out.exists():
        log("real_dmso.h5ad present, skipping cells")
        return
    t = np.load(args.tensors, allow_pickle=True)
    lines = [str(x) for x in t["lines"]]
    shards = sorted(args.context_dir.glob("*.h5ad"))
    if not shards:
        raise FileNotFoundError(f"no context shards under {args.context_dir}")
    kept: dict[str, list] = {ln: [] for ln in lines}
    names: dict[str, set] = {ln: set() for ln in lines}
    var = None
    for si, path in enumerate(shards):
        a = ad.read_h5ad(path)
        if var is None:
            var = a.var.copy()
        ctrl = a.obs["is_control"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        cid = a.obs["cell_id"].astype(str).to_numpy()
        for ln in lines:
            if len(kept[ln]) >= N_REAL:
                continue
            idx = np.flatnonzero(ctrl & (cid == ln))
            idx = [i for i in idx if a.obs_names[i] not in names[ln]][: N_REAL - len(kept[ln])]
            if idx:
                kept[ln].append(sparse.csr_matrix(a.X[idx]))
                names[ln].update(a.obs_names[idx])
        counts = {ln: sum(b.shape[0] for b in kept[ln]) for ln in lines}
        log(f"shard {si + 1}/{len(shards)} {path.name}: lines full {sum(v >= N_REAL for v in counts.values())}/{len(lines)}, "
            f"min per line {min(counts.values())}")
        if all(v >= N_REAL for v in counts.values()):
            break
    counts = {ln: sum(b.shape[0] for b in kept[ln]) for ln in lines}
    short = sorted(ln for ln, v in counts.items() if v < N_REAL)
    if short:
        log(f"lines short of {N_REAL} DMSO cells across all shards (dropped from rung 2): {short}")
    order = [ln for ln in lines if counts[ln] >= N_REAL]
    X = sparse.vstack([sparse.vstack(kept[ln])[:N_REAL] for ln in order]).tocsr()
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"line": np.repeat(order, N_REAL)}), var=var)
    adata.obs.index = [f"{ln}_{i}" for ln in order for i in range(N_REAL)]
    if "feature_name" not in adata.var.columns:
        adata.var["feature_name"] = adata.var_names
    adata.write_h5ad(out)
    lib = np.asarray(X.sum(1)).ravel()
    log(f"wrote real_dmso.h5ad: {adata.n_obs} cells, {len(order)} lines, {adata.n_vars} panel genes; library size median "
        f"{np.median(lib):.0f} (IQR {np.percentile(lib, 25):.0f}-{np.percentile(lib, 75):.0f}); genes detected median {int(np.median((X > 0).sum(1)))}")


# ----------------------------------------------------------------------------- embed


def embed(args: argparse.Namespace) -> None:
    import anndata as ad
    from scipy import sparse

    from heldout_embed import load_stack_model

    out = args.cache / "bridge_embeddings.npz"
    if out.exists():
        log("bridge_embeddings.npz present, skipping embed")
        return
    real = ad.read_h5ad(args.cache / "real_dmso.h5ad")
    t = np.load(args.tensors, allow_pickle=True)
    E, lines, genes = t["E"], [str(x) for x in t["lines"]], pd.Index(t["genes"])
    # the genes both populations carry: Stack's panel (the real cells' genes) within the rung 1 table
    panel = real.var["feature_name"].astype(str).to_numpy()
    shared = np.flatnonzero(genes.get_indexer(panel) >= 0)
    e_cols = genes.get_indexer(panel[shared])
    log(f"{len(shared)} of {len(panel)} panel genes are in the rung 1 table; both populations restricted to them")
    order = [ln for ln in lines if ln in set(real.obs["line"])]
    rng = np.random.default_rng(SEED)
    blocks, obs = [], []
    for ln in order:
        R = real[real.obs["line"] == ln].X.tocsr()[:, shared]
        lib = np.asarray(R.sum(1)).ravel()  # each real cell's library over the shared genes
        prof = E[lines.index(ln)][e_cols]
        p = prof / max(prof.sum(), 1e-12)
        S = sparse.csr_matrix(np.stack([rng.multinomial(int(round(lib[j])), p) for j in range(SET)]).astype(np.float32))
        blocks += [R[0::2][:SET], R[1::2][:SET], S]
        obs += [(ln, k) for k in KINDS for _ in range(SET)]
    adata = ad.AnnData(X=sparse.vstack(blocks).tocsr(), obs=pd.DataFrame(obs, columns=["line", "kind"]))
    adata.obs.index = [f"{a}_{b}_{i}" for i, (a, b) in enumerate(obs)]
    adata.var_names = list(panel[shared])
    adata.var["feature_name"] = list(panel[shared])
    h5 = args.cache / "bridge_cells.h5ad"
    adata.write_h5ad(h5)
    log(f"wrote bridge_cells.h5ad: {adata.n_obs} cells = {len(order)} lines x 3 blocks x {SET}")

    model = load_stack_model(Path(os.environ["CKPT_BASE"]), args.cache)
    emb, _ = model.get_latent_representation(
        adata_path=str(h5), genelist_path=str(args.genelist), gene_name_col="feature_name",
        batch_size=8, show_progress=False, num_workers=0, random_state=SEED,
    )
    emb = np.asarray(emb, dtype=np.float32)
    assert emb.shape[0] == adata.n_obs
    np.savez(out, emb=emb, line=adata.obs["line"].to_numpy(), kind=adata.obs["kind"].to_numpy())
    log(f"embedded {emb.shape[0]} cells x {emb.shape[1]} with the base encoder")


# ----------------------------------------------------------------------------- score


def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return 1.0 - A @ B.T


def score(args: argparse.Namespace) -> None:
    z = np.load(args.cache / "bridge_embeddings.npz", allow_pickle=True)
    emb, line, kind = z["emb"], z["line"], z["kind"]
    order = list(dict.fromkeys(line))
    M = {k: np.stack([emb[(line == ln) & (kind == k)].mean(0) for ln in order]) for k in KINDS}
    real = (M["real_a"] + M["real_b"]) / 2
    n = len(order)
    d_sr = cosine_dist(M["synthetic"], real)  # synthetic of line i vs real of line j
    d_ab = cosine_dist(M["real_a"], M["real_b"])
    d_match = np.diag(d_sr)
    rank_synth = 1 + (d_sr < d_match[:, None]).sum(1)
    rank_real = 1 + (d_ab < np.diag(d_ab)[:, None]).sum(1)
    off = d_sr[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    per = pd.DataFrame({
        "line": order,
        "d_synthetic_to_own_real": d_match,
        "d_real_half_a_to_b": np.diag(d_ab),
        "ratio": d_match / np.diag(d_ab),
        "d_synthetic_to_other_real_median": np.median(off, axis=1),
        "rank_synthetic": rank_synth,
        "rank_real_half": rank_real,
    }).round(4)
    per.to_csv(args.out / "rung2_bridge_per_line.csv", index=False)
    summary = pd.Series({
        "n_lines": n,
        "chance_rank": (n + 1) / 2,
        "synthetic_rank1_fraction": float((rank_synth == 1).mean()),
        "synthetic_rank_median": float(np.median(rank_synth)),
        "synthetic_rank_top3_fraction": float((rank_synth <= 3).mean()),
        "real_half_rank1_fraction": float((rank_real == 1).mean()),
        "ratio_median": float(np.median(per["ratio"])),
        "d_match_median": float(np.median(d_match)),
        "d_other_median": float(np.median(off)),
        "d_real_halves_median": float(np.median(np.diag(d_ab))),
    }).round(4)
    summary.to_csv(args.out / "rung2_bridge_summary.csv", header=["value"])
    log("summary:\n" + summary.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("cells", "embed", "score", "all"), default="all")
    ap.add_argument("--tensors", type=Path, required=True, help="rung 1 tensors.npz (E, lines, genes)")
    ap.add_argument("--context-dir", type=Path, default=Path("context_by_drug"), help="per-drug context shards with real Tahoe cells")
    ap.add_argument("--genelist", type=Path, default=Path("stack-large/basecount_1000per_15000max.pkl"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("cells", "all"):
        cells(args)
    if args.stage in ("embed", "all"):
        embed(args)
    if args.stage in ("score", "all"):
        score(args)
    (args.out / "run.json").write_text(json.dumps({"stage": args.stage, "n_real": N_REAL, "set": SET, "seed": SEED,
                                                   "revision": TAHOE_REVISION, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
