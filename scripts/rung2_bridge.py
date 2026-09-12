"""Rung 2: do cells synthesised from a pseudobulk land where the same material's real cells land?

Rung 1's synthetic-cell Stack run rests on a bridge: a profile is normalised to gene
probabilities and cells are drawn as multinomial(library, p). This script tests the bridge on
real Tahoe cells (real_cells_5um.h5ad from scripts/tahoe_real_cells.py), over the largest N
the data allows: every line's baseline, and every (line, drug) treated state with enough
real cells.

Populations, each a block whose size is a multiple of Stack's 128-cell set, so every
attention set holds one population of one line (and one drug):

  baseline   per line: real half A (even positions), real half B (odd) and a synthetic
             population of the same size, drawn from the line's pseudobulk baseline with the
             library sizes of its real cells so depth cannot separate them. Halves are the
             largest multiple of 128 the line's DMSO cells allow: 256 at 512 cells, 128 from
             256 cells.
  treated    per (line, drug) with >= 128 real treated cells at 5 uM: real treated (128) and
             synthetic treated (128 drawn from baseline x 2^lfc with matched libraries)

Embedded with the base Stack encoder; one mean embedding per block. Cosine distances.

  rung2_baseline_per_line.csv
    rank_synthetic    rank of the line's own real population among all lines' real
                      populations, from its synthetic population (1 = lands on its own line)
    ratio             d(synthetic, own real) / d(real half A, real half B): how far the
                      synthetic population sits from its real one, in units of the distance
                      between two real halves of the same line
    rank_real_half    real half A's own line among all lines' real half B -- the metric's
                      positive control
  rung2_treated_per_condition.csv
    rank_line         from the synthetic treated (l, d): rank of real treated (l, d) among real
                      treated (l', d) over the lines with that drug -- does it land on its line
    rank_drug         rank of real treated (l, d) among real treated (l, d') over that line's
                      drugs -- does it land on its drug
    closer_to_treated whether the synthetic treated population is nearer real treated (l, d)
                      than the line's real baseline: does the synthesised shift point the
                      right way
    ratio             d(synthetic treated, real treated) / d(real half A, real half B) of the
                      same line
Chance rank is (n + 1) / 2 in each case; the summaries carry the n.
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

SET = 128  # base Stack's set size; every block below is a multiple of it
N_BASE = 512  # real DMSO cells per line the download keeps at most (some lines have fewer)
N_TREATED = 128
SEED = 0
DMSO = "DMSO_TF"
CHUNK_CELLS = 120_000  # cells per embedding call


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def draw(p: np.ndarray, libs: np.ndarray, rng: np.random.Generator):
    from scipy import sparse

    return sparse.csr_matrix(np.stack([rng.multinomial(int(round(n)), p) for n in libs]).astype(np.float32))


# ----------------------------------------------------------------------------- embed


def embed(args: argparse.Namespace) -> None:
    import anndata as ad
    from scipy import sparse

    from heldout_embed import load_stack_model

    out = args.cache / "bridge_block_embeddings.npz"
    if out.exists():
        log("bridge_block_embeddings.npz present, skipping embed")
        return
    real = ad.read_h5ad(args.real_cells)
    t = np.load(args.tensors, allow_pickle=True)
    E, Y = t["E"], t["Y"]
    lines, drugs, genes = [str(x) for x in t["lines"]], [str(x) for x in t["drugs"]], pd.Index(t["genes"])
    assert list(real.var_names) == list(genes), "real cells must be over the rung 1 gene table"
    with args.genelist.open("rb") as fh:
        panel = [str(g) for g in pickle.load(fh)]
    shared = np.flatnonzero(genes.get_indexer(panel) >= 0)
    shared_names = list(genes[np.sort(genes.get_indexer(panel)[genes.get_indexer(panel) >= 0])])
    shared = np.array(genes.get_indexer(shared_names))
    log(f"{len(shared)} of {len(panel)} Stack panel genes are in the table; all blocks restricted to them")
    rng = np.random.default_rng(SEED)
    obs = real.obs
    blocks: list[tuple[str, str, str, object]] = []  # (line, drug, kind, X over shared genes)

    for i, ln in enumerate(lines):
        C = real[(obs["line"] == ln) & (obs["kind"] == "control")].X.tocsr()[:, shared]
        half = min(N_BASE // 2, (C.shape[0] // 2) // SET * SET)  # the largest multiple-of-SET half the line supports
        if half < SET:
            log(f"{ln}: {C.shape[0]} real DMSO cells < {2 * SET}, baseline skipped")
            continue
        C = C[: 2 * half]
        lib = np.asarray(C.sum(1)).ravel()
        prof = E[i][shared]
        p = prof / max(prof.sum(), 1e-12)
        blocks += [(ln, DMSO, "real_a", C[0::2]), (ln, DMSO, "real_b", C[1::2]), (ln, DMSO, "synthetic", draw(p, lib[:half], rng))]
    n_base_lines = len({b[0] for b in blocks})

    n_pairs = 0
    for i, ln in enumerate(lines):
        for j, d in enumerate(drugs):
            T = real[(obs["line"] == ln) & (obs["drug"] == d)].X
            if T.shape[0] < N_TREATED or not np.isfinite(Y[i, j]).any():
                continue
            T = sparse.csr_matrix(T)[:N_TREATED][:, shared]
            lib = np.asarray(T.sum(1)).ravel()
            prof = E[i][shared] * np.exp2(np.nan_to_num(Y[i, j][shared], nan=0.0))
            p = prof / max(prof.sum(), 1e-12)
            blocks += [(ln, d, "real_treated", T), (ln, d, "synthetic_treated", draw(p, lib, rng))]
            n_pairs += 1
    log(f"blocks: {n_base_lines} baselines x 3, {n_pairs} treated (line, drug) pairs x 2; "
        f"{sum(b[3].shape[0] for b in blocks)} cells")

    model = load_stack_model(Path(os.environ["CKPT_BASE"]), args.cache)
    means = []
    start = 0
    while start < len(blocks):
        chunk, n = [], 0
        while start < len(blocks) and n + blocks[start][3].shape[0] <= CHUNK_CELLS:
            chunk.append(blocks[start]); n += blocks[start][3].shape[0]; start += 1
        adata = ad.AnnData(X=sparse.vstack([b[3] for b in chunk]).tocsr())
        adata.var_names = shared_names
        adata.var["feature_name"] = shared_names
        emb, _ = model.get_latent_representation(
            adata_path=adata, genelist_path=str(args.genelist), gene_name_col="feature_name",
            batch_size=8, show_progress=False, num_workers=0, random_state=SEED,
        )
        emb = np.asarray(emb, dtype=np.float32)
        assert emb.shape[0] == adata.n_obs
        pos = 0
        for ln, d, kind, X in chunk:
            means.append((ln, d, kind, emb[pos:pos + X.shape[0]].mean(0)))
            pos += X.shape[0]
        log(f"embedded {start}/{len(blocks)} blocks")
    meta = pd.DataFrame([(a, b, c) for a, b, c, _ in means], columns=["line", "drug", "kind"])
    np.savez(out, emb=np.stack([m[3] for m in means]), line=meta["line"].to_numpy(), drug=meta["drug"].to_numpy(), kind=meta["kind"].to_numpy())
    log(f"wrote {out.name}: {len(means)} block means")


# ----------------------------------------------------------------------------- score


def unit(A: np.ndarray) -> np.ndarray:
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def score(args: argparse.Namespace) -> None:
    z = np.load(args.cache / "bridge_block_embeddings.npz", allow_pickle=True)
    emb, line, drug, kind = unit(z["emb"]), z["line"], z["drug"], z["kind"]
    key = {(l, d, k): i for i, (l, d, k) in enumerate(zip(line, drug, kind))}
    base_lines = sorted({l for (l, d, k) in key if d == DMSO and k == "synthetic"})

    # --- baseline
    RA = np.stack([emb[key[(l, DMSO, "real_a")]] for l in base_lines])
    RB = np.stack([emb[key[(l, DMSO, "real_b")]] for l in base_lines])
    S = np.stack([emb[key[(l, DMSO, "synthetic")]] for l in base_lines])
    R = unit(RA + RB)
    d_sr = 1 - S @ R.T
    d_ab = 1 - np.einsum("ij,ij->i", RA, RB)
    d_match = np.diag(d_sr)
    n = len(base_lines)
    rank_s = 1 + (d_sr < d_match[:, None]).sum(1)
    d_ab_all = 1 - RA @ RB.T
    rank_r = 1 + (d_ab_all < np.diag(d_ab_all)[:, None]).sum(1)
    off = d_sr[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    per = pd.DataFrame({"line": base_lines, "d_synthetic_to_own_real": d_match, "d_real_half_a_to_b": d_ab,
                        "ratio": d_match / d_ab, "d_synthetic_to_other_real_median": np.median(off, 1),
                        "rank_synthetic": rank_s, "rank_real_half": rank_r}).round(4)
    per.to_csv(args.out / "rung2_baseline_per_line.csv", index=False)
    summary = {
        "baseline_n_lines": n, "baseline_chance_rank": (n + 1) / 2,
        "baseline_rank1_fraction": float((rank_s == 1).mean()), "baseline_rank_median": float(np.median(rank_s)),
        "baseline_top3_fraction": float((rank_s <= 3).mean()), "baseline_real_half_rank1_fraction": float((rank_r == 1).mean()),
        "baseline_ratio_median": float(np.median(per["ratio"])),
        "baseline_d_match_median": float(np.median(d_match)), "baseline_d_other_median": float(np.median(off)),
        "baseline_d_real_halves_median": float(np.median(d_ab)),
    }

    # --- treated
    pairs = sorted({(l, d) for (l, d, k) in key if k == "synthetic_treated"})
    if pairs:
        half_dist = dict(zip(base_lines, d_ab))
        real_ctrl = {l: R[i] for i, l in enumerate(base_lines)}
        by_drug: dict[str, list] = {}
        by_line: dict[str, list] = {}
        for l, d in pairs:
            by_drug.setdefault(d, []).append(l)
            by_line.setdefault(l, []).append(d)
        rows = []
        for l, d in pairs:
            s = emb[key[(l, d, "synthetic_treated")]]
            r = emb[key[(l, d, "real_treated")]]
            d_match_t = float(1 - s @ r)
            others_l = [float(1 - s @ emb[key[(l2, d, "real_treated")]]) for l2 in by_drug[d] if l2 != l]
            others_d = [float(1 - s @ emb[key[(l, d2, "real_treated")]]) for d2 in by_line[l] if d2 != d]
            d_ctrl = float(1 - s @ real_ctrl[l]) if l in real_ctrl else np.nan
            rows.append({"line": l, "drug": d, "d_synthetic_to_real_treated": d_match_t,
                         "rank_line": 1 + sum(x < d_match_t for x in others_l), "n_lines_with_drug": len(others_l) + 1,
                         "rank_drug": 1 + sum(x < d_match_t for x in others_d), "n_drugs_for_line": len(others_d) + 1,
                         "d_synthetic_to_real_control": d_ctrl,
                         "closer_to_treated": float(d_match_t < d_ctrl) if np.isfinite(d_ctrl) else np.nan,
                         "ratio": d_match_t / half_dist[l] if l in half_dist else np.nan})
        tr = pd.DataFrame(rows).round(4)
        tr.to_csv(args.out / "rung2_treated_per_condition.csv", index=False)
        summary.update({
            "treated_n_conditions": len(tr), "treated_n_lines": tr["line"].nunique(), "treated_n_drugs": tr["drug"].nunique(),
            "treated_rank_line_1_fraction": float((tr["rank_line"] == 1).mean()), "treated_rank_line_median": float(tr["rank_line"].median()),
            "treated_chance_rank_line": float(((tr["n_lines_with_drug"] + 1) / 2).mean()),
            "treated_rank_drug_1_fraction": float((tr["rank_drug"] == 1).mean()), "treated_rank_drug_median": float(tr["rank_drug"].median()),
            "treated_chance_rank_drug": float(((tr["n_drugs_for_line"] + 1) / 2).mean()),
            "treated_closer_to_treated_fraction": float(tr["closer_to_treated"].mean()),
            "treated_ratio_median": float(tr["ratio"].median()),
        })
    s = pd.Series(summary).round(4)
    s.to_csv(args.out / "rung2_summary.csv", header=["value"])
    log("summary:\n" + s.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("embed", "score", "all"), default="all")
    ap.add_argument("--real-cells", type=Path, required=True, help="real_cells_5um.h5ad")
    ap.add_argument("--tensors", type=Path, required=True, help="rung 1 tensors.npz (E, Y, lines, drugs, genes)")
    ap.add_argument("--genelist", type=Path, default=Path("stack-large/basecount_1000per_15000max.pkl"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("embed", "all"):
        embed(args)
    if args.stage in ("score", "all"):
        score(args)
    (args.out / "run.json").write_text(json.dumps({"stage": args.stage, "n_base": N_BASE, "n_treated": N_TREATED, "set": SET,
                                                   "seed": SEED, "real_cells": str(args.real_cells),
                                                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
