"""Real Tahoe-100M cells for rung 1's block: DMSO cells of the 49 lines and treated cells at 5 uM
for the 108 drugs, read from the raw expression shards over the rung 1 gene table.

Why: rung 2 showed that cells synthesised from a pseudobulk carry a line's identity but sit
6x further from its real cells than two real halves sit from each other. Stack must be
tested on real cells. The August context files hold real cells for 33 drugs only.

Layout of the read. Tahoe's 3,388 expression shards are scanned in N blocks (array tasks);
block t takes the shards whose sorted index is congruent to t mod N. Per row group the small
columns (sample, cell_line_id) are read first; a row group is decoded (genes token ids +
expressions) only if it holds a wanted cell. Wanted = (DMSO_TF sample, line in the block) or
(sample at 5 uM of one of the 108 drugs, line in the block), capped per block at
DMSO_PER_LINE cells per line and TREATED_PER_PAIR cells per (line, drug). Each block writes
its cells as one CSR over the rung 1 gene table plus a metadata table; ``--combine`` merges
the blocks, caps again in shard order, and writes ``real_cells_5um.h5ad`` with obs columns
line, drug (DMSO_TF for controls), kind (control/treated), sample, plate, shard.

Dose comes from Tahoe's sample_metadata (sample -> drugname_drugconc), the drug names from
the same table, so they match the DE table's names. Cellosaurus -> DepMap comes from the DE
shards' own columns. Authenticated: huggingface_hub reads HF_TOKEN from the environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

TAHOE = "tahoebio/Tahoe-100M"
TAHOE_REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
DMSO = "DMSO_TF"
DOSE = 5.0
DMSO_PER_LINE = 512
TREATED_PER_PAIR = 128
META_COLS = ["sample", "cell_line_id", "plate"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snapshot_dir() -> Path:
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    return cache / "datasets--tahoebio--Tahoe-100M" / "snapshots" / TAHOE_REVISION


def dose_um(drugname_drugconc: str) -> float:
    try:
        return float(ast.literal_eval(drugname_drugconc)[0][1])
    except Exception:  # noqa: BLE001 -- an unparsable entry is simply not a wanted dose
        return float("nan")


def wanted_samples(drugs: list[str]) -> tuple[dict[str, str], set[str]]:
    """sample -> drug for the 108 drugs at DOSE, and the set of DMSO samples."""
    sm = pd.read_parquet(snapshot_dir() / "metadata" / "sample_metadata.parquet")
    drug_col = "drug" if "drug" in sm.columns else next(c for c in sm.columns if "drug" in c.lower() and "conc" not in c.lower())
    sm["dose_um"] = sm["drugname_drugconc"].astype(str).map(dose_um)
    treated = sm[(sm[drug_col].isin(drugs)) & (np.isclose(sm["dose_um"], DOSE))]
    dmso = set(sm.loc[sm[drug_col] == DMSO, "sample"].astype(str))
    found = sorted(set(treated[drug_col]))
    missing = sorted(set(drugs) - set(found))
    log(f"sample metadata: {len(sm)} samples; {len(treated)} wells for {len(found)} of {len(drugs)} drugs at {DOSE} uM; "
        f"{len(dmso)} DMSO wells; drugs with no {DOSE} uM well: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return dict(zip(treated["sample"].astype(str), treated[drug_col].astype(str))), dmso


def crosswalk(tahoe_de_dir: Path, lines: list[str]) -> dict[str, str]:
    import duckdb

    shards = sorted(str(p) for p in tahoe_de_dir.rglob("*.parquet") if "pseudobulk_differential_expression" in str(p))
    df = duckdb.connect().execute(
        "SELECT DISTINCT Cell_ID_Cellosaur AS cvcl, Cell_ID_DepMap AS depmap FROM read_parquet(?) WHERE Cell_ID_DepMap IS NOT NULL",
        [shards[::3]],
    ).df()
    cw = {c: d for c, d in zip(df["cvcl"], df["depmap"]) if d in set(lines)}
    if len(set(cw.values())) < len(lines):
        raise RuntimeError(f"crosswalk misses lines: {sorted(set(lines) - set(cw.values()))}")
    return cw


def gene_lookup(genes: pd.Index) -> np.ndarray:
    gm = pd.read_parquet(snapshot_dir() / "metadata" / "gene_metadata.parquet")
    col = genes.get_indexer(gm["gene_symbol"].astype(str))
    lookup = np.full(int(gm["token_id"].max()) + 1, -1, dtype=np.int64)
    lookup[gm["token_id"].to_numpy().astype(int)] = col
    log(f"gene map: {int((col >= 0).sum())} of {len(gm)} Tahoe tokens are in the rung 1 gene table")
    return lookup


def list_shards() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(TAHOE, repo_type="dataset", revision=TAHOE_REVISION)
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


# ----------------------------------------------------------------------------- one block


def run_block(args: argparse.Namespace) -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from scipy import sparse

    done = args.cache / f"block_{args.block}.done.json"
    if done.exists():
        log(f"block {args.block} done, skipping")
        return
    t = np.load(args.tensors, allow_pickle=True)
    lines, drugs, genes = [str(x) for x in t["lines"]], [str(x) for x in t["drugs"]], pd.Index(t["genes"])
    sample_drug, dmso_samples = wanted_samples(drugs)
    cw = crosswalk(args.tahoe_dir, lines)
    lookup = gene_lookup(genes)
    shards = list_shards()
    mine = [(i, s) for i, s in enumerate(shards) if i % args.n_blocks == args.block]
    log(f"block {args.block}/{args.n_blocks}: {len(mine)} of {len(shards)} shards")

    fs = HfFileSystem()
    quota: dict[tuple[str, str], int] = {}
    rows_meta: list[dict] = []
    mats: list = []
    n_rg_decoded = 0
    for k, (si, path) in enumerate(mine):
        t0 = time.time()
        with fs.open(f"datasets/{TAHOE}@{TAHOE_REVISION}/{path}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            plates_seen: set[str] = set()
            for rg in range(pf.num_row_groups):
                meta = pf.read_row_group(rg, columns=META_COLS)
                sample = np.asarray(meta.column("sample").to_pylist(), dtype=object).astype(str)
                cvcl = np.asarray(meta.column("cell_line_id").to_pylist(), dtype=object).astype(str)
                plate = np.asarray(meta.column("plate").to_pylist(), dtype=object).astype(str)
                plates_seen.update(plate[:1])
                keep_rows = []
                for r in range(len(sample)):
                    line = cw.get(cvcl[r])
                    if line is None:
                        continue
                    s = sample[r]
                    if s in dmso_samples:
                        key, cap = (line, DMSO), DMSO_PER_LINE
                    elif s in sample_drug:
                        key, cap = (line, sample_drug[s]), TREATED_PER_PAIR
                    else:
                        continue
                    if quota.get(key, 0) >= cap:
                        continue
                    quota[key] = quota.get(key, 0) + 1
                    keep_rows.append(r)
                    rows_meta.append({"line": line, "drug": key[1], "kind": "control" if key[1] == DMSO else "treated",
                                      "sample": s, "plate": plate[r], "shard": path})
                if not keep_rows:
                    continue
                n_rg_decoded += 1
                expr = pf.read_row_group(rg, columns=["genes", "expressions"])
                g_col = pc.take(expr.column("genes"), keep_rows).to_pylist()
                x_col = pc.take(expr.column("expressions"), keep_rows).to_pylist()
                data, indices, indptr = [], [], [0]
                for gl, xl in zip(g_col, x_col):
                    toks = np.asarray(gl, dtype=np.int64)
                    vals = np.asarray(xl, dtype=np.float32)
                    ok = (toks < len(lookup)) & (lookup[np.minimum(toks, len(lookup) - 1)] >= 0)
                    data.append(vals[ok])
                    indices.append(lookup[toks[ok]])
                    indptr.append(indptr[-1] + int(ok.sum()))
                mats.append(sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), np.array(indptr)), shape=(len(keep_rows), len(genes))))
        n_ctrl = sum(v for (_, d), v in quota.items() if d == DMSO)
        n_trt = sum(v for (_, d), v in quota.items() if d != DMSO)
        log(f"shard {k + 1}/{len(mine)} (#{si}, plates {sorted(plates_seen)[:3]}): kept so far {n_ctrl} control, {n_trt} treated "
            f"over {len([1 for (_, d) in quota if d != DMSO])} (line, drug) pairs; row groups decoded {n_rg_decoded}; {time.time() - t0:.0f}s")

    X = sparse.vstack(mats).tocsr() if mats else sparse.csr_matrix((0, len(genes)), dtype=np.float32)
    sparse.save_npz(args.cache / f"block_{args.block}.npz", X)
    pd.DataFrame(rows_meta).to_parquet(args.cache / f"block_{args.block}.parquet", index=False)
    done.write_text(json.dumps({"block": args.block, "n_blocks": args.n_blocks, "cells": int(X.shape[0]),
                                "shards": len(mine), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
    log(f"block {args.block}: wrote {X.shape[0]} cells")


# ----------------------------------------------------------------------------- combine


def combine(args: argparse.Namespace) -> None:
    import anndata as ad
    from scipy import sparse

    missing = [b for b in range(args.n_blocks) if not (args.cache / f"block_{b}.done.json").exists()]
    if missing:
        raise SystemExit(f"blocks not done: {missing}")
    t = np.load(args.tensors, allow_pickle=True)
    genes = pd.Index(t["genes"])
    metas, mats = [], []
    for b in range(args.n_blocks):
        m = pd.read_parquet(args.cache / f"block_{b}.parquet")
        if len(m):
            metas.append(m)
            mats.append(sparse.load_npz(args.cache / f"block_{b}.npz"))
    meta = pd.concat(metas, ignore_index=True)
    X = sparse.vstack(mats).tocsr()
    order = np.lexsort((meta.index.to_numpy(), meta["shard"].to_numpy()))  # shard order, then arrival
    meta, X = meta.iloc[order].reset_index(drop=True), X[order]
    cap = np.where(meta["kind"] == "control", DMSO_PER_LINE, TREATED_PER_PAIR)
    rank = meta.groupby(["line", "drug"]).cumcount().to_numpy()
    keep = rank < cap
    meta, X = meta[keep].reset_index(drop=True), X[keep]
    adata = ad.AnnData(X=X, obs=meta)
    adata.obs.index = [f"c{i}" for i in range(adata.n_obs)]
    adata.var_names = list(genes)
    adata.var["feature_name"] = list(genes)
    adata.write_h5ad(args.cache / "real_cells_5um.h5ad")
    counts = meta.groupby(["line", "drug"]).size().rename("n").reset_index()
    counts.to_csv(args.out / "real_cells_counts.csv", index=False)
    ctrl = counts[counts["drug"] == DMSO]
    trt = counts[counts["drug"] != DMSO]
    lib = np.asarray(X.sum(1)).ravel()
    log(f"combined: {adata.n_obs} cells; control per line min/median {int(ctrl['n'].min())}/{int(ctrl['n'].median())} over {len(ctrl)} lines; "
        f"treated pairs {len(trt)} (of 49 x 108), cells per pair min/median {int(trt['n'].min())}/{int(trt['n'].median())}, "
        f"pairs below 32 cells: {int((trt['n'] < 32).sum())}; library median {np.median(lib):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors", type=Path, required=True)
    ap.add_argument("--tahoe-dir", type=Path, required=True, help="DE shards on scratch, for the Cellosaurus -> DepMap crosswalk")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-blocks", type=int, default=16)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.combine:
        combine(args)
    elif args.block is not None:
        run_block(args)
    else:
        raise SystemExit("pass --block N or --combine")


if __name__ == "__main__":
    main()
