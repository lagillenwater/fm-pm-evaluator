"""Task 4 of rung 1: DMSO cells over Tahoe-100M's 50 grid lines -- selection, halves, pseudobulk.

Design.md section 4: every line description is built from the same cells -- up to 1,000
untreated (``DMSO_TF``) cells per grid line, spread evenly over its plates, picked with a
fixed seed. This script scans Tahoe-100M's 3,388 cell-matrix shards (Hugging Face revision
``2dc57900b7981cfcf5e211527169a0b006546a95``) in blocks, keeping only DMSO cells of the 50
grid lines' Cellosaurus ids whose key (``fmharness.heldout.cells.cell_keys``) falls in a
25% superset, then combines the blocks into the plate-balanced, capped selection
(``fmharness.heldout.cells.select_cells``), one AnnData file per line, per-line and
per-half pseudobulk log2(CPM+1) tables, a selection-count table, and the tranche record.

Three modes:

* One block (``--block t --n-blocks 64``): the shards whose sorted index ``i`` has
  ``i %% n_blocks == t``. Writes ``dmso_{t}.parquet`` (kept cells' metadata),
  ``dmso_{t}.npz`` (their counts over the Stack panel, as CSR arrays), and
  ``dmso_{t}_all.parquet`` (every DMSO cell of a grid line seen in this block, counted per
  (cellosaurus, plate) -- not just the superset) into ``--cache``.
* ``--combine``: refuses (non-zero exit, naming them) unless every block's three outputs are
  done. Sums the all-DMSO counts to get each line's plate count, selects cells, assigns
  halves, and fails if any (line, plate) superset holds fewer cells than
  ``min(quota, cells actually seen for that (line, plate) in the full scan)`` -- the superset
  reproduces the full-data selection exactly when, and only when, that holds (a sampling
  shortfall the fixed 25%% superset should not have produced). Writes ``cells/line_{i}.h5ad``
  per grid line, ``expression.parquet``,
  ``expression_halves.parquet`` into ``--cache``, ``rung1_cells.csv`` into ``--out-dir``,
  and registers the 50 per-line files as tranche ``tahoe100m-dmso-cells.v1`` under
  ``--tranche-dir`` (default ``data/tranches``).

Every output is followed by a ``<name>.done.json`` completion record
(``fmharness.heldout.records``); a rerun skips an output whose record is already valid.

    uv run python scripts/heldout_dmso_cells.py --block 0 --n-blocks 64 \\
        --grid docs/tasks/rung1-held-out-prediction/rung1_grid.json \\
        --crosswalk docs/tasks/rung1-held-out-prediction/rung1_line_crosswalk.csv \\
        --genelist stack-large/basecount_1000per_15000max.pkl \\
        --cache /scratch/alpine/$USER/rung1_cache

    uv run python scripts/heldout_dmso_cells.py --combine --n-blocks 64 \\
        --grid docs/tasks/rung1-held-out-prediction/rung1_grid.json \\
        --crosswalk docs/tasks/rung1-held-out-prediction/rung1_line_crosswalk.csv \\
        --genelist stack-large/basecount_1000per_15000max.pkl \\
        --cache /scratch/alpine/$USER/rung1_cache \\
        --out-dir docs/tasks/rung1-held-out-prediction
"""

# pandas and scipy ship no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *their* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from datetime import date
from pathlib import Path
from typing import IO, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmharness.heldout.cells import (  # noqa: E402
    DMSO_DRUG,
    cell_keys,
    half_of,
    pseudobulk_log_cpm,
    select_cells,
    superset_mask,
)
from fmharness.heldout.grid import Grid, load_grid  # noqa: E402
from fmharness.heldout.records import is_done, sha256_file, write_record  # noqa: E402
from fmharness.schema import Tranche  # noqa: E402
from fmharness.tahoe import scatter_flat_tokens  # noqa: E402

TAHOE = "tahoebio/Tahoe-100M"
TAHOE_REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
TRANCHE_ID = "tahoe100m-dmso-cells.v1"
GENE_PANEL_SIZE = 15012
META_COLS = ("drug", "cell_line_id", "plate", "sample")
META_COLUMNS = ("shard_index", "row_group", "row", "key", "cellosaurus", "plate", "sample")


def _empty_meta_frame() -> pd.DataFrame:
    """An empty ``meta`` frame with the same dtypes a non-empty one gets -- in particular
    ``key`` as ``uint64`` throughout, never ``int64``/``object``/``float64`` (ruling 13):
    concatenating a real block's ``uint64`` key column with a placeholder of a different
    dtype would silently upcast the whole column, corrupting every key in it."""
    return pd.DataFrame(
        {
            "shard_index": pd.array([], dtype="int64"),
            "row_group": pd.array([], dtype="int64"),
            "row": pd.array([], dtype="int64"),
            "key": pd.array([], dtype="uint64"),
            "cellosaurus": pd.array([], dtype="object"),
            "plate": pd.array([], dtype="object"),
            "sample": pd.array([], dtype="object"),
        }
    )


def _empty_all_counts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cellosaurus": pd.array([], dtype="object"),
            "plate": pd.array([], dtype="object"),
            "n_dmso_seen": pd.array([], dtype="int64"),
        }
    )


# --------------------------------------------------------------------------------------------
# Small metadata tables: the crosswalk, the Stack gene panel
# --------------------------------------------------------------------------------------------


def load_crosswalk(path: Path) -> pd.DataFrame:
    """``rung1_line_crosswalk.csv`` as written by ``scripts/heldout_answers.py --crosswalk``:
    columns ``line, cellosaurus, cell_name``, keeping the literal ``NA`` line."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _panel_symbols(raw: object) -> list[str]:
    if isinstance(raw, pd.Index | pd.Series):
        return [str(s) for s in raw.tolist()]
    if isinstance(raw, np.ndarray):
        return [str(s) for s in raw.tolist()]
    if isinstance(raw, list | tuple):
        return [str(s) for s in raw]
    raise TypeError(f"unsupported gene-list pickle type: {type(raw)!r}")


def load_gene_panel(
    genelist_path: Path, gene_metadata: pd.DataFrame
) -> tuple[list[str], dict[int, int]]:
    """The Stack gene panel's symbols (in the pickle's own order) and a token-id -> column map.

    Matches panel symbols to ``gene_metadata``'s ``gene_symbol`` case-insensitively
    (uppercase), dropping any panel symbol with no match and any panel symbol whose token
    duplicates an already-placed one. Raises ``SystemExit`` if the pickle does not hold
    15,012 unique symbols (ruling 10).
    """
    with genelist_path.open("rb") as fh:
        raw = pickle.load(fh)  # a static, project-controlled gene-list file
    symbols = _panel_symbols(raw)
    if len(set(symbols)) != GENE_PANEL_SIZE:
        raise SystemExit(
            f"gene-list pickle must hold {GENE_PANEL_SIZE} unique gene symbols, "
            f"found {len(set(symbols))} (of {len(symbols)} entries)"
        )

    meta = cast(Any, gene_metadata)
    upper_to_token: dict[str, int] = {}
    for symbol, token in zip(meta["gene_symbol"].astype(str), meta["token_id"], strict=True):
        upper_to_token.setdefault(symbol.upper(), int(token))

    used_tokens: set[int] = set()
    panel_syms: list[str] = []
    token_to_col: dict[int, int] = {}
    for symbol in symbols:
        token = upper_to_token.get(symbol.upper())
        if token is None or token in used_tokens:
            continue
        used_tokens.add(token)
        panel_syms.append(symbol)
        token_to_col[token] = len(panel_syms) - 1

    print(
        f"gene panel: {len(panel_syms)} of {len(symbols)} panel genes covered by "
        f"Tahoe gene metadata"
    )
    return panel_syms, token_to_col


# --------------------------------------------------------------------------------------------
# One shard: keep DMSO cells of the grid's lines, superset-filtered; decode only those
# --------------------------------------------------------------------------------------------


def scan_shard(
    fileobj: IO[bytes],
    shard_index: int,
    grid_cellosaurus: set[str],
    token_to_col: dict[int, int],
    n_cols: int,
    seed: int = 0,
    fraction: float = 0.25,
) -> tuple[pd.DataFrame, sparse.csr_matrix, pd.DataFrame]:
    """Scan one shard's row groups, keeping DMSO cells of ``grid_cellosaurus`` whose key falls
    in the superset.

    Small metadata columns are read per row group first, and a row group with no such cell is
    skipped without ever touching its ``genes``/``expressions`` columns. A row group that does
    have one has those columns decoded whole (parquet decodes a row group at a time, not cell
    by cell), then only the kept rows are taken out of that decode -- via one
    ``pyarrow.compute.take`` over the row group's list columns, not a per-cell Python loop --
    and copied so nothing keeps the row-group buffer pinned in memory. Returns:

    * ``meta`` -- one row per kept cell: ``shard_index, row_group, row, key, cellosaurus,
      plate, sample`` (``key`` always ``uint64``, ruling 13).
    * ``counts`` -- those cells' raw counts over the panel, a CSR aligned with ``meta``'s rows.
    * ``all_counts`` -- one row per (cellosaurus, plate) with ``n_dmso_seen``, counting every
      DMSO cell of a grid line seen in this shard (not just the ones the superset kept) --
      the input ``select_cells`` needs to size each line's per-plate quota.
    """
    pfile = pq.ParquetFile(fileobj)
    meta_blocks: list[pd.DataFrame] = []
    counts_blocks: list[sparse.csr_matrix] = []
    all_counts_blocks: list[pd.DataFrame] = []
    grid_array = np.array(sorted(grid_cellosaurus), dtype=object)

    for rg in range(pfile.num_row_groups):
        table = pfile.read_row_group(rg, columns=list(META_COLS))
        drug = np.asarray(table.column("drug").to_pylist(), dtype=object)
        cl = np.asarray(table.column("cell_line_id").to_pylist(), dtype=object)
        plate = np.asarray(table.column("plate").to_pylist(), dtype=object)
        sample = np.asarray(table.column("sample").to_pylist(), dtype=object)

        is_dmso = drug == DMSO_DRUG
        is_grid = np.isin(cl, grid_array)
        dmso_grid = is_dmso & is_grid
        if not dmso_grid.any():
            continue

        seen = pd.DataFrame({"cellosaurus": cl[dmso_grid], "plate": plate[dmso_grid]})
        all_counts_blocks.append(seen.value_counts().rename("n_dmso_seen").reset_index())

        rows_idx = np.nonzero(dmso_grid)[0]
        keys = cell_keys(shard_index, rg, rows_idx, seed)
        kept = superset_mask(keys, fraction)
        if not kept.any():
            continue
        keep_rows = rows_idx[kept]
        keep_keys = keys[kept].astype(np.uint64)

        meta_blocks.append(
            pd.DataFrame(
                {
                    "shard_index": np.int64(shard_index),
                    "row_group": np.int64(rg),
                    "row": keep_rows.astype(np.int64),
                    "key": keep_keys,
                    "cellosaurus": cl[keep_rows],
                    "plate": plate[keep_rows],
                    "sample": sample[keep_rows],
                }
            )
        )

        # Decoding is at row-group granularity (parquet's own read unit): this reads every
        # cell's genes/expressions in the row group, then `pc.take` pulls out only the kept
        # rows in one vectorized operation -- no per-cell Python loop.
        arrs = pfile.read_row_group(rg, columns=["genes", "expressions"])
        take_idx = pa.array(keep_rows.astype(np.int64))
        genes_taken = pc.take(arrs.column("genes"), take_idx).combine_chunks()
        expr_taken = pc.take(arrs.column("expressions"), take_idx).combine_chunks()
        # .copy(): a taken/flattened array still backs onto the row-group's own allocation.
        gene_values = genes_taken.values.to_numpy(zero_copy_only=False).copy()
        gene_offsets = genes_taken.offsets.to_numpy(zero_copy_only=False)
        expr_values = expr_taken.values.to_numpy(zero_copy_only=False).copy()

        lengths = np.diff(gene_offsets)
        counts_blocks.append(
            scatter_flat_tokens(gene_values, expr_values, lengths, token_to_col, n_cols)
        )

    meta = pd.concat(meta_blocks, ignore_index=True) if meta_blocks else _empty_meta_frame()
    counts = cast(
        sparse.csr_matrix,
        sparse.vstack(counts_blocks).tocsr()
        if counts_blocks
        else sparse.csr_matrix((0, n_cols), dtype=np.float32),
    )
    all_counts = (
        cast(
            pd.DataFrame,
            pd.concat(all_counts_blocks, ignore_index=True)
            .groupby(["cellosaurus", "plate"], as_index=False)["n_dmso_seen"]
            .sum(),
        )
        if all_counts_blocks
        else _empty_all_counts_frame()
    )
    return meta, counts, all_counts


# --------------------------------------------------------------------------------------------
# Block mode: Hugging Face streaming (Alpine-only; not exercised by local tests)
# --------------------------------------------------------------------------------------------


def list_shards(token: str | None) -> list[str]:
    """Every ``data/train-XXXXX-of-03388.parquet`` shard, sorted (deterministic block split)."""
    from huggingface_hub import HfApi  # type: ignore  # Alpine-only, heavy import

    api = HfApi(token=token)
    files = api.list_repo_files(TAHOE, repo_type="dataset", revision=TAHOE_REVISION)
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def run_block(
    t: int,
    n_blocks: int,
    grid_cellosaurus: set[str],
    panel_syms: list[str],
    token_to_col: dict[int, int],
    cache: Path,
    seed: float,
    fraction: float,
) -> None:
    meta_path = cache / f"dmso_{t}.parquet"
    npz_path = cache / f"dmso_{t}.npz"
    all_path = cache / f"dmso_{t}_all.parquet"
    if is_done(meta_path) and is_done(npz_path) and is_done(all_path):
        print(f"block {t} already done, skipping")
        return

    from huggingface_hub import HfFileSystem  # type: ignore  # Alpine-only, heavy import

    token = os.environ.get("HF_TOKEN")
    shards = list_shards(token)
    my_shards = [(i, s) for i, s in enumerate(shards) if i % n_blocks == t]
    fs = HfFileSystem(token=token)
    n_cols = len(panel_syms)

    metas: list[pd.DataFrame] = []
    counts_list: list[sparse.csr_matrix] = []
    all_list: list[pd.DataFrame] = []
    for shard_index, path in my_shards:
        with fs.open(f"datasets/{TAHOE}@{TAHOE_REVISION}/{path}", "rb") as fh:
            meta, counts, all_counts = scan_shard(
                fh, shard_index, grid_cellosaurus, token_to_col, n_cols, int(seed), fraction
            )
        metas.append(meta)
        counts_list.append(counts)
        all_list.append(all_counts)
        print(f"  shard {shard_index} ({path}): {len(meta)} cells kept", flush=True)

    write_block_outputs(meta_path, npz_path, all_path, metas, counts_list, all_list, n_cols)


def write_block_outputs(
    meta_path: Path,
    npz_path: Path,
    all_path: Path,
    metas: list[pd.DataFrame],
    counts_list: list[sparse.csr_matrix],
    all_list: list[pd.DataFrame],
    n_cols: int,
) -> None:
    """Concatenate one block's per-shard results and write its three completion-recorded
    outputs. Split out from ``run_block`` so tests can drive it without Hugging Face."""
    combined_meta = pd.concat(metas, ignore_index=True) if metas else _empty_meta_frame()
    combined_counts = (
        sparse.vstack(counts_list).tocsr()
        if counts_list
        else sparse.csr_matrix((0, n_cols), dtype=np.float32)
    )
    combined_all: pd.DataFrame = (
        cast(
            pd.DataFrame,
            pd.concat(all_list, ignore_index=True)
            .groupby(["cellosaurus", "plate"], as_index=False)["n_dmso_seen"]
            .sum(),
        )
        if all_list
        else _empty_all_counts_frame()
    )

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    combined_meta.to_parquet(meta_path, index=False)
    write_record(meta_path)

    np.savez(
        npz_path,
        data=combined_counts.data,
        indices=combined_counts.indices,
        indptr=combined_counts.indptr,
        shape=np.array(combined_counts.shape, dtype=np.int64),
    )
    write_record(npz_path)

    combined_all.to_parquet(all_path, index=False)
    write_record(all_path)
    print(
        f"wrote {meta_path.name}: {len(combined_meta)} cells; "
        f"{all_path.name}: {len(combined_all)} (cellosaurus, plate) rows"
    )


# --------------------------------------------------------------------------------------------
# Combine: select, halve, pseudobulk, write per-line AnnData, register the tranche
# --------------------------------------------------------------------------------------------


def write_line_h5ad(
    path: Path, counts: sparse.csr_matrix, obs: pd.DataFrame, panel_syms: list[str]
) -> None:
    """One grid line's selected cells as an AnnData: raw counts (float32 CSR) over the panel,
    ``obs`` as given, ``var`` indexed by the panel symbols with a ``feature_name`` column.

    Imports ``anndata`` locally (ruling 9): every other stage of this script is anndata-free.
    """
    import anndata as ad  # local import: only this function needs it

    adata = ad.AnnData(X=counts.astype(np.float32), obs=obs.reset_index(drop=True))
    adata.obs_names = [str(i) for i in range(adata.n_obs)]
    adata.var_names = list(panel_syms)
    adata.var["feature_name"] = panel_syms
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _combined_block_frames(
    cache: Path, n_blocks: int
) -> tuple[pd.DataFrame, sparse.csr_matrix, pd.DataFrame]:
    meta_paths = [cache / f"dmso_{t}.parquet" for t in range(n_blocks)]
    npz_paths = [cache / f"dmso_{t}.npz" for t in range(n_blocks)]
    all_paths = [cache / f"dmso_{t}_all.parquet" for t in range(n_blocks)]
    missing = [str(p) for p in (*meta_paths, *npz_paths, *all_paths) if not is_done(p)]
    if missing:
        raise SystemExit(
            f"combine refuses: {len(missing)} of {3 * n_blocks} block outputs are missing "
            f"or invalid, run --block for each of: {missing}"
        )

    meta = pd.concat([pd.read_parquet(p) for p in meta_paths], ignore_index=True)
    mats: list[sparse.csr_matrix] = []
    n_cols: int | None = None
    for p in npz_paths:
        with np.load(p) as d:
            shape = tuple(int(v) for v in d["shape"])
            n_cols = shape[1] if n_cols is None else n_cols
            mats.append(sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=shape))
    counts = cast(
        sparse.csr_matrix,
        sparse.vstack(mats).tocsr()
        if mats
        else sparse.csr_matrix((0, n_cols or 0), dtype=np.float32),
    )
    all_counts = cast(
        pd.DataFrame,
        pd.concat([pd.read_parquet(p) for p in all_paths], ignore_index=True)
        .groupby(["cellosaurus", "plate"], as_index=False)["n_dmso_seen"]
        .sum(),
    )

    # Sort by key (unique across all cells) before anything downstream reads row order, so
    # every output -- selection, the h5ad row order, pseudobulk -- is independent of how many
    # blocks the scan ran in or what order its shards were read in (ruling: byte-identical
    # outputs regardless of --n-blocks).
    order = np.argsort(meta["key"].to_numpy(dtype=np.uint64), kind="stable")
    meta = meta.iloc[order].reset_index(drop=True)
    counts = counts[order]
    return meta, counts, all_counts


def combine(
    grid: Grid,
    crosswalk: pd.DataFrame,
    panel_syms: list[str],
    cache: Path,
    out_dir: Path,
    tranche_dir: Path,
    n_blocks: int,
    per_line: int,
) -> None:
    meta, counts, all_counts = _combined_block_frames(cache, n_blocks)

    cellosaurus_to_line = dict(
        zip(crosswalk["cellosaurus"].astype(str), crosswalk["line"].astype(str), strict=True)
    )
    meta = meta.assign(line=meta["cellosaurus"].astype(str).map(cellosaurus_to_line.get))
    all_counts = all_counts.assign(
        line=all_counts["cellosaurus"].astype(str).map(cellosaurus_to_line.get)
    )

    plates_per_line: dict[str, int] = (
        all_counts[all_counts["n_dmso_seen"] > 0].groupby("line")["plate"].nunique().to_dict()
    )
    missing_lines = sorted(ln for ln in grid.lines if ln not in plates_per_line)
    if missing_lines:
        raise SystemExit(f"no DMSO cells at all (in any block) for grid lines: {missing_lines}")

    meta_for_selection = cast(pd.DataFrame, meta[["line", "plate", "key"]])
    selected = select_cells(meta_for_selection, plates_per_line, per_line=per_line)

    quota_by_line = {ln: math.ceil(per_line / n) for ln, n in plates_per_line.items()}
    superset_size = cast(pd.Series, meta.groupby(["line", "plate"]).size())
    superset_counts = superset_size.reset_index(name="n_superset")
    full_counts = cast(
        pd.DataFrame,
        all_counts.rename(columns={"n_dmso_seen": "n_full"})[["line", "plate", "n_full"]],
    )
    joined = superset_counts.merge(full_counts, on=["line", "plate"], how="outer")
    joined["n_superset"] = joined["n_superset"].fillna(0).astype(int)
    joined["n_full"] = joined["n_full"].fillna(0).astype(int)
    joined["quota"] = joined["line"].map(quota_by_line.get)
    # The superset reproduces the full-data selection for a (line, plate) exactly when it
    # holds at least min(quota, n_full) cells -- a plate with fewer than its quota available
    # in the FULL scan can only ever give what it has (no shortfall), but the superset must
    # never come up short of what the full scan shows was actually there, up to the quota.
    required = np.minimum(joined["quota"].to_numpy(), joined["n_full"].to_numpy())
    shortfall = cast(pd.DataFrame, joined[joined["n_superset"].to_numpy() < required])
    if not shortfall.empty:
        names = [f"{r.line}/{r.plate}" for r in cast(Any, shortfall.itertuples())]
        raise SystemExit(
            f"superset undercounts {len(names)} (line, plate) pair(s) below "
            f"min(quota, full scan count) -- the 25% superset should not have missed cells "
            f"the full scan shows were actually there: {names}"
        )

    meta = meta.assign(
        half=half_of(meta["key"].to_numpy(dtype=np.uint64)),
        selected=selected,
    )
    selected_idx = np.nonzero(selected)[0]
    kept = meta.iloc[selected_idx].reset_index(drop=True)
    kept_counts = counts[selected_idx]

    cells_dir = cache / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(grid.lines):
        line_mask = (kept["line"] == line).to_numpy()
        obs = pd.DataFrame(
            {
                "cellosaurus": kept.loc[line_mask, "cellosaurus"].to_numpy(),
                "line": kept.loc[line_mask, "line"].to_numpy(),
                "plate": kept.loc[line_mask, "plate"].to_numpy(),
                "key": [
                    f"{int(k):016x}" for k in kept.loc[line_mask, "key"].to_numpy(dtype=np.uint64)
                ],
                "half": kept.loc[line_mask, "half"].to_numpy(dtype=np.int8),
            }
        )
        line_path = cells_dir / f"line_{i}.h5ad"
        write_line_h5ad(line_path, kept_counts[line_mask], obs, panel_syms)
        write_record(line_path)

    labels, log2cpm = pseudobulk_log_cpm(kept_counts, kept["line"].tolist())
    expr = pd.DataFrame(log2cpm, index=pd.Index(labels, name="line"), columns=panel_syms)
    expr_path = cache / "expression.parquet"
    expr.to_parquet(expr_path)
    write_record(expr_path)

    half_groups = list(zip(kept["line"].tolist(), kept["half"].tolist(), strict=True))
    half_labels, half_log2cpm = pseudobulk_log_cpm(kept_counts, half_groups)
    half_index = pd.MultiIndex.from_tuples(
        [(str(line), int(half)) for line, half in half_labels], names=["line", "half"]
    )
    expr_halves = pd.DataFrame(half_log2cpm, index=half_index, columns=panel_syms)
    expr_halves_path = cache / "expression_halves.parquet"
    expr_halves.to_parquet(expr_halves_path)
    write_record(expr_halves_path)

    n_selected_size = cast(pd.Series, kept.groupby(["line", "plate"]).size())
    n_selected = n_selected_size.reset_index(name="n_selected")
    cells_table = joined.merge(n_selected, on=["line", "plate"], how="left")
    cells_table["n_selected"] = cells_table["n_selected"].fillna(0).astype(int)
    cells_table = cells_table.rename(columns={"n_full": "n_dmso_seen"})
    cells_table = cast(
        pd.DataFrame, cells_table[["line", "plate", "n_dmso_seen", "n_superset", "n_selected"]]
    )
    cells_table = cells_table.sort_values(by=["line", "plate"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_csv_path = out_dir / "rung1_cells.csv"
    cells_table.to_csv(cells_csv_path, index=False)
    write_record(cells_csv_path)

    register_tranche(grid, cells_dir, tranche_dir)
    print(
        f"combine: {len(kept)} cells selected over {len(grid.lines)} lines "
        f"({len(cells_table)} (line, plate) rows)"
    )


def manifest_lines(cells_dir: Path, grid: Grid) -> list[tuple[str, int, str]]:
    """(relpath, size, sha256) for the 50 per-line files, sorted by relpath."""
    rows: list[tuple[str, int, str]] = []
    for i in range(len(grid.lines)):
        path = cells_dir / f"line_{i}.h5ad"
        if not path.exists():
            raise SystemExit(f"expected per-line file missing: {path}")
        rel = f"cells/{path.name}"
        rows.append((rel, path.stat().st_size, sha256_file(path)))
    return sorted(rows, key=lambda r: r[0])


def register_tranche(grid: Grid, cells_dir: Path, tranche_dir: Path) -> None:
    """Write ``tahoe100m-dmso-cells.v1``'s manifest and record (ruling 8). Refuses to
    overwrite an existing record, per every other tranche registration in this project."""
    tranche_path = tranche_dir / f"{TRANCHE_ID}.json"
    manifest_path = tranche_dir / f"{TRANCHE_ID}.manifest.txt"
    if tranche_path.exists():
        raise SystemExit(f"{tranche_path} exists; a tranche is ingested once, then immutable")

    manifest = manifest_lines(cells_dir, grid)
    tranche_dir.mkdir(parents=True, exist_ok=True)
    manifest_text = "".join(f"{rel}\t{size}\t{sha}\n" for rel, size, sha in manifest)
    manifest_path.write_text(manifest_text)

    tranche = Tranche(
        tranche_id=TRANCHE_ID,
        source=f"{TAHOE}:expression_data",
        version=TAHOE_REVISION,
        ingestion_date=date.today(),
        patient_count=0,
        sample_count=len(grid.lines),
        drug_count=0,
        content_hash=sha256_file(manifest_path),
        description=(
            "Up to 1,000 DMSO_TF (vehicle control) cells per Tahoe-100M grid cell line, "
            "selected with a fixed seed (0): a 25% key-based superset "
            "(fmharness.heldout.cells.superset_mask), then the smallest-key cells per "
            "(line, plate) up to ceil(1000 / plates for that line), capped at 1,000 per "
            "line (fmharness.heldout.cells.select_cells). Built by "
            "scripts/heldout_dmso_cells.py; see docs/DATA.md."
        ),
    )
    tranche_path.write_text(tranche.model_dump_json(indent=2) + "\n")
    print(f"registered {TRANCHE_ID}: {len(manifest)} files, version {TAHOE_REVISION}")
    print(f"content_hash {tranche.content_hash}")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--crosswalk", type=Path, required=True)
    ap.add_argument("--genelist", type=Path, required=True)
    ap.add_argument(
        "--gene-metadata",
        type=Path,
        help="metadata/gene_metadata.parquet (Tahoe HF revision); read locally if given, "
        "else fetched from Hugging Face",
    )
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--tranche-dir", type=Path, default=Path("data/tranches"))
    ap.add_argument("--block", type=int)
    ap.add_argument("--n-blocks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--per-line", type=int, default=1000)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    crosswalk = load_crosswalk(args.crosswalk)

    if args.gene_metadata is not None:
        gene_metadata = pd.read_parquet(args.gene_metadata)
    else:
        from huggingface_hub import HfFileSystem  # type: ignore  # Alpine-only, heavy import

        fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
        with fs.open(
            f"datasets/{TAHOE}@{TAHOE_REVISION}/metadata/gene_metadata.parquet", "rb"
        ) as fh:
            gene_metadata = pd.read_parquet(fh)
    panel_syms, token_to_col = load_gene_panel(args.genelist, gene_metadata)

    if args.combine:
        if args.out_dir is None:
            raise SystemExit("--combine needs --out-dir")
        combine(
            grid,
            crosswalk,
            panel_syms,
            args.cache,
            args.out_dir,
            args.tranche_dir,
            args.n_blocks,
            args.per_line,
        )
        return

    if args.block is None:
        raise SystemExit("need --block <t> --n-blocks <n> (or --combine)")

    grid_cellosaurus = set(crosswalk["cellosaurus"].astype(str))
    run_block(
        args.block,
        args.n_blocks,
        grid_cellosaurus,
        panel_syms,
        token_to_col,
        args.cache,
        args.seed,
        args.fraction,
    )


if __name__ == "__main__":
    main()
