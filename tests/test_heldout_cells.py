"""Task 4 of rung 1: DMSO cell keys, plate-balanced selection, halves, pseudobulk log CPM.

Every test runs the REAL functions from ``fmharness.heldout.cells`` and
``scripts/heldout_dmso_cells.py`` -- known-answer fixtures for the pure pieces, then one
end-to-end run of the block and combine functions over tiny local parquet shards in Tahoe's
own column shape.
"""

# pandas and scipy ship no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *their* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import importlib.util
import math
import pickle
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

from fmharness.heldout.cells import (
    cell_keys,
    half_of,
    pseudobulk_log_cpm,
    select_cells,
    superset_mask,
)
from fmharness.heldout.grid import Grid
from fmharness.heldout.records import is_done
from fmharness.schema import Tranche

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hc = _load_module("heldout_dmso_cells", "scripts/heldout_dmso_cells.py")

pytestmark = pytest.mark.known_answer


# ==============================================================================================
# cell_keys
# ==============================================================================================


def test_cell_keys_independent_of_shard_read_order() -> None:
    """Keys depend only on (shard_index, row_group, row), never on the order cells are
    processed in -- e.g. shard 7 read before shard 2 gives shard 7's cells the same keys."""
    positions = [(7, 3, 0), (2, 1, 5), (7, 3, 1), (0, 0, 0), (2, 1, 4)]

    def keys_for(order: list[int]) -> dict[tuple[int, int, int], int]:
        out: dict[tuple[int, int, int], int] = {}
        for idx in order:
            shard, rg, row = positions[idx]
            k = cell_keys(shard, rg, np.array([row]))
            out[positions[idx]] = int(k[0])
        return out

    forward = keys_for([0, 1, 2, 3, 4])
    shuffled = keys_for([3, 1, 4, 0, 2])
    assert forward == shuffled

    # also identical when computed as one vectorized batch, any order
    shards = np.array([p[0] for p in positions])
    rgs = np.array([p[1] for p in positions])
    rows = np.array([p[2] for p in positions])
    batch = cell_keys(shards, rgs, rows)
    for i, pos in enumerate(positions):
        assert int(batch[i]) == forward[pos]


def test_cell_keys_seed_changes_keys() -> None:
    rows = np.arange(20)
    a = cell_keys(1, 0, rows, seed=0)
    b = cell_keys(1, 0, rows, seed=1)
    assert not np.array_equal(a, b)


def test_cell_keys_dtype_is_uint64() -> None:
    assert cell_keys(0, 0, np.array([0, 1, 2])).dtype == np.uint64


# ==============================================================================================
# superset_mask
# ==============================================================================================


def test_superset_mask_matches_threshold_definition() -> None:
    keys = np.array([0, 2**63, 2**64 - 1, 2**62 - 1, 2**62], dtype=np.uint64)
    mask = superset_mask(keys, fraction=0.25)
    threshold = math.floor(0.25 * 2**64)
    expected = keys < np.uint64(threshold)
    assert np.array_equal(mask, expected)


def test_superset_mask_roughly_the_right_fraction() -> None:
    keys = cell_keys(np.zeros(200_000, dtype=np.int64), 0, np.arange(200_000))
    mask = superset_mask(keys, fraction=0.25)
    assert 0.23 < mask.mean() < 0.27


# ==============================================================================================
# half_of
# ==============================================================================================


def test_half_of_is_bit_one_and_independent_of_superset_threshold() -> None:
    keys = np.arange(16, dtype=np.uint64)
    halves = half_of(keys)
    expected = ((keys >> np.uint64(1)) & np.uint64(1)).astype(np.int8)
    assert np.array_equal(halves, expected)
    assert halves.dtype == np.int8


def test_halves_are_about_balanced() -> None:
    keys = cell_keys(0, 0, np.arange(10_000))
    halves = half_of(keys)
    assert 0.45 < halves.mean() < 0.55


def test_halves_unchanged_by_selection() -> None:
    """A cell's half never changes because of which cells were selected around it -- selection
    reads bit 0's threshold; half reads bit 1. Selecting a subset of rows and recomputing
    half_of on just that subset gives the same values as the full computation, row for row."""
    n = 5000
    keys = cell_keys(3, 1, np.arange(n))
    meta = pd.DataFrame(
        {
            "line": np.where(np.arange(n) % 2 == 0, "L1", "L2"),
            "plate": np.where((np.arange(n) // 10) % 2 == 0, "P1", "P2"),
            "key": keys,
        }
    )
    full_halves = half_of(keys)
    plates_per_line = {"L1": 2, "L2": 2}
    selected = select_cells(meta, plates_per_line, per_line=50)
    assert np.array_equal(half_of(meta.loc[selected, "key"].to_numpy()), full_halves[selected])


# ==============================================================================================
# select_cells: quota, cap, uneven plates
# ==============================================================================================


def test_select_cells_quota_and_cap_exact_with_uneven_plates() -> None:
    """L1 has 3 plates (quota = ceil(10/3) = 4 per plate, cap 10 total); one of its plates
    (P3) has only 2 cells -- fewer than its quota of 4 -- and should give all 2 of them
    without error. L2 has 1 plate with plenty of cells (quota = cap = 10)."""
    rows: list[dict[str, object]] = []
    rng_key = 0
    # L1/P1: 6 cells, L1/P2: 6 cells, L1/P3: 2 cells (short of its quota of 4)
    for plate, n in (("P1", 6), ("P2", 6), ("P3", 2)):
        for _ in range(n):
            rows.append({"line": "L1", "plate": plate, "key": np.uint64(rng_key)})
            rng_key += 1
    # L2/P1: 30 cells, one plate, quota = cap = 10
    for _ in range(30):
        rows.append({"line": "L2", "plate": "P1", "key": np.uint64(rng_key)})
        rng_key += 1
    meta = pd.DataFrame(rows)

    plates_per_line = {"L1": 3, "L2": 1}
    selected = select_cells(meta, plates_per_line, per_line=10)

    kept = meta[selected]
    counts = kept.groupby(["line", "plate"]).size()
    assert counts.loc[("L1", "P1")] == 4  # quota, plenty available
    assert counts.loc[("L1", "P2")] == 4  # quota, plenty available
    assert counts.loc[("L1", "P3")] == 2  # short plate: gives all it has
    assert kept[kept["line"] == "L1"].shape[0] == 10  # per-line cap still exactly 10
    assert counts.loc[("L2", "P1")] == 10  # single plate: quota == cap

    # the kept cells are exactly the smallest keys within each stage (known-answer: keys were
    # assigned in ascending insertion order per plate/line above)
    l1p1_keys = sorted(meta[(meta["line"] == "L1") & (meta["plate"] == "P1")]["key"])
    kept_l1p1_keys = sorted(kept[(kept["line"] == "L1") & (kept["plate"] == "P1")]["key"])
    assert kept_l1p1_keys == l1p1_keys[:4]


def test_select_cells_missing_line_in_plates_per_line_raises() -> None:
    meta = pd.DataFrame({"line": ["L1"], "plate": ["P1"], "key": np.array([0], dtype=np.uint64)})
    with pytest.raises(ValueError, match="L1"):
        select_cells(meta, {}, per_line=10)


# ==============================================================================================
# pseudobulk_log_cpm
# ==============================================================================================


def test_pseudobulk_log_cpm_hand_computed_three_cells() -> None:
    # 2 genes, 3 cells: cell 0,1 -> group "A"; cell 2 -> group "B"
    counts = sparse.csr_matrix(np.array([[2.0, 8.0], [4.0, 6.0], [10.0, 0.0]], dtype=np.float32))
    groups = ["A", "A", "B"]
    labels, log2cpm = pseudobulk_log_cpm(counts, groups)
    assert list(labels) == ["A", "B"]

    # group A: summed counts [6, 14], total 20 -> CPM [300000, 700000]
    a_cpm = np.array([6.0, 14.0]) / 20.0 * 1_000_000.0
    a_expected = np.log2(a_cpm + 1.0)
    # group B: summed counts [10, 0], total 10 -> CPM [1000000, 0]
    b_cpm = np.array([10.0, 0.0]) / 10.0 * 1_000_000.0
    b_expected = np.log2(b_cpm + 1.0)

    np.testing.assert_allclose(log2cpm[0], a_expected)
    np.testing.assert_allclose(log2cpm[1], b_expected)


def test_pseudobulk_log_cpm_tuple_groups_sorted_labels() -> None:
    counts = sparse.csr_matrix(np.eye(4, dtype=np.float32))
    groups = [("L2", 0), ("L1", 1), ("L1", 0), ("L2", 1)]
    labels, log2cpm = pseudobulk_log_cpm(counts, groups)
    assert list(labels) == [("L1", 0), ("L1", 1), ("L2", 0), ("L2", 1)]
    assert log2cpm.shape == (4, 4)


# ==============================================================================================
# gene panel loading
# ==============================================================================================


def test_load_gene_panel_rejects_wrong_symbol_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hc, "GENE_PANEL_SIZE", 5)
    pkl_path = tmp_path / "genes.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump(["G1", "G2", "G3"], fh)
    gene_metadata = pd.DataFrame({"gene_symbol": ["G1", "G2", "G3"], "token_id": [1, 2, 3]})
    with pytest.raises(SystemExit, match="5"):
        hc.load_gene_panel(pkl_path, gene_metadata)


def test_load_gene_panel_matches_case_insensitively_and_dedupes_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hc, "GENE_PANEL_SIZE", 3)
    pkl_path = tmp_path / "genes.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump(["g1", "G2", "g3"], fh)
    # two metadata rows map to the same uppercase symbol -> same token; g3 has no match
    gene_metadata = pd.DataFrame({"gene_symbol": ["G1", "g2", "G2"], "token_id": [101, 102, 102]})
    panel_syms, token_to_col = hc.load_gene_panel(pkl_path, gene_metadata)
    assert panel_syms == ["g1", "G2"]
    assert token_to_col == {101: 0, 102: 1}


# ==============================================================================================
# End-to-end: two tiny local shards -> scan_shard -> block outputs -> combine
# ==============================================================================================

PANEL_SYMS = ["GA", "GB", "GC"]
TOKEN_TO_COL = {101: 0, 102: 1, 103: 2}
MARKER_TOKEN = 999  # not in the panel -- dropped by scatter_tokens


def _cell(genes: list[int], expr: list[float]) -> tuple[list[int], list[float]]:
    return [MARKER_TOKEN, *genes], [0.0, *expr]


def _write_shard(path: Path, rows: list[dict[str, object]]) -> None:
    genes_col = [r["genes"] for r in rows]
    expr_col = [r["expressions"] for r in rows]
    table = pa.table(
        {
            "drug": pa.array([r["drug"] for r in rows], type=pa.string()),
            "cell_line_id": pa.array([r["cell_line_id"] for r in rows], type=pa.string()),
            "plate": pa.array([r["plate"] for r in rows], type=pa.string()),
            "sample": pa.array([r["sample"] for r in rows], type=pa.string()),
            "pubchem_cid": pa.array([r["pubchem_cid"] for r in rows], type=pa.string()),
            "genes": pa.array(genes_col, type=pa.list_(pa.int64())),
            "expressions": pa.array(expr_col, type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, path)


def _fixture_shards(tmp_path: Path) -> tuple[Path, Path]:
    """Two tiny shards: DMSO + treated cells for grid lines L1 (2 plates), L2 (1 plate), plus
    a non-DMSO cell and a non-grid line's DMSO cell that must both be dropped."""
    rows0: list[dict[str, object]] = []
    for i in range(3):  # L1 / P1: 3 DMSO cells
        genes, expr = _cell([101, 102], [float(i + 1), float(2 * i + 1)])
        rows0.append(
            dict(
                drug="DMSO_TF",
                cell_line_id="CVCL_L1",
                plate="P1",
                sample=f"s0_{i}",
                pubchem_cid="",
                genes=genes,
                expressions=expr,
            )
        )
    # a treated (non-DMSO) cell of L1 -- must be dropped
    genes, expr = _cell([101], [99.0])
    rows0.append(
        dict(
            drug="Some Drug",
            cell_line_id="CVCL_L1",
            plate="P1",
            sample="s0_trt",
            pubchem_cid="123",
            genes=genes,
            expressions=expr,
        )
    )
    # a DMSO cell of a non-grid line -- must be dropped
    genes, expr = _cell([103], [7.0])
    rows0.append(
        dict(
            drug="DMSO_TF",
            cell_line_id="CVCL_OUT",
            plate="P1",
            sample="s0_out",
            pubchem_cid="",
            genes=genes,
            expressions=expr,
        )
    )
    shard0 = tmp_path / "shard0.parquet"
    _write_shard(shard0, rows0)

    rows1: list[dict[str, object]] = []
    for i in range(2):  # L1 / P2: 2 DMSO cells
        genes, expr = _cell([102, 103], [float(i + 5), float(i + 1)])
        rows1.append(
            dict(
                drug="DMSO_TF",
                cell_line_id="CVCL_L1",
                plate="P2",
                sample=f"s1_{i}",
                pubchem_cid="",
                genes=genes,
                expressions=expr,
            )
        )
    for i in range(2):  # L2 / P1: 2 DMSO cells
        genes, expr = _cell([101, 103], [float(i + 10), float(i + 1)])
        rows1.append(
            dict(
                drug="DMSO_TF",
                cell_line_id="CVCL_L2",
                plate="P1",
                sample=f"s1_l2_{i}",
                pubchem_cid="",
                genes=genes,
                expressions=expr,
            )
        )
    shard1 = tmp_path / "shard1.parquet"
    _write_shard(shard1, rows1)
    return shard0, shard1


GRID = Grid(lines=("L1", "L2"), drugs=(), metadata_name={}, excluded_pairs=())
CROSSWALK = pd.DataFrame(
    {"line": ["L1", "L2"], "cellosaurus": ["CVCL_L1", "CVCL_L2"], "cell_name": ["Name1", "Name2"]}
)


@pytest.mark.step_build
def test_dmso_cells_end_to_end(tmp_path: Path) -> None:
    shard0, shard1 = _fixture_shards(tmp_path)
    grid_cellosaurus = {"CVCL_L1", "CVCL_L2"}

    with shard0.open("rb") as fh:
        meta0, counts0, all0 = hc.scan_shard(
            fh, 0, grid_cellosaurus, TOKEN_TO_COL, len(PANEL_SYMS), seed=0, fraction=1.0
        )
    with shard1.open("rb") as fh:
        meta1, counts1, all1 = hc.scan_shard(
            fh, 1, grid_cellosaurus, TOKEN_TO_COL, len(PANEL_SYMS), seed=0, fraction=1.0
        )

    # dropped rows never appear
    assert not meta0["cellosaurus"].eq("CVCL_OUT").any()
    assert len(meta0) == 3  # only the 3 DMSO L1/P1 cells from shard 0
    assert len(meta1) == 4  # 2 L1/P2 + 2 L2/P1

    cache = tmp_path / "cache"
    n_blocks = 1
    hc.write_block_outputs(
        cache / "dmso_0.parquet",
        cache / "dmso_0.npz",
        cache / "dmso_0_all.parquet",
        [meta0, meta1],
        [counts0, counts1],
        [all0, all1],
        len(PANEL_SYMS),
    )
    assert is_done(cache / "dmso_0.parquet")
    assert is_done(cache / "dmso_0.npz")
    assert is_done(cache / "dmso_0_all.parquet")

    out_dir = tmp_path / "out"
    tranche_dir = tmp_path / "tranches"
    hc.combine(
        GRID, CROSSWALK, PANEL_SYMS, cache, out_dir, tranche_dir, n_blocks=n_blocks, per_line=10
    )

    # ---- per-line h5ad ----
    import anndata as ad

    l1 = ad.read_h5ad(cache / "cells" / "line_0.h5ad")
    l2 = ad.read_h5ad(cache / "cells" / "line_1.h5ad")
    assert list(l1.obs.columns) == ["cellosaurus", "line", "plate", "key", "half"]
    assert (l1.obs["line"] == "L1").all()
    assert l1.n_obs == 5  # all 3 (P1) + 2 (P2) DMSO L1 cells kept (per_line=10, no cap bite)
    assert l2.n_obs == 2
    assert list(l1.var["feature_name"]) == PANEL_SYMS
    assert list(l1.var_names) == PANEL_SYMS
    l1_x = cast(sparse.csr_matrix, l1.X)
    assert l1_x.dtype == np.float32
    # first L1/P1 cell: genes [101,102] -> cols [0,1], values [1.0, 1.0] (i=0: i+1=1, 2*0+1=1)
    row0 = np.asarray(l1_x[(l1.obs["plate"] == "P1").to_numpy()][0].todense()).ravel()
    assert row0.tolist() == [1.0, 1.0, 0.0]
    assert all(len(k) == 16 for k in l1.obs["key"])  # 16-digit lowercase hex
    assert all(k == k.lower() for k in l1.obs["key"])

    # ---- expression tables ----
    expr = pd.read_parquet(cache / "expression.parquet")
    assert sorted(expr.index.tolist()) == ["L1", "L2"]
    assert list(expr.columns) == PANEL_SYMS
    assert is_done(cache / "expression.parquet")

    expr_halves = pd.read_parquet(cache / "expression_halves.parquet")
    assert list(expr_halves.index.names) == ["line", "half"]
    assert is_done(cache / "expression_halves.parquet")

    # ---- rung1_cells.csv ----
    cells_csv = pd.read_csv(out_dir / "rung1_cells.csv")
    assert list(cells_csv.columns) == ["line", "plate", "n_dmso_seen", "n_superset", "n_selected"]
    row = cells_csv[(cells_csv["line"] == "L1") & (cells_csv["plate"] == "P1")].iloc[0]
    assert row["n_dmso_seen"] == 3
    assert row["n_superset"] == 3
    assert row["n_selected"] == 3
    assert is_done(out_dir / "rung1_cells.csv")

    # ---- tranche ----
    tranche_path = tranche_dir / "tahoe100m-dmso-cells.v1.json"
    manifest_path = tranche_dir / "tahoe100m-dmso-cells.v1.manifest.txt"
    tranche = Tranche.model_validate_json(tranche_path.read_text())
    assert tranche.tranche_id == "tahoe100m-dmso-cells.v1"
    assert tranche.sample_count == 2
    manifest_lines_read = manifest_path.read_text().splitlines()
    assert len(manifest_lines_read) == 2
    assert manifest_lines_read == sorted(manifest_lines_read)
    for line in manifest_lines_read:
        rel, _size, sha = line.split("\t")
        assert rel.startswith("cells/line_")
        assert len(sha) == 64

    # ---- registration refuses a second time ----
    with pytest.raises(SystemExit, match="exists"):
        hc.register_tranche(GRID, cache / "cells", tranche_dir)


@pytest.mark.step_build
def test_combine_refuses_unless_all_blocks_done(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="missing"):
        hc.combine(
            GRID, CROSSWALK, PANEL_SYMS, cache, out_dir, tmp_path / "tr", n_blocks=1, per_line=10
        )


@pytest.mark.step_build
def test_combine_fails_on_quota_shortfall(tmp_path: Path) -> None:
    """A (line, plate) whose superset undercounts its quota, while the full scan shows more
    cells were available, must fail rather than silently under-filling."""
    # L1 has 1 plate; quota == cap == 5. The superset (what scan_shard kept) has only 2 rows,
    # but the full scan (all_counts) says 10 cells were seen on that plate -- a shortfall.
    meta = pd.DataFrame(
        {
            "shard_index": [0, 0],
            "row_group": [0, 0],
            "row": [0, 1],
            "key": np.array([1, 2], dtype=np.uint64),
            "cellosaurus": ["CVCL_L1", "CVCL_L1"],
            "plate": ["P1", "P1"],
            "sample": ["s0", "s1"],
        }
    )
    counts = sparse.csr_matrix((2, len(PANEL_SYMS)), dtype=np.float32)
    all_counts = pd.DataFrame({"cellosaurus": ["CVCL_L1"], "plate": ["P1"], "n_dmso_seen": [10]})

    cache = tmp_path / "cache"
    hc.write_block_outputs(
        cache / "dmso_0.parquet",
        cache / "dmso_0.npz",
        cache / "dmso_0_all.parquet",
        [meta],
        [counts],
        [all_counts],
        len(PANEL_SYMS),
    )
    out_dir = tmp_path / "out"
    grid = Grid(lines=("L1",), drugs=(), metadata_name={}, excluded_pairs=())
    crosswalk = pd.DataFrame({"line": ["L1"], "cellosaurus": ["CVCL_L1"], "cell_name": ["N1"]})
    with pytest.raises(SystemExit, match="quota"):
        hc.combine(
            grid, crosswalk, PANEL_SYMS, cache, out_dir, tmp_path / "tr", n_blocks=1, per_line=5
        )
