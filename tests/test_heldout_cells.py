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
import warnings
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
from fmharness.heldout.records import is_done, sha256_file
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


def test_cell_keys_splitmix64_known_answer_pin() -> None:
    """Position (0, 0, 0) with seed 0 packs to 0, so this pins ``cell_keys`` to the reference
    splitmix64 generator's first output from initial state 0: 0xE220A8397B1DCDAF. Our packing
    XORs the seed in before mixing, so seed 0 changes nothing and this position's key is
    exactly splitmix64's first call on state 0 -- the standard published test vector."""
    k = cell_keys(0, 0, np.array([0]), seed=0)[0]
    assert int(k) == 0xE220A8397B1DCDAF


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


def test_superset_mask_coerces_input_to_uint64() -> None:
    """Ruling 13: a caller passing ``int64`` keys (not yet ``uint64``) must not silently get a
    float-promoted, precision-losing comparison -- the input is coerced to ``uint64`` first."""
    keys_i64 = np.array([0, 2**62, 2**63 - 1], dtype=np.int64)
    keys_u64 = keys_i64.astype(np.uint64)
    assert np.array_equal(superset_mask(keys_i64, fraction=0.25), superset_mask(keys_u64, 0.25))


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


def test_halves_are_balanced_among_selected_cells() -> None:
    """A cell's half is bit 1 of its key; selection is governed by an ordering comparison over
    the key's full value (in effect its high-order bits) against per-(line, plate) and
    per-line rank cutoffs -- a different, independent bit. So halves among the cells
    *selection actually keeps* should still land near 50/50, not skew because selection
    happens to correlate with bit 1."""
    n = 20_000
    keys = cell_keys(3, 1, np.arange(n))
    meta = pd.DataFrame(
        {
            "line": np.where(np.arange(n) % 2 == 0, "L1", "L2"),
            "plate": np.where((np.arange(n) // 10) % 2 == 0, "P1", "P2"),
            "key": keys,
        }
    )
    plates_per_line = {"L1": 2, "L2": 2}
    selected = select_cells(meta, plates_per_line, per_line=200)
    kept_halves = half_of(meta.loc[selected, "key"].to_numpy())
    assert selected.sum() > 0
    assert 0.45 < kept_halves.mean() < 0.55
    # and recomputing half_of on just the selected keys matches slicing the full computation --
    # a cell's half is a pure function of its own key, unaffected by which other cells were kept.
    assert np.array_equal(kept_halves, half_of(keys)[selected])


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


def test_select_cells_per_line_cap_actually_bites() -> None:
    """3 plates x 6 cells, per_line=10: the per-plate quota is ceil(10/3)=4, so the plate
    stage alone keeps 12 cells (4 x 3) -- more than the 10-cell cap, so the line-level cap
    must actually remove 2 of them (unlike the fixture above, where the cap was never
    binding). Keys are assigned by plate in increasing blocks (plate A: 0-5, B: 6-11, C:
    12-17), so the known answer is exact: the plate stage keeps {0,1,2,3}, {6,7,8,9},
    {12,13,14,15}; the two largest of those twelve keys (14, 15 -- both plate C) are the ones
    the cap must drop."""
    rows: list[dict[str, object]] = []
    for plate, start in (("A", 0), ("B", 6), ("C", 12)):
        for offset in range(6):
            rows.append({"line": "L1", "plate": plate, "key": np.uint64(start + offset)})
    meta = pd.DataFrame(rows)

    selected = select_cells(meta, {"L1": 3}, per_line=10)
    kept = meta[selected]

    assert kept.shape[0] == 10
    assert set(kept["key"].tolist()) == {0, 1, 2, 3, 6, 7, 8, 9, 12, 13}
    dropped = set(meta[~selected]["key"].tolist())
    # the plate stage itself already excludes keys 4, 5, 10, 11, 16, 17 (each plate's two
    # largest); the cap additionally drops exactly 14 and 15, the two largest cells that
    # survived the plate stage.
    assert {14, 15} <= dropped
    counts = kept.groupby("plate").size()
    assert counts.loc["A"] == 4
    assert counts.loc["B"] == 4
    assert counts.loc["C"] == 2  # the cap took both of C's cap-stage losers


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


def _expected_dmso_cells() -> list[dict[str, object]]:
    """Every DMSO cell the fixture shards actually contain, keyed by its real
    (shard_index, row_group, row) position, with the raw-count vector it must scatter to and
    the key ``cell_keys`` must produce for that exact position -- the ground truth
    ``test_dmso_cells_end_to_end`` checks the pipeline's output against, position by position,
    rather than assuming any particular row order survives selection."""
    records: list[dict[str, object]] = []

    def vec(entries: dict[int, float]) -> np.ndarray:
        v = np.zeros(len(PANEL_SYMS), dtype=np.float32)
        for token, value in entries.items():
            v[TOKEN_TO_COL[token]] = value
        return v

    for i in range(3):  # shard 0, row group 0, rows 0-2: L1/P1
        records.append(
            dict(
                shard_index=0,
                row_group=0,
                row=i,
                cellosaurus="CVCL_L1",
                plate="P1",
                vec=vec({101: float(i + 1), 102: float(2 * i + 1)}),
            )
        )
    for i in range(2):  # shard 1, row group 0, rows 0-1: L1/P2
        records.append(
            dict(
                shard_index=1,
                row_group=0,
                row=i,
                cellosaurus="CVCL_L1",
                plate="P2",
                vec=vec({102: float(i + 5), 103: float(i + 1)}),
            )
        )
    for i in range(2):  # shard 1, row group 0, rows 2-3: L2/P1
        records.append(
            dict(
                shard_index=1,
                row_group=0,
                row=2 + i,
                cellosaurus="CVCL_L2",
                plate="P1",
                vec=vec({101: float(i + 10), 103: float(i + 1)}),
            )
        )
    for record in records:
        shard_index = cast(int, record["shard_index"])
        row_group = cast(int, record["row_group"])
        row = cast(int, record["row"])
        key = cell_keys(shard_index, row_group, np.array([row]))[0]
        record["key_hex"] = f"{int(key):016x}"
    return records


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
    assert all(len(k) == 16 for k in l1.obs["key"])  # 16-digit lowercase hex
    assert all(k == k.lower() for k in l1.obs["key"])

    # Every selected cell's obs key equals the hex of cell_keys recomputed from that cell's
    # real (shard_index, row_group, row) position (ruling 13) -- not merely well-formed, but
    # exactly reproducible from position alone -- and its counts match what that exact cell's
    # tokenized genes/expressions should scatter to (so scan_shard's vectorized take/CSR build
    # is checked cell-by-cell, not just in aggregate).
    expected_by_key = {r["key_hex"]: r for r in _expected_dmso_cells()}
    for adata, _line in ((l1, "L1"), (l2, "L2")):
        x = cast(sparse.csr_matrix, adata.X)
        for row_i, (key_hex, cellosaurus, plate) in enumerate(
            zip(adata.obs["key"], adata.obs["cellosaurus"], adata.obs["plate"], strict=True)
        ):
            assert key_hex in expected_by_key, f"key {key_hex} matches no fixture cell"
            expected = expected_by_key[key_hex]
            assert cellosaurus == expected["cellosaurus"]
            assert plate == expected["plate"]
            got = np.asarray(x[row_i].todense()).ravel()
            np.testing.assert_array_equal(got, expected["vec"])

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
def test_descriptions_read_only_dmso_cells(tmp_path: Path) -> None:
    """Invariant 3 (plan.md): a line's description carries no drug response.

    The filter is structural -- ``scan_shard`` keeps ``drug == DMSO_TF`` alone -- and this
    asserts its effect on the numbers a model is actually handed. The fixture shards carry a
    TREATED L1 cell with an extreme profile (99 counts of GA) and a DMSO cell of a line outside
    the grid; the line's expression must be the pseudobulk of its five DMSO cells and nothing
    else, and the final assertion shows the treated cell would have moved it.
    """
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

    cache = tmp_path / "cache"
    hc.write_block_outputs(
        cache / "dmso_0.parquet",
        cache / "dmso_0.npz",
        cache / "dmso_0_all.parquet",
        [meta0, meta1],
        [counts0, counts1],
        [all0, all1],
        len(PANEL_SYMS),
    )
    hc.combine(
        GRID,
        CROSSWALK,
        PANEL_SYMS,
        cache,
        tmp_path / "out",
        tmp_path / "tranches",
        n_blocks=1,
        per_line=10,
    )

    expression = pd.read_parquet(cache / "expression.parquet")
    assert sorted(expression.index.tolist()) == ["L1", "L2"], "a non-grid line was described"

    def log2_cpm(rows: np.ndarray) -> np.ndarray:
        total = rows.sum(axis=0, dtype=np.float64)
        return np.log2(total / total.sum() * 1_000_000.0 + 1.0)

    dmso_l1 = np.array(
        [
            cast(np.ndarray, record["vec"])
            for record in _expected_dmso_cells()
            if record["cellosaurus"] == "CVCL_L1"
        ]
    )
    treated_l1 = np.zeros(len(PANEL_SYMS), dtype=np.float32)
    treated_l1[TOKEN_TO_COL[101]] = 99.0

    described = expression.loc["L1"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(described, log2_cpm(dmso_l1), rtol=1e-12)
    assert not np.allclose(described, log2_cpm(np.vstack([dmso_l1, treated_l1]))), (
        "the treated cell moves this line's description by nothing, so the assertion above "
        "could not have detected it being read"
    )


@pytest.mark.step_build
def test_write_line_h5ad_warns_about_nothing(tmp_path: Path) -> None:
    """Binding constraint (global-constraints.md): test output must be pristine, anndata's
    ``ImplicitModificationWarning`` included.

    It fires when an AnnData is handed a frame whose index is not strings -- anndata converts
    the index and says so. Building ``obs`` and ``var`` with string indexes BEFORE the object
    exists is the fix; repairing them afterwards leaves the warning already emitted.
    """
    import anndata as ad

    obs = pd.DataFrame(
        {
            "cellosaurus": ["CVCL_L1", "CVCL_L1"],
            "line": ["L1", "L1"],
            "plate": ["P1", "P2"],
            "key": ["0" * 16, "1" * 16],
            "half": np.array([0, 1], dtype=np.int8),
        }
    )
    counts = sparse.csr_matrix(np.ones((2, len(PANEL_SYMS)), dtype=np.float32))

    path = tmp_path / "line_0.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ad.ImplicitModificationWarning)
        hc.write_line_h5ad(path, counts, obs, PANEL_SYMS)

    written = ad.read_h5ad(path)
    assert list(written.var_names) == PANEL_SYMS
    assert list(written.var["feature_name"]) == PANEL_SYMS
    assert list(written.obs["plate"]) == ["P1", "P2"]


@pytest.mark.step_build
def test_cell_selection_is_independent_of_shard_order(tmp_path: Path) -> None:
    """Output bytes must not depend on how shards are split into blocks or the order they are
    scanned in: one run puts both fixture shards into a single block (shard 0 then shard 1);
    the other puts each shard in its own block, scanned in the opposite order (shard 1's block
    written first). Both must select the identical cells and produce byte-identical
    ``line_{i}.h5ad`` files and expression tables."""
    grid_cellosaurus = {"CVCL_L1", "CVCL_L2"}

    def _scan(shard_path: Path, shard_index: int):
        with shard_path.open("rb") as fh:
            return hc.scan_shard(
                fh,
                shard_index,
                grid_cellosaurus,
                TOKEN_TO_COL,
                len(PANEL_SYMS),
                seed=0,
                fraction=1.0,
            )

    shard0, shard1 = _fixture_shards(tmp_path)

    # Run A: one block (n_blocks=1), shards scanned and written in order 0, then 1.
    meta0_a, counts0_a, all0_a = _scan(shard0, 0)
    meta1_a, counts1_a, all1_a = _scan(shard1, 1)
    cache_a = tmp_path / "cache_a"
    hc.write_block_outputs(
        cache_a / "dmso_0.parquet",
        cache_a / "dmso_0.npz",
        cache_a / "dmso_0_all.parquet",
        [meta0_a, meta1_a],
        [counts0_a, counts1_a],
        [all0_a, all1_a],
        len(PANEL_SYMS),
    )
    out_a = tmp_path / "out_a"
    hc.combine(
        GRID, CROSSWALK, PANEL_SYMS, cache_a, out_a, tmp_path / "tr_a", n_blocks=1, per_line=10
    )

    # Run B: two blocks (n_blocks=2), shard 1's block written before shard 0's -- shard_index
    # still reflects each shard's real position (1 and 0 respectively), only the processing
    # and block-file order are shuffled.
    meta1_b, counts1_b, all1_b = _scan(shard1, 1)
    meta0_b, counts0_b, all0_b = _scan(shard0, 0)
    cache_b = tmp_path / "cache_b"
    hc.write_block_outputs(
        cache_b / "dmso_0.parquet",
        cache_b / "dmso_0.npz",
        cache_b / "dmso_0_all.parquet",
        [meta1_b],
        [counts1_b],
        [all1_b],
        len(PANEL_SYMS),
    )
    hc.write_block_outputs(
        cache_b / "dmso_1.parquet",
        cache_b / "dmso_1.npz",
        cache_b / "dmso_1_all.parquet",
        [meta0_b],
        [counts0_b],
        [all0_b],
        len(PANEL_SYMS),
    )
    out_b = tmp_path / "out_b"
    hc.combine(
        GRID, CROSSWALK, PANEL_SYMS, cache_b, out_b, tmp_path / "tr_b", n_blocks=2, per_line=10
    )

    import anndata as ad

    for i in range(len(GRID.lines)):
        path_a = cache_a / "cells" / f"line_{i}.h5ad"
        path_b = cache_b / "cells" / f"line_{i}.h5ad"
        assert sha256_file(path_a) == sha256_file(path_b), f"line_{i}.h5ad differs by run order"
        a = ad.read_h5ad(path_a)
        b = ad.read_h5ad(path_b)
        assert a.obs["key"].tolist() == b.obs["key"].tolist()

    expr_a = pd.read_parquet(cache_a / "expression.parquet")
    expr_b = pd.read_parquet(cache_b / "expression.parquet")
    pd.testing.assert_frame_equal(expr_a, expr_b)

    halves_a = pd.read_parquet(cache_a / "expression_halves.parquet")
    halves_b = pd.read_parquet(cache_b / "expression_halves.parquet")
    pd.testing.assert_frame_equal(halves_a, halves_b)


@pytest.mark.step_build
def test_combine_refuses_unless_all_blocks_done(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="missing"):
        hc.combine(
            GRID, CROSSWALK, PANEL_SYMS, cache, out_dir, tmp_path / "tr", n_blocks=1, per_line=10
        )


def _quota_shortfall_case(tmp_path: Path, n_superset: int, n_full: int, per_line: int) -> None:
    """A single line, single plate (so quota == per_line): ``n_superset`` cells with unique
    keys were kept by the superset filter, while the full (unfiltered) scan of that
    (line, plate) actually saw ``n_full`` DMSO cells. Runs ``combine`` -- raises ``SystemExit``
    if and only if the guard fires."""
    meta = pd.DataFrame(
        {
            "shard_index": [0] * n_superset,
            "row_group": [0] * n_superset,
            "row": list(range(n_superset)),
            "key": np.arange(1, n_superset + 1, dtype=np.uint64),
            "cellosaurus": ["CVCL_L1"] * n_superset,
            "plate": ["P1"] * n_superset,
            "sample": [f"s{i}" for i in range(n_superset)],
        }
    )
    counts = sparse.csr_matrix((n_superset, len(PANEL_SYMS)), dtype=np.float32)
    all_counts = pd.DataFrame(
        {"cellosaurus": ["CVCL_L1"], "plate": ["P1"], "n_dmso_seen": [n_full]}
    )

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
    hc.combine(
        grid, crosswalk, PANEL_SYMS, cache, out_dir, tmp_path / "tr", n_blocks=1, per_line=per_line
    )


@pytest.mark.step_build
def test_quota_shortfall_fails_at_equality_n_full_equals_quota(tmp_path: Path) -> None:
    """n_full == quota (5), superset short (3): the guard is ``n_superset < min(quota,
    n_full)``, and at equality ``min(5, 5) == 5``, so 3 < 5 must still fail -- the old
    ``n_full > quota`` (strict) guard would have missed this exactly-at-equality case."""
    with pytest.raises(SystemExit, match="quota"):
        _quota_shortfall_case(tmp_path, n_superset=3, n_full=5, per_line=5)


@pytest.mark.step_build
def test_quota_shortfall_fails_when_full_scan_below_quota_but_above_superset(
    tmp_path: Path,
) -> None:
    """n_superset (3) < n_full (5) < quota (10): the full scan never had enough cells to hit
    quota, but it had more than the superset kept, so the superset still undercounted what was
    actually there. The old ``n_full > quota`` guard would have passed this silently (n_full=5
    is not > quota=10), even though the superset shorted a real, recoverable shortfall."""
    with pytest.raises(SystemExit, match="quota"):
        _quota_shortfall_case(tmp_path, n_superset=3, n_full=5, per_line=10)


@pytest.mark.step_build
def test_quota_shortfall_passes_at_min_quota_full(tmp_path: Path) -> None:
    """n_superset (5) == min(quota, n_full) == min(5, 10): the superset reproduces the
    full-data selection exactly, so this must NOT raise."""
    _quota_shortfall_case(tmp_path, n_superset=5, n_full=10, per_line=5)
