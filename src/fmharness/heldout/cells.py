"""DMSO cell keys, plate-balanced selection, halves, and pseudobulk log CPM (rung 1 task 4).

Design.md section 4: every line description is built from the same cells -- up to 1,000
untreated (DMSO-only) cells per line, spread evenly over its plates, picked with a fixed
seed. This module holds the pure, testable pieces of that rule:

* ``cell_keys`` -- a deterministic per-cell key from its position in the Hugging Face parquet
  shards (splitmix64 of a packed 64-bit position), so the same cell gets the same key
  whatever order shards are scanned in.
* ``superset_mask`` -- a cheap first filter (a quarter of all keys) applied while scanning,
  so the heavy per-cell decode only ever touches a bounded slice of the ~100M-cell corpus.
* ``select_cells`` -- the plate-balanced, capped selection out of that superset.
* ``half_of`` -- an independent split-half bit for measuring internal reliability of the
  descriptions built from these cells.
* ``pseudobulk_log_cpm`` -- raw counts summed per group, log2(counts per million + 1).

The Hugging Face streaming itself (network IO, the Stack gene panel match) lives in
``scripts/heldout_dmso_cells.py``, which is the only thing that calls these functions in
anger; this module is pure numpy/pandas/scipy so it can be exercised without a network.
"""

# pandas and scipy ship no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *their* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

#: The Hugging Face key of Tahoe's vehicle-control drug (task 4's cells are all this drug).
DMSO_DRUG = "DMSO_TF"

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 over a ``uint64`` array. Overflow is the point (fixed-width
    wraparound), so it runs under ``np.errstate(over="ignore")``."""
    with np.errstate(over="ignore"):
        x = x + _GOLDEN
        x = (x ^ (x >> np.uint64(30))) * _MIX1
        x = (x ^ (x >> np.uint64(27))) * _MIX2
        x = x ^ (x >> np.uint64(31))
    return x


def cell_keys(
    shard_index: int | Sequence[int] | np.ndarray,
    row_group: int | Sequence[int] | np.ndarray,
    rows: Sequence[int] | np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """A ``uint64`` key per cell, from splitmix64 of its packed shard position, XOR-ed with
    ``seed`` before mixing.

    ``shard_index`` and ``row_group`` may be scalars (broadcast over ``rows``) or arrays the
    same length as ``rows``. The packing is ``(shard_index << 40) | (row_group << 20) | row``,
    so the key depends only on where a cell sits in the corpus -- never on the order shards
    happen to be read in, and never on the seed until the final mix. Fully vectorized (numpy
    ``uint64`` arithmetic); no per-cell Python loop.
    """
    rows_arr = np.asarray(rows, dtype=np.uint64)
    shard_arr = np.asarray(shard_index, dtype=np.uint64)
    rg_arr = np.asarray(row_group, dtype=np.uint64)
    with np.errstate(over="ignore"):
        packed = (shard_arr << np.uint64(40)) | (rg_arr << np.uint64(20)) | rows_arr
        packed = packed ^ np.uint64(seed)
    return _splitmix64(packed)


def superset_mask(keys: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    """Keep keys below ``floor(fraction * 2**64)``, a uniform ``fraction``-sized subset.

    The threshold is computed as a Python int via ``math.floor``, not by casting to a numpy
    float -- multiplying a float by an exact power of two (``2**64``) only shifts its
    exponent, so no precision beyond the float's own is lost, and ``math.floor`` on that
    float converts its exact dyadic value to an int rather than rounding through a smaller
    integer type. The comparison itself runs in ``uint64``, never float, so keys near the
    threshold are never misclassified by float rounding.
    """
    threshold = min(math.floor(fraction * (2**64)), 2**64 - 1)
    threshold_u64 = np.uint64(threshold)
    keys_u64 = np.asarray(keys, dtype=np.uint64)
    return keys_u64 < threshold_u64


def half_of(keys: np.ndarray) -> np.ndarray:
    """Bit 1 of each key as ``int8`` (0 or 1). Independent of ``superset_mask``, which compares
    a key's full value (in effect, its high-order bits) against a threshold -- an ordering
    comparison, not a test of any single low bit -- so halves do not depend on the fraction
    used to build the superset, or on which cells were later selected out of it."""
    bit = (np.asarray(keys, dtype=np.uint64) >> np.uint64(1)) & np.uint64(1)
    return bit.astype(np.int8)


def select_cells(
    meta: pd.DataFrame,
    plates_per_line: Mapping[str, int],
    per_line: int = 1000,
) -> np.ndarray:
    """Plate-balanced, capped selection: the quota's smallest keys per (line, plate), then
    the ``per_line`` smallest of those per line.

    ``meta`` has columns ``line``, ``plate``, ``key`` (one row per candidate cell, already
    inside the superset). ``plates_per_line`` counts plates that had ANY DMSO cell for that
    line in the full scan (not just the superset) -- the quota per line is
    ``ceil(per_line / plates_per_line[line])``. A plate with fewer cells than its quota
    contributes all of them (no error, no padding). Returns a bool array aligned with
    ``meta``'s rows (not its index).

    Vectorized via ``groupby(...).rank()`` -- no per-row Python loop.
    """
    df = meta.reset_index(drop=True)
    lines = cast(Any, df)["line"]
    missing = sorted(set(lines.unique().tolist()) - set(plates_per_line))
    if missing:
        raise ValueError(f"plates_per_line is missing lines present in meta: {missing}")

    quota = lines.map(lambda ln: math.ceil(per_line / plates_per_line[ln])).to_numpy()

    plate_rank = cast(Any, df).groupby(["line", "plate"])["key"].rank(method="first").to_numpy()
    keep_plate = plate_rank <= quota

    # For the per-line cap, rank only among plate-kept rows: give every other row a sentinel
    # key above any real key (superset keys are always < 2**64 - 1), so it never displaces a
    # genuine candidate and never survives the <= per_line cutoff itself.
    sentinel = np.uint64(2**64 - 1)
    keys_arr = df["key"].to_numpy(dtype=np.uint64)
    ranking_key = np.where(keep_plate, keys_arr, sentinel)
    ranked = pd.DataFrame({"line": df["line"].to_numpy(), "ranking_key": ranking_key})
    line_rank = cast(Any, ranked).groupby("line")["ranking_key"].rank(method="first").to_numpy()
    keep_line = line_rank <= per_line

    return keep_plate & keep_line


def pseudobulk_log_cpm(
    counts: sparse.csr_matrix, groups: Sequence[Any] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum raw counts per group, scale to counts per million within each group, and return
    ``log2(CPM + 1)``.

    ``groups`` labels each row of ``counts`` (any hashable, orderable value -- a line name, or
    a (line, half) tuple); the returned labels are sorted. Group sums are computed as one
    sparse matrix product (an indicator matrix times ``counts``), not a per-group loop.
    """
    groups_seq = list(groups)
    # Built element-by-element rather than via np.asarray(groups_seq, dtype=object): numpy
    # collapses a list of equal-length tuples into a 2-D array even under dtype=object,
    # which would silently break factorize below. One assignment loop over cells (not a
    # per-group or nested loop) sidesteps that coercion.
    groups_arr = np.empty(len(groups_seq), dtype=object)
    for i, v in enumerate(groups_seq):
        groups_arr[i] = v
    codes, labels = pd.factorize(groups_arr, sort=True)
    n_groups = len(labels)
    n_rows = cast(Any, counts).shape[0]
    indicator = sparse.csr_matrix(
        (np.ones(n_rows, dtype=np.float64), (codes, np.arange(n_rows))),
        shape=(n_groups, n_rows),
    )
    product = cast(Any, indicator) @ cast(Any, counts)
    sums = np.asarray(product.todense(), dtype=np.float64)
    totals = sums.sum(axis=1, keepdims=True)
    cpm = np.divide(sums, totals, out=np.zeros_like(sums), where=totals > 0) * 1_000_000.0
    log2_cpm = np.log2(cpm + 1.0)
    return np.asarray(labels), log2_cpm


__all__ = [
    "DMSO_DRUG",
    "cell_keys",
    "half_of",
    "pseudobulk_log_cpm",
    "select_cells",
    "superset_mask",
]
