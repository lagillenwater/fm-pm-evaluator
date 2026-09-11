"""Tahoe-100M ingest helpers (the heavy ``datasets`` streaming lives in the CLI script).

Pure, testable pieces of the single-cell context build: reconstructing expression from
Tahoe's tokenized (``genes`` token-id + ``expressions`` value) format over the Stack gene
panel, and parsing the dose string. The streaming / IO is in
``scripts/build_tahoe_context.py`` (Alpine-only, needs the ``datasets`` package).
"""

# pandas and scipy ship no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *their* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import ast
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import sparse


def parse_dose_um(drugname_drugconc: str) -> float:
    """Parse Tahoe's dose string, e.g. ``"[('8-Hydroxyquinoline',0.05,'uM')]"`` -> 0.05 (uM)."""
    try:
        return float(ast.literal_eval(drugname_drugconc)[0][1])
    except (ValueError, SyntaxError, TypeError, IndexError):
        return float("nan")


def scatter_flat_tokens(
    tokens: np.ndarray,
    values: np.ndarray,
    lengths: np.ndarray,
    token_to_col: Mapping[int, int],
    n_cols: int,
) -> sparse.csr_matrix:
    """Scatter already-flattened, per-cell-concatenated tokenized expression into a
    (cells x n_cols) CSR over the panel.

    ``tokens`` and ``values`` are the concatenation, in cell order, of every cell's gene-token
    ids and aligned expression values (e.g. Tahoe's ``genes``/``expressions`` columns
    concatenated across cells, or an Arrow list array's flattened ``.values``); ``lengths``
    gives each cell's token count in that same order, so ``lengths.sum() == len(tokens) ==
    len(values)``. Tokens absent from ``token_to_col`` -- off-panel genes and the leading
    marker (not a panel-gene token) -- are dropped here, so callers need no special-casing for
    either. A cell none of whose tokens are on the panel gets an all-zero row, not a missing
    one. Vectorized: one repeat/map/coo construction, no per-cell loop. This is the shared
    core ``scatter_tokens`` wraps (concatenating its ragged per-cell lists first) and
    ``scripts/heldout_dmso_cells.py``'s ``scan_shard`` calls directly from Arrow's own
    offsets/values, without needing per-cell Python lists at all.
    """
    n = len(lengths)
    if n == 0:
        return sparse.csr_matrix((0, n_cols), dtype=np.float32)
    rows = np.repeat(np.arange(n), lengths)
    toks = np.asarray(tokens, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    cols = pd.Series(toks).map(token_to_col).to_numpy()
    keep = ~pd.isna(cols)
    coo = sparse.coo_matrix(
        (vals[keep], (rows[keep], cols[keep].astype(np.int64))),
        shape=(n, n_cols),
        dtype=np.float32,
    )
    return sparse.csr_matrix(coo)


def scatter_tokens(
    genes_list: list[np.ndarray],
    expr_list: list[np.ndarray],
    token_to_col: dict[int, int],
    n_cols: int,
) -> sparse.csr_matrix:
    """Scatter per-cell tokenized expression into a (cells x n_cols) CSR over the panel.

    Each Tahoe cell carries ``genes`` (gene token ids; the first is a marker token) and the
    aligned ``expressions`` values. Tokens absent from ``token_to_col`` -- off-panel genes and
    the leading marker (not a panel-gene token) -- are dropped, so the marker needs no
    special-casing. A thin wrapper: concatenates the ragged per-cell lists into flat arrays
    plus per-cell lengths, then calls ``scatter_flat_tokens`` for the actual scatter.
    """
    n = len(genes_list)
    if n == 0:
        return sparse.csr_matrix((0, n_cols), dtype=np.float32)
    lengths = np.fromiter((len(g) for g in genes_list), count=n, dtype=np.int64)
    toks = np.concatenate([np.asarray(g, dtype=np.int64) for g in genes_list])
    vals = np.concatenate([np.asarray(e, dtype=np.float64) for e in expr_list])
    return scatter_flat_tokens(toks, vals, lengths, token_to_col, n_cols)
