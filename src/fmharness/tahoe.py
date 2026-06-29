"""Tahoe-100M ingest helpers (the heavy ``datasets`` streaming lives in the CLI script).

Pure, testable pieces of the single-cell context build: reconstructing expression from
Tahoe's tokenized (``genes`` token-id + ``expressions`` value) format over the Stack gene
panel, and parsing the dose string. The streaming / IO is in
``scripts/build_tahoe_context.py`` (Alpine-only, needs the ``datasets`` package).
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from scipy import sparse


def parse_dose_um(drugname_drugconc: str) -> float:
    """Parse Tahoe's dose string, e.g. ``"[('8-Hydroxyquinoline',0.05,'uM')]"`` -> 0.05 (uM)."""
    try:
        return float(ast.literal_eval(drugname_drugconc)[0][1])
    except (ValueError, SyntaxError, TypeError, IndexError):
        return float("nan")


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
    special-casing. Vectorized via one ragged concatenation rather than a per-cell loop.
    """
    n = len(genes_list)
    if n == 0:
        return sparse.csr_matrix((0, n_cols), dtype=np.float32)
    lengths = np.fromiter((len(g) for g in genes_list), count=n, dtype=np.int64)
    rows = np.repeat(np.arange(n), lengths)
    toks = np.concatenate([np.asarray(g, dtype=np.int64) for g in genes_list])
    vals = np.concatenate([np.asarray(e, dtype=np.float64) for e in expr_list])
    cols = pd.Series(toks).map(token_to_col).to_numpy()
    keep = ~pd.isna(cols)
    coo = sparse.coo_matrix(
        (vals[keep], (rows[keep], cols[keep].astype(np.int64))),
        shape=(n, n_cols),
        dtype=np.float32,
    )
    return sparse.csr_matrix(coo)
