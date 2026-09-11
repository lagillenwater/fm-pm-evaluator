"""Answer matrices from the scanned Tahoe rows, and which pairs are scoreable.

Task 3 of rung 1 builds the measured answer every prediction is scored against: for each
(line, drug) in the grid, the mean ``log2FoldChange`` over its plates at 5 uM, per gene, plus
which genes were called differentially expressed (``padj`` below 0.05) on at least one plate.
This module holds the pure assembly and scoring; ``scripts/heldout_answers.py`` does the
DuckDB scan that produces the rows these functions consume.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmharness.heldout.grid import Grid

#: padj threshold below which a gene is called responding on a plate (design.md section 2).
RESPONDER_ALPHA = 0.05


@dataclass(frozen=True)
class Answers:
    """The grid's answer: mean ``log2FoldChange`` and responder calls, one array per pair.

    ``lines`` and ``drugs`` are the grid's own tuples, in grid order. ``genes`` is sorted: the
    union of genes with a finite ``mean_lfc`` in any grid pair. ``delta`` is float32
    ``[L, D, G]``, NaN where a (line, drug, gene) triple was never tested. ``responding`` is
    bool ``[L, D, G]``: True where ``delta`` is finite AND the pair's minimum adjusted p-value
    for that gene, over its plates, is below ``RESPONDER_ALPHA``.
    """

    lines: tuple[str, ...]
    drugs: tuple[str, ...]
    genes: tuple[str, ...]
    delta: np.ndarray
    responding: np.ndarray


def assemble_answers(rows: pd.DataFrame, grid: Grid) -> Answers:
    """Scatter the scanned (line, drug, gene) rows into the grid's dense answer arrays.

    ``rows`` has (at least) columns ``line, drug, gene, mean_lfc, min_padj`` -- one row per
    (line, drug, gene) grouping the scan produced. Rows whose line or drug is outside the grid
    are ignored (the scan already restricts to the grid; this holds even if it did not). Built
    entirely by scattering integer codes into preallocated arrays: no per-row Python loop, no
    ``pivot_table``.
    """
    line_index = {line: i for i, line in enumerate(grid.lines)}
    drug_index = {drug: i for i, drug in enumerate(grid.drugs)}

    line_s = rows["line"].astype(str)
    drug_s = rows["drug"].astype(str)
    in_grid = line_s.isin(line_index) & drug_s.isin(drug_index)
    grid_rows = rows.loc[in_grid]

    mean_lfc_all = grid_rows["mean_lfc"].to_numpy(dtype=np.float64)
    finite_all = np.isfinite(mean_lfc_all)
    genes = tuple(sorted(set(grid_rows.loc[finite_all, "gene"].astype(str))))
    gene_index = {gene: i for i, gene in enumerate(genes)}

    gene_codes_all = grid_rows["gene"].astype(str).map(gene_index)
    keep = gene_codes_all.notna().to_numpy()
    sub = grid_rows.loc[keep]
    gene_codes = gene_codes_all.loc[keep].to_numpy(dtype=np.int64)
    line_codes = sub["line"].astype(str).map(line_index).to_numpy(dtype=np.int64)
    drug_codes = sub["drug"].astype(str).map(drug_index).to_numpy(dtype=np.int64)
    sub_mean_lfc = sub["mean_lfc"].to_numpy(dtype=np.float64)
    sub_min_padj = sub["min_padj"].to_numpy(dtype=np.float64)

    n_lines, n_drugs, n_genes = len(grid.lines), len(grid.drugs), len(genes)
    delta = np.full((n_lines, n_drugs, n_genes), np.nan, dtype=np.float32)
    delta[line_codes, drug_codes, gene_codes] = sub_mean_lfc.astype(np.float32)

    sub_finite = np.isfinite(sub_mean_lfc)
    is_responding = sub_finite & np.isfinite(sub_min_padj) & (sub_min_padj < RESPONDER_ALPHA)
    responding = np.zeros((n_lines, n_drugs, n_genes), dtype=bool)
    responding[line_codes[is_responding], drug_codes[is_responding], gene_codes[is_responding]] = (
        True
    )

    return Answers(
        lines=grid.lines, drugs=grid.drugs, genes=genes, delta=delta, responding=responding
    )


def scoreable(
    answers: Answers,
    excluded_pairs: tuple[tuple[str, str], ...],
    min_genes: int = 50,
) -> dict[str, np.ndarray]:
    """Which (line, drug) pairs qualify for scoring, per gene set.

    A pair qualifies on a gene set when at least ``min_genes`` genes qualify for it
    (responding genes, or every finite-delta gene) and the pair is not one of
    ``excluded_pairs``. Returns bool ``[L, D]`` arrays keyed ``"responding"`` and ``"all"``.
    """
    n_lines, n_drugs = len(answers.lines), len(answers.drugs)
    line_index = {line: i for i, line in enumerate(answers.lines)}
    drug_index = {drug: i for i, drug in enumerate(answers.drugs)}

    excluded_mask = np.zeros((n_lines, n_drugs), dtype=bool)
    pairs_in_grid = [
        (line_index[line], drug_index[drug])
        for line, drug in excluded_pairs
        if line in line_index and drug in drug_index
    ]
    if pairs_in_grid:
        rows_idx, cols_idx = zip(*pairs_in_grid, strict=True)
        excluded_mask[np.array(rows_idx), np.array(cols_idx)] = True

    n_responding = answers.responding.sum(axis=2)
    n_all = np.isfinite(answers.delta).sum(axis=2)
    return {
        "responding": (n_responding >= min_genes) & ~excluded_mask,
        "all": (n_all >= min_genes) & ~excluded_mask,
    }


__all__ = ["RESPONDER_ALPHA", "Answers", "assemble_answers", "scoreable"]
