"""Rung 1's per-pair score, and the score as a fraction of its ceiling (design.md section 6).

A model's prediction for a (line, drug) pair is scored by the Pearson correlation, across genes,
between the predicted and the measured change, on two gene sets:

* ``responding`` -- the pair's genes with ``padj < 0.05`` on at least one plate;
* ``all`` -- every gene with a finite answer.

A pair is scored on a gene set only when the scoreable-pair masks keep it
(``answers.scoreable``: at least ``MIN_GENES`` qualifying genes, and not an excluded pair), so
every model is scored on the same pairs (invariant 2). The ceiling a prediction can reach is
``√SB``, the square root of the answer's full-data reliability: a prediction meets measurement
noise once, where two measurements meet it twice.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fmharness.heldout.answers import Answers
from fmharness.statistics import masked_rowwise_pearson

#: The two gene sets every pair is scored on, in the order ``score_pairs`` writes them.
GENE_SETS: tuple[str, ...] = ("responding", "all")

#: A pair is scored on a gene set only with at least this many genes to correlate.
MIN_GENES = 50

#: The columns of ``score_pairs``' table, in order.
SCORE_COLUMNS: tuple[str, ...] = ("line", "drug", "gene_set", "r", "n_genes")


def _check_pairs(
    prediction: np.ndarray, lines: np.ndarray, drugs: np.ndarray, answers: Answers
) -> None:
    n_lines, n_drugs, n_genes = answers.delta.shape
    if prediction.ndim != 2 or prediction.shape[1] != n_genes:
        raise ValueError(
            f"prediction must be [pairs, genes] with the answers' {n_genes} genes; "
            f"got shape {prediction.shape}"
        )
    for name, index in (("line", lines), ("drug", drugs)):
        if not np.issubdtype(index.dtype, np.integer):
            raise ValueError(f"{name} indices must be integers; got {index.dtype}")
    if lines.shape != (prediction.shape[0],) or drugs.shape != (prediction.shape[0],):
        raise ValueError(
            f"need one line and one drug index per predicted pair ({prediction.shape[0]}); "
            f"got {lines.shape} and {drugs.shape}"
        )
    if ((lines < 0) | (lines >= n_lines)).any() or ((drugs < 0) | (drugs >= n_drugs)).any():
        raise ValueError(f"a line or drug index is outside the {n_lines} x {n_drugs} grid")


def _check_masks(masks: Mapping[str, np.ndarray], answers: Answers) -> None:
    missing = [gene_set for gene_set in GENE_SETS if gene_set not in masks]
    if missing:
        raise ValueError(f"masks has no entry for gene set(s) {missing}")
    grid = answers.delta.shape[:2]
    misshapen = {g: np.shape(masks[g]) for g in GENE_SETS if np.shape(masks[g]) != grid}
    if misshapen:
        raise ValueError(f"masks must be {list(grid)} (lines x drugs) arrays; got {misshapen}")


def score_pairs(
    pred: np.ndarray,
    line_idx: np.ndarray,
    drug_idx: np.ndarray,
    answers: Answers,
    masks: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Score predicted pairs against the answer on both gene sets.

    ``pred`` is ``[P, G]``: row ``i`` predicts pair ``(line_idx[i], drug_idx[i])`` over the
    answers' genes. A leave-one-line-out round passes one line's drugs, a leave-one-drug-out
    round one drug's lines; each pair's score does not depend on which other pairs are passed.
    ``masks`` maps each gene set to a bool ``[L, D]`` array, normally ``answers.scoreable``;
    ``ValueError`` when one is missing or misshapen.

    Returns one row per kept (pair, gene set) with columns ``SCORE_COLUMNS``: the line and drug
    names from ``answers``, the gene set, ``r`` -- Pearson correlation of prediction and answer
    over the selected genes (the pair's responding genes, or every gene with a finite answer),
    computed in float64 by ``masked_rowwise_pearson`` -- and ``n_genes``, the number of selected
    genes the correlation is taken over. Rows come gene set by gene set (``GENE_SETS`` order),
    pairs in the order given.

    The prediction must be finite at every gene selected for a kept pair (``ValueError``
    otherwise), so no model is scored on fewer genes than another on the same pair. Entries no
    kept pair selects -- untested genes, pairs the masks leave out -- are not read.

    Short pairs: a pair the mask leaves out has no row. A pair the mask keeps with fewer than
    ``MIN_GENES`` selected genes (possible only with masks other than ``scoreable``'s), or with
    zero variance on either side, gets a row with ``r`` NaN. A pair with a NaN score, from fewer
    than ``MIN_GENES`` usable genes, is dropped for every model, so all models share one scored
    population.
    """
    prediction = np.asarray(pred, dtype=np.float64)
    lines = np.asarray(line_idx)
    drugs = np.asarray(drug_idx)
    _check_pairs(prediction, lines, drugs, answers)
    _check_masks(masks, answers)

    line_names = np.asarray(answers.lines, dtype=object)
    drug_names = np.asarray(answers.drugs, dtype=object)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_COLUMNS}
    for gene_set in GENE_SETS:
        kept = np.asarray(masks[gene_set], dtype=bool)[lines, drugs]
        pair_lines, pair_drugs = lines[kept], drugs[kept]
        predicted = prediction[kept]
        measured = answers.delta[pair_lines, pair_drugs].astype(np.float64)
        if gene_set == "responding":
            select = answers.responding[pair_lines, pair_drugs] & np.isfinite(measured)
        else:
            select = np.isfinite(measured)
        unscorable = select & ~np.isfinite(predicted)
        if unscorable.any():
            first = int(np.flatnonzero(unscorable.any(axis=1))[0])
            raise ValueError(
                f"the prediction is not finite at {int(np.count_nonzero(unscorable))} "
                f"{gene_set}-gene entries selected for scoring (first pair: "
                f"{line_names[pair_lines[first]]}, {drug_names[pair_drugs[first]]}); every "
                "model must be scored on the same genes"
            )
        columns["line"].append(line_names[pair_lines])
        columns["drug"].append(drug_names[pair_drugs])
        columns["gene_set"].append(np.full(pair_lines.size, gene_set, dtype=object))
        columns["r"].append(masked_rowwise_pearson(predicted, measured, MIN_GENES, select=select))
        columns["n_genes"].append(np.count_nonzero(select, axis=1).astype(np.int64))
    return pd.DataFrame({name: np.concatenate(parts) for name, parts in columns.items()})


def fraction_of_ceiling(mean_r: float, sqrt_sb: float) -> float:
    """A mean score as a fraction of the ceiling ``√SB``: ``mean_r / sqrt_sb``."""
    if not sqrt_sb > 0:
        raise ValueError(f"the ceiling sqrt_sb must be positive; got {sqrt_sb}")
    return mean_r / sqrt_sb


__all__ = ["GENE_SETS", "MIN_GENES", "SCORE_COLUMNS", "fraction_of_ceiling", "score_pairs"]
