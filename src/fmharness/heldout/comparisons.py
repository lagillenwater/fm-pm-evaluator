"""Rung 1's head-to-head comparisons: redraws of the held-out units, and what is read off them
(design.md section 7).

A comparison is the mean, over the pairs both models scored, of one model's score minus the
other's. Its uncertainty comes from redrawing the held-out units with repeats -- the 50 lines
when a line is hidden, the 107 drugs when a drug is hidden -- and recomputing the mean over the
redrawn units' pairs, 2,000 times. A unit drawn k times contributes each of its pairs k times, so
each draw is ``Σ_u w_u S_u / Σ_u w_u C_u``, with ``w`` the multinomial count of each unit in the
draw, ``S_u`` the sum of the unit's per-pair differences and ``C_u`` its number of pairs. From
the draws:

* the confidence interval is their 2.5th and 97.5th percentiles;
* the two-sided p-value is twice the smaller share of draws on either side of zero, each share
  counted with one added draw so that p is never 0;
* the minimum detectable effect is 2.8 times their standard deviation (the effect a two-sided
  5% test detects 80% of the time: 1.96 + 0.84 standard errors);
* p-values are adjusted by Holm's method within each test (leave one line out, leave one drug
  out).

Pairs share lines and drugs, so the comparison is also redrawn over lines and drugs together
(``two_way_estimates``); the ratio of the two redraw variances is the design effect.

The draws run in blocks, one job each: ``N_BLOCKS`` blocks of ``DRAWS_PER_BLOCK`` draws, block
``b`` seeded ``base_seed + b``, concatenated in block order (``in_blocks``). Every draw comes
from its block's seed alone, so the blocks reproduce the same draws whatever order or process
they run in (invariant 4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np

#: Redraws per comparison, and how they are split into seeded blocks.
N_DRAWS = 2000
N_BLOCKS = 8
DRAWS_PER_BLOCK = 250

#: The minimum detectable effect in redraw standard deviations: z(0.975) + z(0.80), rounded.
MDE_FACTOR = 2.8


class RedrawSummary(TypedDict):
    """What a comparison (or a model's mean score) reports from its redraws.

    ``estimate`` -- the mean over the pairs as scored (not redrawn).
    ``ci_lo``, ``ci_hi`` -- the 2.5th and 97.5th percentiles of the redraws.
    ``p`` -- two-sided, from ``B`` redraws:
    ``min(1, 2 min((1 + #{draws <= 0}) / (1 + B), (1 + #{draws >= 0}) / (1 + B)))``.
    ``sd`` -- the redraws' standard deviation (``ddof=1``); ``mde`` is ``MDE_FACTOR * sd``.
    ``n_draws`` -- ``B``, the finite redraws used; ``n_dropped`` -- redraws that drew no pair
    (NaN), left out.
    """

    estimate: float
    ci_lo: float
    ci_hi: float
    p: float
    sd: float
    mde: float
    n_draws: int
    n_dropped: int


def _check_units(values: np.ndarray, index: np.ndarray, n_units: int, kind: str) -> None:
    if values.ndim != 1 or index.shape != values.shape:
        raise ValueError(
            f"need one {kind} index per pair: got {index.shape} for values shaped {values.shape}"
        )
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError(f"{kind} indices must be integers; got {index.dtype}")
    if index.size and (index.min() < 0 or index.max() >= n_units):
        raise ValueError(f"a {kind} index is outside 0..{n_units - 1}")


def _multinomial_counts(rng: np.random.Generator, n_units: int, n_draws: int) -> np.ndarray:
    """``[n_draws, n_units]``: how many times each unit is drawn in each redraw of ``n_units``
    units with repeats, equal chances."""
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_draws)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator``, NaN where the denominator is 0 (a draw with no pairs)."""
    out = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def redraw_estimates(
    diffs: np.ndarray, unit_index: np.ndarray, n_units: int, n_draws: int, seed: int
) -> np.ndarray:
    """``n_draws`` redraws of the mean of ``diffs`` over ``n_units`` held-out units.

    ``diffs`` is ``[P]`` per-pair values (a score difference, or a score); ``unit_index`` ``[P]``
    the unit (line or drug index, ``0..n_units-1``) each pair belongs to. With
    ``rng = np.random.default_rng(seed)``, the counts are
    ``W = rng.multinomial(n_units, [1/n_units] * n_units, size=n_draws)`` and draw ``b`` is
    ``(W[b] @ S) / (W[b] @ C)``: per-unit sums ``S`` and pair counts ``C``, from ``np.bincount``.
    A draw whose units hold no pairs is NaN.
    """
    values = np.asarray(diffs, dtype=np.float64)
    units = np.asarray(unit_index)
    _check_units(values, units, n_units, "unit")
    counts_drawn = _multinomial_counts(np.random.default_rng(seed), n_units, n_draws)
    sums = np.bincount(units, weights=values, minlength=n_units)
    pairs = np.bincount(units, minlength=n_units)
    return _ratio(counts_drawn @ sums, counts_drawn @ pairs)


def two_way_estimates(
    diffs: np.ndarray,
    line_index: np.ndarray,
    drug_index: np.ndarray,
    n_lines: int,
    n_drugs: int,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """``n_draws`` redraws of the mean of ``diffs`` over lines and drugs redrawn together.

    With ``rng = np.random.default_rng(seed)``, line counts ``Wl = rng.multinomial(n_lines, ...,
    size=n_draws)`` are drawn first, then drug counts ``Wd`` likewise; pair ``(l, d)`` enters
    draw ``b`` with weight ``Wl[b, l] Wd[b, d]``. With per-(line, drug) sums ``S`` and pair counts
    ``C``, draw ``b`` is ``(Wl[b] S Wd[b]ᵀ) / (Wl[b] C Wd[b]ᵀ)``; NaN when it weighs no pair.
    """
    values = np.asarray(diffs, dtype=np.float64)
    lines = np.asarray(line_index)
    drugs = np.asarray(drug_index)
    _check_units(values, lines, n_lines, "line")
    _check_units(values, drugs, n_drugs, "drug")
    rng = np.random.default_rng(seed)
    line_counts = _multinomial_counts(rng, n_lines, n_draws)
    drug_counts = _multinomial_counts(rng, n_drugs, n_draws)
    cell = lines * n_drugs + drugs
    sums = np.bincount(cell, weights=values, minlength=n_lines * n_drugs)
    pairs = np.bincount(cell, minlength=n_lines * n_drugs)
    numerator = ((line_counts @ sums.reshape(n_lines, n_drugs)) * drug_counts).sum(axis=1)
    denominator = ((line_counts @ pairs.reshape(n_lines, n_drugs)) * drug_counts).sum(axis=1)
    return _ratio(numerator, denominator)


def redraw_block_seed(base_seed: int, block: int) -> int:
    """The seed of redraw block ``block``: ``base_seed + block``."""
    return base_seed + block


def concatenate_blocks(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Redraw blocks joined into one array, in the order given (block order)."""
    return np.concatenate(list(blocks))


def in_blocks(
    draw: Callable[[int, int], np.ndarray], n_blocks: int, draws_per_block: int, base_seed: int
) -> np.ndarray:
    """Every block's redraws in one process: ``draw(draws_per_block, seed)`` for each block's
    seed (``redraw_block_seed``), concatenated in block order -- the same draws as the blocks
    run one per job."""
    return concatenate_blocks(
        [draw(draws_per_block, redraw_block_seed(base_seed, block)) for block in range(n_blocks)]
    )


def summarize_redraws(estimate: float, draws: np.ndarray) -> RedrawSummary:
    """The interval, p-value, standard deviation and MDE of a set of redraws (``RedrawSummary``).

    NaN draws (redraws that drew no pair) are left out and counted in ``n_dropped``; at least 2
    finite draws are needed.
    """
    values = np.asarray(draws, dtype=np.float64)
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n < 2:
        raise ValueError(f"summarizing redraws needs at least 2 finite draws; got {n}")
    ci_lo, ci_hi = np.percentile(finite, [2.5, 97.5])
    at_or_below = (1 + np.count_nonzero(finite <= 0)) / (1 + n)
    at_or_above = (1 + np.count_nonzero(finite >= 0)) / (1 + n)
    sd = float(finite.std(ddof=1))
    return RedrawSummary(
        estimate=float(estimate),
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
        p=float(min(1.0, 2.0 * min(at_or_below, at_or_above))),
        sd=sd,
        mde=MDE_FACTOR * sd,
        n_draws=n,
        n_dropped=int(values.size - n),
    )


def mean_score_ci(
    scores: np.ndarray, unit_index: np.ndarray, n_units: int, n_draws: int, seed: int
) -> RedrawSummary:
    """A model's mean score over its pairs, with the interval from redrawing its held-out units
    (``redraw_estimates``, then ``summarize_redraws``)."""
    values = np.asarray(scores, dtype=np.float64)
    draws = redraw_estimates(values, unit_index, n_units, n_draws, seed)
    return summarize_redraws(float(values.mean()), draws)


def design_effect(two_way: np.ndarray, one_way: np.ndarray) -> float:
    """How much redrawing lines and drugs together widens a comparison: the variance (``ddof=1``)
    of its two-way redraws over the variance of its one-way redraws, NaN draws left out."""
    both = np.asarray(two_way, dtype=np.float64)
    single = np.asarray(one_way, dtype=np.float64)
    variance_both = float(np.var(both[np.isfinite(both)], ddof=1))
    variance_single = float(np.var(single[np.isfinite(single)], ddof=1))
    return variance_both / variance_single


def holm(p: np.ndarray) -> np.ndarray:
    """Holm's step-down adjusted p-values, in the input order.

    Sorted ascending, the i-th smallest of m p-values (i from 0) is multiplied by ``m - i``; a
    running maximum keeps the adjusted values in the same order as the raw ones, and each is
    capped at 1. Ties keep their input order.
    """
    values = np.asarray(p, dtype=np.float64)
    if values.ndim != 1 or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("p must be a 1-D array of p-values between 0 and 1")
    m = values.size
    order = np.argsort(values, kind="stable")
    stepped = np.maximum.accumulate((m - np.arange(m)) * values[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, stepped)
    return adjusted


__all__ = [
    "DRAWS_PER_BLOCK",
    "MDE_FACTOR",
    "N_BLOCKS",
    "N_DRAWS",
    "RedrawSummary",
    "concatenate_blocks",
    "design_effect",
    "holm",
    "in_blocks",
    "mean_score_ci",
    "redraw_block_seed",
    "redraw_estimates",
    "summarize_redraws",
    "two_way_estimates",
]
