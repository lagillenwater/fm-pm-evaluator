"""Line descriptions: standardizing, PCA, NMF, random stand-ins, kernel, identity match.

Design.md section 4: every non-Stack line description is built from the same up-to-1,000
DMSO cells per line -- expression (log2(CPM+1) over the panel, standardized across the 50
lines), PCA and NMF summaries of that expression, and a random stand-in of the same width per
description, drawn once with a fixed seed. This module holds the pure, testable pieces:

* ``standardize`` / ``standardize_stats`` / ``standardize_apply`` -- column standardization
  across lines, and the split between computing and applying stats so a half can be
  standardized using the FULL data's column means and SDs (never its own).
* ``pca_components`` -- SVD of the standardized expression matrix, sign-fixed per component.
* ``nmf_components`` / ``nmf_project`` -- non-negative matrix factorization of raw
  log2(CPM+1) expression, and projecting new rows onto a fixed set of gene programs.
* ``random_stand_in`` -- the random description every real one is compared against.
* ``linear_kernel`` -- a normalized line-by-line similarity kernel (mean diagonal 1).
* ``identity_match`` / ``identity_correlations`` -- the build step's known-answer control: a
  description built from half a line's cells should best match that line's other half.
* ``shuffled_identity_null`` -- the same control's negative: with cells' line labels
  shuffled, matches should fall to chance.

``scripts/heldout_descriptions.py`` is the only thing that calls these in anger; this module
is pure numpy/scipy/sklearn so it can be exercised without any cache on disk.
"""

# pandas and scipy ship no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *their* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF, non_negative_factorization

from fmharness.heldout.cells import pseudobulk_log_cpm

#: The one seed per random stand-in, so every task reusing a stand-in draws the same numbers
#: (task 5 declares them; task 6 and the fit stage reuse them, per the brief).
RANDOM_SEEDS: dict[str, int] = {
    "expression": 101,
    "pca": 102,
    "nmf": 103,
    "stack_base": 104,
    "stack_cytokine": 105,
    "stack_drug": 106,
}

#: The one NMF seed (design.md section 4 / the brief's resolution): every k uses the same seed.
NMF_SEED = 0


def standardize_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each column's mean and SD (``ddof=0``) across rows (lines)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply column stats computed elsewhere (e.g. from the full data) to ``x``.

    A zero-``std`` column becomes all zeros rather than dividing by zero -- a gene (or
    component) with no variance across the lines that produced ``mean``/``std`` carries no
    signal to standardize.
    """
    out = np.zeros_like(x, dtype=np.float64)
    nonzero = std > 0
    out[:, nonzero] = (x[:, nonzero] - mean[nonzero]) / std[nonzero]
    return out


def standardize(x: np.ndarray) -> np.ndarray:
    """Each column centred to mean 0 and scaled to SD 1 (``ddof=0``) across rows (lines).

    A zero-variance column becomes 0 rather than dividing by zero.
    """
    mean, std = standardize_stats(x)
    return standardize_apply(x, mean, std)


def pca_components(x_std: np.ndarray, k_max: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """PCA of a standardized lines-by-genes matrix.

    Returns ``(scores, loadings)``: an SVD of ``x_std`` gives ``U, S, Vt``; ``scores = U[:,
    :k_max] * S[:k_max]`` (``L x k_max``) and ``loadings = Vt[:k_max]`` (``k_max x G``). Each
    component's sign is flipped, if needed, so its loading's largest-magnitude entry is
    positive -- the same flip applied to its score column, so ``scores`` and ``loadings``
    stay consistent with each other.
    """
    u, s, vt = np.linalg.svd(x_std, full_matrices=False)
    k = min(k_max, u.shape[1], vt.shape[0])
    scores = u[:, :k] * s[:k]
    loadings = vt[:k].copy()
    scores = scores.copy()

    largest_idx = np.argmax(np.abs(loadings), axis=1)
    sign = np.sign(loadings[np.arange(k), largest_idx])
    sign[sign == 0] = 1.0
    loadings = loadings * sign[:, None]
    scores = scores * sign[None, :]
    return scores, loadings


def pca_project(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray, loadings: np.ndarray
) -> np.ndarray:
    """Project new rows (e.g. a half's expression) onto ``loadings``, standardizing ``x``
    first with the FULL data's column ``mean``/``std`` (never the half's own)."""
    x_std = standardize_apply(x, mean, std)
    return x_std @ loadings.T


def nmf_components(
    x_nonneg: np.ndarray, k: int, seed: int = NMF_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative matrix factorization of a non-negative (not standardized) matrix.

    ``sklearn.decomposition.NMF(n_components=k, init="nndsvda", random_state=seed,
    max_iter=2000)``. Returns ``(W, H)``.
    """
    model = NMF(n_components=cast(Any, k), init="nndsvda", random_state=seed, max_iter=2000)
    w = model.fit_transform(x_nonneg)
    h = cast(Any, model.components_)
    return np.asarray(w), np.asarray(h)


def nmf_project(x_nonneg: np.ndarray, h: np.ndarray, seed: int = NMF_SEED) -> np.ndarray:
    """Project new rows onto a fixed set of gene programs ``h`` (``H`` held constant)."""
    n_components = h.shape[0]
    w, _, _ = non_negative_factorization(
        x_nonneg,
        H=h,
        n_components=n_components,
        update_H=False,
        init="custom",
        random_state=seed,
        max_iter=2000,
    )
    return np.asarray(w)


def random_stand_in(n_lines: int, width: int, seed: int) -> np.ndarray:
    """Standard normal ``float64`` random numbers, the same size as a real description."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_lines, width))


def linear_kernel(z: np.ndarray) -> np.ndarray:
    """A normalized linear kernel: standardize ``z``, then ``K = Z Z^T``, scaled so
    ``mean(diag(K)) == 1``.

    Raises ``ValueError`` if the resulting kernel is all zero (every column of ``z`` had zero
    variance, so standardizing left nothing to compute a kernel from).
    """
    z_std = standardize(z)
    k = z_std @ z_std.T
    diag_mean = float(np.diag(k).mean())
    if diag_mean == 0.0:
        raise ValueError("linear_kernel: kernel is all zero (every input column has zero variance)")
    return k / diag_mean


def _row_normalize(x: np.ndarray) -> np.ndarray:
    """Each row centred to mean 0 and scaled to unit L2 norm; an all-constant row (zero norm)
    stays zero, so it correlates at 0 with everything rather than raising."""
    mean = x.mean(axis=1, keepdims=True)
    centred = x - mean
    norm = np.sqrt((centred**2).sum(axis=1, keepdims=True))
    out = np.zeros_like(centred)
    nonzero = norm[:, 0] > 0
    out[nonzero] = centred[nonzero] / norm[nonzero]
    return out


def identity_correlations(half_a: np.ndarray, half_b: np.ndarray) -> np.ndarray:
    """Pearson correlation between every row of ``half_a`` and every row of ``half_b`` (each
    row centred and normalized across its own dimensions first): an ``(L, L)`` array."""
    a = _row_normalize(half_a)
    b = _row_normalize(half_b)
    return a @ b.T


def identity_match(half_a: np.ndarray, half_b: np.ndarray) -> float:
    """The share of lines (rows) whose most-correlated row in ``half_b`` is their own (same
    row index in ``half_a``)."""
    corr = identity_correlations(half_a, half_b)
    best = np.argmax(corr, axis=1)
    return float(np.mean(best == np.arange(corr.shape[0])))


def _shuffled_halves(
    counts: sparse.csr_matrix,
    lines_arr: np.ndarray,
    halves_arr: np.ndarray,
    categories: list[Any],
    shuffled_lines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudobulk log2(CPM+1), per (shuffled line, half), reindexed onto the FULL fixed
    ``categories`` list (vectorized via ``DataFrame.reindex``, no per-category loop) -- a line
    a shuffle happened to leave with no cells in a half gets an all-zero row rather than being
    dropped, so both halves' matrices always share the same shape and row order."""
    mats: list[np.ndarray] = []
    for half in (0, 1):
        mask = halves_arr == half
        labels, pseudobulk = pseudobulk_log_cpm(counts[mask], shuffled_lines[mask])
        frame = pd.DataFrame(pseudobulk, index=pd.Index(labels)).reindex(categories, fill_value=0.0)
        mats.append(frame.to_numpy(dtype=np.float64))
    return mats[0], mats[1]


def shuffled_identity_null(
    counts: sparse.csr_matrix,
    lines: Sequence[Any] | np.ndarray,
    halves: Sequence[int] | np.ndarray,
    describe: Callable[[np.ndarray], np.ndarray],
    n_shuffles: int,
    seed: int,
) -> np.ndarray:
    """Identity match under ``n_shuffles`` permutations of the cells' line labels.

    Each shuffle draws one permutation (seeded ``np.random.default_rng(seed)``), reassigns
    every cell's line label according to it (halves stay fixed to their real cells),
    pseudobulks log2(CPM+1) per (shuffled line, half) into two ``L x G`` matrices over the
    FULL fixed set of lines, applies ``describe`` to each, and returns ``identity_match`` per
    shuffle. Vectorized within a shuffle (sparse group sums via ``pseudobulk_log_cpm``); the
    loop is over shuffles only.
    """
    lines_arr = np.asarray(lines)
    halves_arr = np.asarray(halves)
    categories = sorted(set(lines_arr.tolist()))
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffles, dtype=np.float64)
    n_cells = lines_arr.shape[0]
    for s in range(n_shuffles):
        shuffled_lines = lines_arr[rng.permutation(n_cells)]
        a, b = _shuffled_halves(counts, lines_arr, halves_arr, categories, shuffled_lines)
        out[s] = identity_match(describe(a), describe(b))
    return out


def shuffled_identity_correlations(
    counts: sparse.csr_matrix,
    lines: Sequence[Any] | np.ndarray,
    halves: Sequence[int] | np.ndarray,
    describe: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> tuple[list[Any], np.ndarray]:
    """The identity-correlation grid for exactly the FIRST shuffle ``shuffled_identity_null``
    would draw with the same ``seed`` -- the negative-control grid for the build figure.
    Returns ``(categories, corr)`` in the same fixed line order the null uses."""
    lines_arr = np.asarray(lines)
    halves_arr = np.asarray(halves)
    categories = sorted(set(lines_arr.tolist()))
    rng = np.random.default_rng(seed)
    n_cells = lines_arr.shape[0]
    shuffled_lines = lines_arr[rng.permutation(n_cells)]
    a, b = _shuffled_halves(counts, lines_arr, halves_arr, categories, shuffled_lines)
    return categories, identity_correlations(describe(a), describe(b))


__all__ = [
    "NMF_SEED",
    "RANDOM_SEEDS",
    "identity_correlations",
    "identity_match",
    "linear_kernel",
    "nmf_components",
    "nmf_project",
    "pca_components",
    "pca_project",
    "random_stand_in",
    "shuffled_identity_correlations",
    "shuffled_identity_null",
    "standardize",
    "standardize_apply",
    "standardize_stats",
]
