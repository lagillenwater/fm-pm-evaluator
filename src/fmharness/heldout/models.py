"""Rung 1's models: what each one predicts for a hidden line (task 8 adds a hidden drug).

Design.md section 5. When a line is hidden, a model is given the other lines' measured changes
for every drug and a description of every line's untreated state, and predicts the hidden line's
change for every drug and gene:

* ``drug_average`` -- each drug's mean change over the training lines. It holds everything
  about a pair that does not depend on the line, so a model beats it only by using what it was
  told about the line.
* ``ridge_lolo`` -- for each drug and gene, a ridge regression of the training lines'
  departures from the drug average on their descriptions, written in kernel form (the
  normalized linear kernel of ``descriptions.linear_kernel``); the hidden line's predicted
  departure is added back to the average. One penalty is shared by every drug and gene, chosen
  by the exact leave-one-line-out error, computed in closed form from the hat matrix's diagonal
  rather than by refitting (invariant 6).
* ``ridge_lolo_k`` -- the same, choosing the number of PCA or NMF components jointly with the
  penalty.
* ``nearest_lines_lolo`` -- the mean change over the k training lines whose descriptions
  correlate most with the hidden line's (``similarity_from_description``); k is chosen by
  leave-one-line-out error.

Conventions every function shares. ``delta0`` is the answer array ``[L, D, G]`` with untested
entries set to 0 (untested genes count as zero change in training); ``tested`` marks the entries
the screen measured. Tuning losses are mean squared leave-one-out error over tested training
entries only (invariant 7). Holding the drug average fixed at its full training value while
tuning is the documented simplification of invariant 6. No function reads the hidden line's
answers (invariant 1): training blocks are taken by indexing the training lines, and means
over lines are masked sums that never add a line outside the mask.

At full size one array of departures (49 training lines x 107 drugs x ~45k genes) is ~1.9 GB of
float64, so fits process drugs in blocks sized to a byte budget (``MAX_BYTES``). Every per-drug
quantity is computed the same way whatever the block size and combined across drugs once at
the end, so a fit's result does not depend on the budget, bit for bit.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from fmharness.heldout.descriptions import identity_correlations, linear_kernel, standardize

#: The ridge penalty grid every ridge fit searches. ``linear_kernel`` scales each kernel to a
#: mean diagonal of 1 (invariant 7), so one grid sits on the same scale for every description.
LAMBDAS: np.ndarray = np.logspace(-3, 3, 13)
LAMBDAS.flags.writeable = False

#: Default budget, in bytes, for the float64 working arrays one fit holds at once (about 2 GB).
MAX_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class Fit:
    """One model's prediction in one round, and the setting its tuning chose.

    ``prediction`` -- the hidden unit's predicted change: ``[D, G]`` when a line is hidden.
    ``lam`` -- the chosen ridge penalty; ``None`` for a model without one.
    ``k`` -- the chosen number of description components (ridge over PCA or NMF) or of nearest
    lines; ``None`` when the model chooses no such number.
    ``loss_min`` -- the tuning loss at the chosen setting: mean squared leave-one-out error over
    the training entries the screen tested.
    ``at_edge`` -- the chosen penalty is the first or last grid value, or the chosen k is the
    smallest or largest candidate: a choice the grid, not the data, may have limited.
    """

    prediction: np.ndarray
    lam: float | None
    k: int | None
    loss_min: float
    at_edge: bool


# ==============================================================================================
# Shared by both schemes: drug blocks, input checks, masked means


def drug_blocks(n_drugs: int, bytes_per_drug: int, max_bytes: int) -> list[tuple[int, int]]:
    """Consecutive ``(start, stop)`` drug ranges, each holding as many drugs as ``max_bytes``
    allows at ``bytes_per_drug`` -- at least one, so a single drug is the smallest block."""
    per_block = max(1, max_bytes // max(1, bytes_per_drug))
    return [(start, min(start + per_block, n_drugs)) for start in range(0, n_drugs, per_block)]


def _check_answers(delta0: np.ndarray, tested: np.ndarray) -> None:
    if delta0.ndim != 3:
        raise ValueError(f"delta0 must be [lines, drugs, genes]; got shape {delta0.shape}")
    if tested.shape != delta0.shape or tested.dtype != np.bool_:
        raise ValueError(
            f"tested must be a boolean array shaped like delta0 {delta0.shape}; "
            f"got {tested.dtype} {tested.shape}"
        )


def _check_square(matrix: np.ndarray, n_lines: int, name: str) -> None:
    if matrix.shape != (n_lines, n_lines):
        raise ValueError(f"{name} must be [{n_lines}, {n_lines}]; got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} has non-finite entries")


def _check_lambdas(lambdas: np.ndarray) -> np.ndarray:
    grid = np.asarray(lambdas, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not (np.isfinite(grid) & (grid > 0)).all():
        raise ValueError("lambdas must be a non-empty 1-D array of finite positive penalties")
    return grid


def _line_mean(delta0: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Mean of ``delta0`` over the distinct line indices ``lines``: ``[D, G]`` float64.

    A sum over the line axis masked to ``lines``, not a copy of the selected lines (a copy
    would be ~1.9 GB at full size): lines outside the mask never enter the sum.
    """
    chosen = np.zeros(delta0.shape[0], dtype=bool)
    chosen[lines] = True
    total = np.add.reduce(delta0, axis=0, dtype=np.float64, where=chosen[:, None, None])
    return total / float(np.count_nonzero(chosen))


def drug_average(delta0: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Each drug's mean change over the training lines ``train``: ``[D, G]``.

    Untested entries enter as the zeros ``delta0`` holds for them. ``train`` lists distinct
    integer line indices.
    """
    if delta0.ndim != 3:
        raise ValueError(f"delta0 must be [lines, drugs, genes]; got shape {delta0.shape}")
    lines = np.asarray(train)
    if (
        lines.ndim != 1
        or lines.size == 0
        or not np.issubdtype(lines.dtype, np.integer)
        or np.unique(lines).size != lines.size
    ):
        raise ValueError("train must be a non-empty 1-D array of distinct integer line indices")
    return _line_mean(delta0, lines)


# ==============================================================================================
# Leave one line out


def _training_lines(n_lines: int, held: int) -> np.ndarray:
    """Every line index except ``held``, in order."""
    held = operator.index(held)
    if not 0 <= held < n_lines:
        raise ValueError(f"held line {held} is outside 0..{n_lines - 1}")
    if n_lines < 3:
        raise ValueError("leaving one line out of training needs at least 3 lines")
    return np.flatnonzero(np.arange(n_lines) != held)


def _training_block(
    delta0: np.ndarray, lines: np.ndarray, start: int, stop: int, centre: np.ndarray | None
) -> np.ndarray:
    """The training lines' answers for drugs ``start:stop`` as a C-contiguous float64
    ``[drugs, lines, genes]`` block, less ``centre[start:stop]`` when ``centre`` is given."""
    block = np.ascontiguousarray(np.moveaxis(delta0[lines, start:stop], 1, 0), dtype=np.float64)
    if centre is not None:
        block -= centre[start:stop, None, :]
    return block


def _leave_one_out_losses(
    errors_op: np.ndarray,
    delta0: np.ndarray,
    tested: np.ndarray,
    lines: np.ndarray,
    centre: np.ndarray | None,
    max_bytes: int,
) -> np.ndarray:
    """Mean squared leave-one-line-out error over tested training entries, one per setting.

    ``errors_op`` is ``[S, T, T]`` for ``T`` training lines: for each of ``S`` settings, the
    line-by-line matrix that turns the training answers (less ``centre``, when given) into the
    error each training line gets when it is left out, applied to every drug and gene at once.
    Each block's errors are squared, zeroed where the screen did not test the entry, and summed
    per (setting, drug); those per-drug sums are formed the same way whatever the block size and
    added over drugs once, at the end.
    """
    n_settings, n_train = errors_op.shape[0], lines.size
    _, n_drugs, n_genes = delta0.shape
    per_drug = np.empty((n_settings, n_drugs))
    n_tested = 0
    op = errors_op[:, None, :, :]
    bytes_per_drug = (n_settings + 2) * n_train * n_genes * 8
    for start, stop in drug_blocks(n_drugs, bytes_per_drug, max_bytes):
        block = _training_block(delta0, lines, start, stop, centre)
        measured = np.moveaxis(tested[lines, start:stop], 1, 0)
        errors = np.matmul(op, block)
        np.square(errors, out=errors)
        np.multiply(errors, measured, out=errors)
        per_drug[:, start:stop] = errors.reshape(n_settings, stop - start, -1).sum(axis=2)
        n_tested += int(np.count_nonzero(measured))
    if n_tested == 0:
        raise ValueError("the training lines have no tested entries to tune on")
    losses = per_drug.sum(axis=1) / n_tested
    if not np.isfinite(losses).all():
        raise FloatingPointError("a leave-one-out tuning loss is not finite")
    return losses


def _training_eigen(kernel: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(U, s)`` with the training kernel ``K[train][:, train] = U diag(s) Uᵀ``; eigenvalues
    below 0 (rounding error in a positive semidefinite kernel) are set to 0."""
    s, u = np.linalg.eigh(kernel[np.ix_(lines, lines)])
    return u, np.clip(s, 0.0, None)


def _ridge_errors_op(u: np.ndarray, s: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Kernel ridge's exact leave-one-line-out error operator, ``[len(lambdas), T, T]``.

    For departures ``R`` and penalty ``λ``, the dual coefficients are ``alpha = (K + λI)⁻¹ R``,
    the fitted departures ``K alpha = H R`` with hat matrix ``H = U diag(s/(s+λ)) Uᵀ``, and the
    training residual ``R - K alpha = U diag(λ/(s+λ)) Uᵀ R``. Line ``i``'s error when left out is
    its residual divided by ``1 - h_i``, with ``h_i = Σ_a U[i,a]² s_a/(s_a+λ)`` -- exact for
    ridge. The operator is that residual map with row ``i`` divided by ``1 - h_i``.
    """
    penalty = lambdas[:, None]
    shrunk = penalty / (s + penalty)
    leverage = (s / (s + penalty)) @ (u**2).T
    residual = np.matmul(u * shrunk[:, None, :], u.T)
    return residual / (1.0 - leverage)[:, :, None]


def _ridge_weights(
    kernel: np.ndarray, held: int, lines: np.ndarray, u: np.ndarray, s: np.ndarray, lam: float
) -> np.ndarray:
    """``k(held, train) (K_train + λI)⁻¹``: the weight each training line's departure carries
    in the hidden line's predicted departure, ``[T]``."""
    return ((kernel[held, lines] @ u) / (s + lam)) @ u.T


def _weighted_departures(
    weights: np.ndarray,
    delta0: np.ndarray,
    lines: np.ndarray,
    centre: np.ndarray,
    max_bytes: int,
) -> np.ndarray:
    """``centre + Σ_i weights_i (delta0[lines_i] - centre)`` for every drug and gene, ``[D, G]``."""
    _, n_drugs, n_genes = delta0.shape
    prediction = np.empty((n_drugs, n_genes))
    for start, stop in drug_blocks(n_drugs, 3 * lines.size * n_genes * 8, max_bytes):
        block = _training_block(delta0, lines, start, stop, centre)
        prediction[start:stop] = centre[start:stop] + np.matmul(weights, block)
    return prediction


def ridge_lolo(
    kernel: np.ndarray,
    delta0: np.ndarray,
    tested: np.ndarray,
    held: int,
    lambdas: np.ndarray = LAMBDAS,
    *,
    max_bytes: int = MAX_BYTES,
) -> Fit:
    """Ridge regression on a line description, hiding line ``held``.

    ``kernel`` is the ``[L, L]`` normalized linear kernel of the description over all lines
    (descriptions hold no drug response, so building it over every line reveals no answer).
    Departures from the training drug average are regressed on the description, for every drug
    and gene at once with one penalty shared by all of them: the penalty in ``lambdas`` with the
    smallest exact leave-one-line-out loss. The prediction is the training drug average plus
    ``k(held, train) (K_train + λI)⁻¹`` applied to the training departures -- ridge's
    prediction in kernel form. Pass ``lambdas=np.array([λ])`` to fit at one penalty (a one-value
    grid's choice is on both its edges, so that ``Fit.at_edge`` is True).
    """
    _check_answers(delta0, tested)
    n_lines = delta0.shape[0]
    lines = _training_lines(n_lines, held)
    _check_square(kernel, n_lines, "kernel")
    grid = _check_lambdas(lambdas)
    centre = _line_mean(delta0, lines)
    u, s = _training_eigen(kernel, lines)
    losses = _leave_one_out_losses(
        _ridge_errors_op(u, s, grid), delta0, tested, lines, centre, max_bytes
    )
    best = int(np.argmin(losses))
    lam = float(grid[best])
    weights = _ridge_weights(kernel, operator.index(held), lines, u, s, lam)
    return Fit(
        prediction=_weighted_departures(weights, delta0, lines, centre, max_bytes),
        lam=lam,
        k=None,
        loss_min=float(losses[best]),
        at_edge=best in (0, grid.size - 1),
    )


def ridge_lolo_k(
    descriptions: Mapping[int, np.ndarray],
    delta0: np.ndarray,
    tested: np.ndarray,
    held: int,
    lambdas: np.ndarray = LAMBDAS,
    *,
    max_bytes: int = MAX_BYTES,
) -> Fit:
    """Ridge regression choosing the number of description components and the penalty together.

    ``descriptions`` maps each candidate component count k to its ``[L, k]`` description (PCA
    or NMF with k components). Each is turned into its normalized linear kernel and fitted as in
    ``ridge_lolo``; the (k, penalty) pair with the smallest leave-one-line-out loss is chosen.
    ``Fit.at_edge`` also flags the smallest or largest candidate k. Ties go to the smaller k,
    then the smaller penalty.
    """
    _check_answers(delta0, tested)
    n_lines = delta0.shape[0]
    lines = _training_lines(n_lines, held)
    grid = _check_lambdas(lambdas)
    ks = sorted(descriptions)
    if not ks:
        raise ValueError("descriptions must hold at least one component count")
    misshapen = [
        k
        for k in ks
        if descriptions[k].shape != (n_lines, k) or not np.isfinite(descriptions[k]).all()
    ]
    if misshapen:
        raise ValueError(f"descriptions for k={misshapen} are not finite [{n_lines}, k] arrays")
    kernels = [linear_kernel(descriptions[k]) for k in ks]
    eigens = [_training_eigen(kernel, lines) for kernel in kernels]
    centre = _line_mean(delta0, lines)
    errors_op = np.concatenate([_ridge_errors_op(u, s, grid) for u, s in eigens])
    losses = _leave_one_out_losses(errors_op, delta0, tested, lines, centre, max_bytes)
    by_k = losses.reshape(len(ks), grid.size)
    k_index, lam_index = (int(i) for i in np.unravel_index(int(np.argmin(by_k)), by_k.shape))
    u, s = eigens[k_index]
    lam = float(grid[lam_index])
    weights = _ridge_weights(kernels[k_index], operator.index(held), lines, u, s, lam)
    return Fit(
        prediction=_weighted_departures(weights, delta0, lines, centre, max_bytes),
        lam=lam,
        k=ks[k_index],
        loss_min=float(by_k[k_index, lam_index]),
        at_edge=lam_index in (0, grid.size - 1) or k_index in (0, len(ks) - 1),
    )


def similarity_from_description(description: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of lines' standardized descriptions, ``[L, L]``.

    Each column is standardized across lines first, so no column dominates by its scale; each
    line's standardized row is then correlated with every other line's. A line whose
    standardized row is constant correlates 0 with every line. Nearest lines ranks by this.
    """
    standardized = standardize(description)
    return identity_correlations(standardized, standardized)


def _neighbour_ranks(similarity: np.ndarray) -> np.ndarray:
    """``rank[i, j]``: line ``j``'s place among line ``i``'s other lines, most similar first
    (0 = nearest), ties to the lower index; ``rank[i, i]`` is ``n``, so a line is never its own
    neighbour."""
    n = int(similarity.shape[0])
    order = np.argsort(-similarity, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(n), (n, n)), axis=1)
    diagonal = np.arange(n)
    ranks -= ranks > ranks[diagonal, diagonal][:, None]
    ranks[diagonal, diagonal] = n
    return ranks


def nearest_lines_lolo(
    similarity: np.ndarray,
    delta0: np.ndarray,
    tested: np.ndarray,
    held: int,
    ks: Sequence[int] = (3, 5, 10, 20),
    *,
    max_bytes: int = MAX_BYTES,
) -> Fit:
    """Nearest lines, hiding line ``held``: the mean change over the k most similar training lines.

    ``similarity`` is ``[L, L]`` (``similarity_from_description`` of the expression
    description). Tuning leaves each training line out in turn and predicts it by the mean of
    its k most similar other training lines; the candidate k with the smallest loss is chosen
    (capped at one fewer than the training lines for this inner fit). The hidden line's
    prediction is the mean over its k most similar training lines (capped at all of them).
    Neighbours are ranked by similarity, ties to the lower line index. ``Fit.k`` is the chosen
    candidate as given; ``Fit.lam`` is ``None``.
    """
    _check_answers(delta0, tested)
    n_lines = delta0.shape[0]
    lines = _training_lines(n_lines, held)
    _check_square(similarity, n_lines, "similarity")
    candidates = np.asarray(ks, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0 or (candidates < 1).any():
        raise ValueError("ks must be a non-empty sequence of positive neighbour counts")
    n_train = lines.size
    ranks = _neighbour_ranks(similarity[np.ix_(lines, lines)])
    inner_k = np.minimum(candidates, n_train - 1)[:, None, None]
    errors_op = np.eye(n_train) - (ranks < inner_k) / inner_k
    losses = _leave_one_out_losses(errors_op, delta0, tested, lines, None, max_bytes)
    best = int(np.argmin(losses))
    k = int(candidates[best])
    held_order = np.argsort(-similarity[operator.index(held), lines], kind="stable")
    return Fit(
        prediction=_line_mean(delta0, lines[held_order[: min(k, n_train)]]),
        lam=None,
        k=k,
        loss_min=float(losses[best]),
        at_edge=k in (int(candidates.min()), int(candidates.max())),
    )


__all__ = [
    "LAMBDAS",
    "MAX_BYTES",
    "Fit",
    "drug_average",
    "drug_blocks",
    "nearest_lines_lolo",
    "ridge_lolo",
    "ridge_lolo_k",
    "similarity_from_description",
]
