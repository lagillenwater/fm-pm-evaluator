"""Rung 1's models: what each one predicts for a hidden line or a hidden drug.

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

When a drug is hidden, a model is given every line's measured changes for the other drugs, the
drugs' chemical similarity ``T`` (Tanimoto similarity of Morgan fingerprints) and a description
of every line, and predicts the hidden drug's change in every line and gene:

* ``ridge_lodo`` -- for each gene, one ridge regression over every training (line, drug) pair
  at once, of each line's departures from its mean over the training drugs, in kernel form with
  the kernel ``T[d, d'] * (1 + K_line[l, l'])``: chemically similar drugs share effects, on
  average (the 1) and in lines with similar descriptions (``K_line``). The hidden drug's
  predicted departure is added back to each line's mean. One penalty is shared by every gene,
  chosen by the exact leave-one-drug-out error, computed in closed form (invariant 6).
* chemistry only -- ``ridge_lodo`` with ``K_line`` all zeros: similar drugs share effects on
  average, whatever the line.
* ``ridge_lodo_k`` -- the same, choosing the number of PCA or NMF components jointly with the
  penalty.

Conventions every function shares. ``delta0`` is the answer array ``[L, D, G]`` with untested
entries set to 0 (untested genes count as zero change in training); ``tested`` marks the entries
the screen measured. Tuning losses are mean squared leave-one-out error over tested training
entries only (invariant 7). Each leave-one-out fit re-estimates the round's reference -- the
drug average without the left-out line, or each line's mean without the left-out drug --
exactly and in closed form (invariant 6): a reference held at its full training value would
still contain the left-out unit's answer and bias every tuning loss. Leaving a line out, a
centred description hands that answer back at small penalties, so wide descriptions would look
best on pure noise. No function reads the hidden unit's answers (invariant 1): training blocks
are taken by indexing the training lines or drugs, and means over lines are masked sums that
never add a line outside the mask.

At full size one array of departures (49 training lines x 107 drugs x ~45k genes) is ~1.9 GB of
float64, so fits process data in blocks sized to a byte budget (``MAX_BYTES``): drugs when a
line is hidden, genes when a drug is hidden (genes are independent given the two kernels).
When a line is hidden, every per-drug quantity is computed the same way whatever the block size
and combined across drugs once at the end, so a fit's result does not depend on the budget,
bit for bit. When a drug is hidden, the per-block loss sums and matrix products change shape
with the block size, so results agree across budgets to rounding (1e-12 relative), not bit for
bit.
"""

from __future__ import annotations

import itertools
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

    ``prediction`` -- the hidden unit's predicted change: ``[D, G]`` when a line is hidden,
    ``[L, G]`` when a drug is hidden.
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
# Shared by both schemes: drug and gene blocks, input checks, masked means


def _blocks(n: int, bytes_each: int, max_bytes: int, kind: str) -> list[tuple[int, int]]:
    """Consecutive ``(start, stop)`` ranges over ``n`` drugs or genes (``kind``), each holding as
    many as ``max_bytes`` allows at ``bytes_each``; ``ValueError`` when one alone needs more."""
    if bytes_each > max_bytes:
        raise ValueError(
            f"one {kind} needs {bytes_each:,} bytes of working arrays, over the max_bytes "
            f"budget of {max_bytes:,}"
        )
    per_block = max_bytes // max(1, bytes_each)
    return [(start, min(start + per_block, n)) for start in range(0, n, per_block)]


def drug_blocks(n_drugs: int, bytes_per_drug: int, max_bytes: int) -> list[tuple[int, int]]:
    """Consecutive ``(start, stop)`` drug ranges, each holding as many drugs as ``max_bytes``
    allows at ``bytes_per_drug``.

    Raises ``ValueError`` when a single drug alone needs more than ``max_bytes``: a block cannot
    be smaller than one drug, so the budget cannot be met.
    """
    return _blocks(n_drugs, bytes_per_drug, max_bytes, "drug")


def gene_blocks(n_genes: int, bytes_per_gene: int, max_bytes: int) -> list[tuple[int, int]]:
    """Consecutive ``(start, stop)`` gene ranges, each holding as many genes as ``max_bytes``
    allows at ``bytes_per_gene``; ``ValueError`` when a single gene alone needs more."""
    return _blocks(n_genes, bytes_per_gene, max_bytes, "gene")


def _check_answers(delta0: np.ndarray, tested: np.ndarray) -> None:
    if delta0.ndim != 3:
        raise ValueError(f"delta0 must be [lines, drugs, genes]; got shape {delta0.shape}")
    if tested.shape != delta0.shape or tested.dtype != np.bool_:
        raise ValueError(
            f"tested must be a boolean array shaped like delta0 {delta0.shape}; "
            f"got {tested.dtype} {tested.shape}"
        )


def _check_square(matrix: np.ndarray, size: int, name: str) -> None:
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must be [{size}, {size}]; got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} has non-finite entries")


def _component_kernels(
    descriptions: Mapping[int, np.ndarray], n_lines: int
) -> tuple[list[int], list[np.ndarray]]:
    """The candidate component counts in increasing order, and each one's normalized linear
    kernel; ``ValueError`` unless every description is a finite ``[L, k]`` array."""
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
    return ks, [linear_kernel(descriptions[k]) for k in ks]


def _check_lambdas(lambdas: np.ndarray) -> np.ndarray:
    """The penalty grid as float64, checked: non-empty, 1-D, finite, positive, and strictly
    increasing (``Fit.at_edge`` reads its ends as the smallest and largest penalty, and ties
    go to the smaller penalty by position)."""
    grid = np.asarray(lambdas, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not (np.isfinite(grid) & (grid > 0)).all():
        raise ValueError("lambdas must be a non-empty 1-D array of finite positive penalties")
    if not (np.diff(grid) > 0).all():
        raise ValueError("lambdas must be strictly increasing")
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


def _training_indices(n: int, held: int, kind: str) -> np.ndarray:
    """Every line or drug index (``kind``) except ``held``, in order."""
    held = operator.index(held)
    if not 0 <= held < n:
        raise ValueError(f"held {kind} {held} is outside 0..{n - 1}")
    if n < 3:
        raise ValueError(f"leaving one {kind} out of training needs at least 3 {kind}s")
    return np.flatnonzero(np.arange(n) != held)


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

    Applied along the line axis to the departures ``R`` from the drug average over all ``T``
    training lines, it gives each training line's error when that line is left out of the fit
    AND of the drug average, for every drug and gene at once.

    ``s`` may carry leading axes, ``[..., T]``: one spectrum per ridge problem, all sharing the
    eigenvectors ``u``, giving ``[len(lambdas), ..., T, T]``. Leave one drug out uses this, with
    drugs in place of lines and one spectrum per line eigencomponent
    (``_leave_one_drug_out_ops``).

    Step 1, the average held fixed. For penalty ``λ`` the dual coefficients are
    ``alpha = (K + λI)⁻¹ R``, the fitted departures ``K alpha = H R`` with hat matrix
    ``H = U diag(s/(s+λ)) Uᵀ``, and the training residual ``R - K alpha = M R`` with
    ``M = I - H = U diag(λ/(s+λ)) Uᵀ``. Ridge's leave-one-out shortcut, exact for any targets:
    line ``i``'s error when left out of the fit is ``(M R)_i / (1 - h_i)``, and
    ``1 - h_i = M_ii``. So ``E = diag(1/M_ii) M``.

    Step 2, re-estimating the average without line ``i``. Fix one drug and gene; ``x_j`` is
    line ``j``'s answer, ``m`` the mean over the ``T`` training lines, ``R_j = x_j - m``
    (so ``Σ_j R_j = 0``). Without line ``i`` the average is
    ``m₋ᵢ = (T·m - x_i)/(T-1) = m - R_i/(T-1)``, and the other lines' departures from it are
    ``R_j + R_i/(T-1)``: the fixed-average targets plus the constant ``R_i/(T-1)``. Ridge's
    prediction ``f_i`` for line ``i`` from the other lines is linear in their targets, so
    ``f_i(R₋ᵢ + 1·R_i/(T-1)) = f_i(R₋ᵢ) + f_i(1)·R_i/(T-1)``, and line ``i``'s error is
    ``x_i - m₋ᵢ - f_i(...) = [R_i - f_i(R₋ᵢ)] + R_i·[1 - f_i(1)]/(T-1)``. The bracket on the
    left is ``e_i = (E R)_i``; ``1 - f_i(1)`` is line ``i``'s leave-one-out error when every
    target is 1, i.e. ``(E·1)_i``. Hence ``e'_i = e_i + R_i·(E·1)_i/(T-1)``:
    ``E' = E + diag(E·1)/(T-1)``. With the average held fixed instead, the other lines'
    targets still sum to ``-R_i``, and a centred description's constant direction hands line
    ``i``'s own answer back to it at small penalties.
    """
    n_train = int(u.shape[0])
    penalty = lambdas.reshape(-1, *([1] * s.ndim))
    residual_map = np.matmul(u * (penalty / (s + penalty))[..., None, :], u.T)
    # 1 - h_i read off M's diagonal: subtracting h_i from 1 cancels digits when h_i is near 1.
    # E is formed in M's own memory (dividing by a copy of its diagonal), so the operator is
    # held once, not twice -- under leave one drug out it is hundreds of megabytes.
    errors = residual_map
    errors /= np.diagonal(residual_map, axis1=-2, axis2=-1).copy()[..., :, None]
    diagonal = np.arange(n_train)
    errors[..., diagonal, diagonal] += errors.sum(axis=-1) / (n_train - 1)
    return errors


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
    smallest exact leave-one-line-out loss, each left-out fit re-estimating the drug average
    without its left-out line (``_ridge_errors_op``). The prediction is the training drug
    average plus ``k(held, train) (K_train + λI)⁻¹`` applied to the training departures --
    ridge's prediction in kernel form. Pass ``lambdas=np.array([λ])`` to fit at one penalty
    (a one-value grid's choice is on both its edges, so that ``Fit.at_edge`` is True).
    """
    _check_answers(delta0, tested)
    n_lines = delta0.shape[0]
    lines = _training_indices(n_lines, held, "line")
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
    lines = _training_indices(n_lines, held, "line")
    grid = _check_lambdas(lambdas)
    ks, kernels = _component_kernels(descriptions, n_lines)
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
    (capped at one fewer than the training lines for this inner fit). That prediction averages
    other lines' raw answers and estimates no reference from all training lines, so the
    left-out line's answer never enters its own prediction and nothing needs re-estimating
    (unlike ridge's drug average). The hidden line's
    prediction is the mean over its k most similar training lines (capped at all of them).
    Neighbours are ranked by similarity, ties to the lower line index. ``Fit.k`` is the chosen
    candidate as given; ``Fit.lam`` is ``None``.
    """
    _check_answers(delta0, tested)
    n_lines = delta0.shape[0]
    lines = _training_indices(n_lines, held, "line")
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


# ==============================================================================================
# Leave one drug out


def _drug_departures(
    delta0: np.ndarray, drugs: np.ndarray, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(means, departures)`` for genes ``start:stop``: each line's mean over the training drugs
    ``drugs``, ``[L, genes]``, and the training answers less it, a C-contiguous float64
    ``[L, drugs, genes]`` block.

    ``np.take`` copies only the listed drugs, so no other drug's answer is read, and the block
    is this function's own memory to centre in place.
    """
    block = np.take(delta0[:, :, start:stop], drugs, axis=1).astype(np.float64, copy=False)
    means = block.mean(axis=1)
    block -= means[:, None, :]
    return means, block


def _leave_one_drug_out_ops(
    line_s: np.ndarray, drug_u: np.ndarray, drug_s: np.ndarray, lambdas: np.ndarray
) -> np.ndarray:
    """Kernel ridge's exact leave-one-drug-out error operators, ``[len(lambdas), K, L, Dt, Dt]``.

    ``line_s`` is ``[K, L]``, the eigenvalues of ``K`` line factors
    ``1 + K_line = U_l diag(s_l) U_lᵀ``; ``drug_u`` and ``drug_s`` diagonalize the ``Dt`` training
    drugs' similarity, ``T_d = U_d diag(s_d) U_dᵀ``. Operator ``[λ, k, c]`` acts along the drug
    axis on line eigencomponent ``c`` of the departures, ``(U_lᵀ R)[c]``, and gives component
    ``c`` of every training drug's error when that drug is left out of the fit AND of the line
    means; rotating back by ``U_l`` gives the errors line by line.

    Fix one gene. ``x[l, j]`` is line ``l``'s answer to training drug ``j``, ``a[l]`` its mean over
    the training drugs, ``R = x - a`` (each line's departures sum to 0 over drugs). On pairs in
    (line, drug) order the kernel is ``K = (1 + K_line) ⊗ T_d``, with eigenvectors ``U_l ⊗ U_d``
    and eigenvalues ``w[c, a] = s_l[c] s_d[a]``, so the hat matrix is
    ``H = (U_l ⊗ U_d) diag(w/(w+λ)) (U_l ⊗ U_d)ᵀ``.

    Step 1, the line means held fixed. Leaving drug ``j`` out removes the block of its ``L``
    pairs. Ridge's block leave-out identity, exact for any targets ``y`` (from the partitioned
    inverse of ``K + λI``, as ``I - H = λ(K + λI)⁻¹``): the block's error when it is left out of
    the fit is ``r_j = (I - H_jj)⁻¹ ((I - H) y)_j``, where ``H_jj`` is ``H``'s ``L``-by-``L`` block
    for drug ``j``. Here ``H_jj = U_l diag(h_j) U_lᵀ`` with
    ``h_j[c] = Σ_a U_d[j,a]² w[c,a]/(w[c,a]+λ)``: the block shares the line eigenvectors, so its
    inverse is diagonal in that basis and ``(U_lᵀ r_j)[c] = (U_lᵀ (I - H) R)[c, j] / (1 - h_j[c])``.
    With ``M_c = U_d diag(λ/(w[c]+λ)) U_dᵀ`` the numerator is ``(M_c (U_lᵀ R)[c])_j``, and as
    ``Σ_a U_d[j,a]² = 1``, ``1 - h_j[c] = M_c[j, j]``, read off the diagonal without subtracting
    from 1. So component ``c`` is ordinary leave-one-out ridge over drugs with the kernel
    ``s_l[c] T_d``: ``E_c = diag(1/M_c[j, j]) M_c``, step 1 of ``_ridge_errors_op``.

    Step 2, re-estimating the line means without drug ``j``. They become
    ``a₋ⱼ = a - R[:, j]/(Dt-1)``, so the other drugs' departures from them are the fixed-mean
    targets plus ``v_j ⊗ 1``, with ``v_j = R[:, j]/(Dt-1)`` the same for every drug. Drug
    ``j``'s prediction is linear in its targets, so its error is the step-1 error plus
    ``v_j - f_j(v_j ⊗ 1)``: the block leave-out error of the target ``v_j ⊗ 1`` over all training
    drugs. By the same identity its component ``c`` is ``(U_lᵀ v_j)[c] (1 - g_j[c])/(1 - h_j[c])``,
    where ``1 - g_j[c] = Σ_a U_d[j,a] (U_dᵀ 1)[a] λ/(w[c,a]+λ) = (M_c 1)_j`` (as
    ``U_d U_dᵀ 1 = 1``), again not a subtraction from 1. Hence
    ``(U_lᵀ r'_j)[c] = (E_c (U_lᵀ R)[c])_j + (U_lᵀ R)[c, j] (E_c 1)_j / (Dt-1)``: component by
    component ``E'_c = E_c + diag(E_c 1)/(Dt-1)``, step 2 of ``_ridge_errors_op``, which builds
    every component's operator at once from the batch of spectra ``w``. Without step 2 each
    line's fixed mean would still hold the left-out drug's answer.
    """
    return _ridge_errors_op(drug_u, line_s[:, :, None] * drug_s, lambdas)


def _block_squared_errors(
    line_u: np.ndarray, errors_op: np.ndarray, departures: np.ndarray, measured: np.ndarray
) -> np.ndarray:
    """One gene block's summed squared leave-one-drug-out errors over its tested entries,
    ``[K, S]``, from its departures ``[L, Dt, n]`` and tested marks of the same shape.

    The departures are rotated into each kernel's line eigenbasis once; then, setting by
    setting, the operator is applied along the drug axis, the result rotated back to lines and
    squared where the screen tested it. Each setting writes into the same two work arrays.
    """
    n_settings, n_kernels, n_lines = errors_op.shape[:3]
    weights = measured.reshape(n_lines, -1).astype(np.float64)
    rotated = np.matmul(np.swapaxes(line_u, 1, 2), departures.reshape(n_lines, -1))
    rotated = rotated.reshape((n_kernels, *departures.shape))
    rotated_errors = np.empty(departures.shape)
    errors = np.empty(weights.shape)
    sums = np.empty((n_kernels, n_settings))
    for k, s in itertools.product(range(n_kernels), range(n_settings)):
        np.matmul(errors_op[s, k], rotated[k], out=rotated_errors)
        np.matmul(line_u[k], rotated_errors.reshape(n_lines, -1), out=errors)
        masked = np.multiply(errors, weights, out=rotated_errors.reshape(n_lines, -1))
        sums[k, s] = np.dot(errors.ravel(), masked.ravel())
    return sums


def _leave_one_drug_out_losses(
    line_u: np.ndarray,
    errors_op: np.ndarray,
    delta0: np.ndarray,
    tested: np.ndarray,
    drugs: np.ndarray,
    max_bytes: int,
) -> np.ndarray:
    """Mean squared leave-one-drug-out error over tested training entries, ``[K, S]``: one per
    line kernel ``k`` (eigenvectors ``line_u[k]``) and setting ``s`` of ``errors_op[s, k]``.

    Genes are processed in blocks, as many as ``max_bytes`` holds once the operators are
    counted (``_block_squared_errors``); the blocks' sums are added before any setting is
    chosen.
    """
    n_settings, n_kernels = errors_op.shape[:2]
    n_lines, _, n_genes = delta0.shape
    # per gene, in float64 [L, Dt] units: the departures (and a float32 input's transient copy),
    # the tested marks as weights, the K rotated departures, and the two work arrays
    bytes_per_gene = (n_kernels + 5) * n_lines * drugs.size * 8
    room = max_bytes - errors_op.nbytes
    if room < bytes_per_gene:
        raise ValueError(
            f"the leave-one-drug-out operators ({errors_op.nbytes:,} bytes) and one gene's "
            f"working arrays ({bytes_per_gene:,} bytes) exceed the max_bytes budget of "
            f"{max_bytes:,}"
        )
    totals = np.zeros((n_kernels, n_settings))
    n_tested = 0
    for start, stop in gene_blocks(n_genes, bytes_per_gene, room):
        _, departures = _drug_departures(delta0, drugs, start, stop)
        measured = np.take(tested[:, :, start:stop], drugs, axis=1)
        n_tested += int(np.count_nonzero(measured))
        totals += _block_squared_errors(line_u, errors_op, departures, measured)
        del departures, measured  # released before the next block is copied
    if n_tested == 0:
        raise ValueError("the training drugs have no tested entries to tune on")
    losses = totals / n_tested
    if not np.isfinite(losses).all():
        raise FloatingPointError("a leave-one-out tuning loss is not finite")
    return losses


def _lodo_prediction(
    line_factor: np.ndarray,
    line_u: np.ndarray,
    line_s: np.ndarray,
    drug_u: np.ndarray,
    drug_s: np.ndarray,
    similarity: np.ndarray,
    lam: float,
    delta0: np.ndarray,
    drugs: np.ndarray,
    max_bytes: int,
) -> np.ndarray:
    """The hidden drug's predicted change in every line and gene, ``[L, G]``.

    Ridge's prediction in kernel form is ``a + (1 + K_line) (alpha t)``: ``a`` is each line's mean
    over the training drugs, ``t`` (``similarity``) the hidden drug's similarity to each
    training drug, and ``alpha = (K + λI)⁻¹ R`` the dual coefficients, ``[L, Dt]`` per gene
    (``line_factor = 1 + K_line = U_l diag(line_s) U_lᵀ``). In the two eigenbases
    ``alpha = U_l A U_dᵀ`` with ``A[c, a] = (U_lᵀ R U_d)[c, a] / (w[c, a] + λ)``, so
    ``alpha t = U_l b`` with ``b[c] = Σ_j Q[c, j] (U_lᵀ R)[c, j]`` and
    ``Q = ((U_dᵀ t) / (w + λ)) U_dᵀ``: one rotation of each gene block's departures.
    """
    n_lines, _, n_genes = delta0.shape
    drug_weights = ((drug_u.T @ similarity) / (line_s[:, None] * drug_s + lam)) @ drug_u.T
    to_lines = line_factor @ line_u
    prediction = np.empty((n_lines, n_genes))
    # per gene, in float64 [L, Dt] units: the departures (and a float32 input's transient copy)
    # and their rotation
    for start, stop in gene_blocks(n_genes, 3 * n_lines * drugs.size * 8, max_bytes):
        means, departures = _drug_departures(delta0, drugs, start, stop)
        rotated = np.matmul(line_u.T, departures.reshape(n_lines, -1))
        rotated = rotated.reshape(departures.shape)
        coefficients = np.matmul(drug_weights[:, None, :], rotated)[:, 0, :]
        prediction[:, start:stop] = means + to_lines @ coefficients
        del departures, rotated  # released before the next block is copied
    return prediction


def _ridge_lodo_fit(
    line_kernels: Sequence[np.ndarray],
    ks: Sequence[int] | None,
    T: np.ndarray,
    delta0: np.ndarray,
    tested: np.ndarray,
    held: int,
    lambdas: np.ndarray,
    max_bytes: int,
) -> Fit:
    """Ridge when a drug is hidden, over one or more line kernels: the (kernel, penalty) pair
    with the smallest exact leave-one-drug-out loss, ties to the earlier kernel and then the
    smaller penalty, and the hidden drug's prediction under it. ``ks`` names the kernels by
    component count, or is ``None`` for a single kernel with no count."""
    n_lines, n_drugs, _ = delta0.shape
    held = operator.index(held)
    drugs = _training_indices(n_drugs, held, "drug")
    _check_square(T, n_drugs, "T")
    grid = _check_lambdas(lambdas)
    factors = [1.0 + kernel for kernel in line_kernels]
    every_line = np.arange(int(n_lines))
    eigens = [_training_eigen(factor, every_line) for factor in factors]
    line_u = np.stack([u for u, _ in eigens])
    line_s = np.stack([s for _, s in eigens])
    drug_u, drug_s = _training_eigen(T, drugs)
    losses = _leave_one_drug_out_losses(
        line_u,
        _leave_one_drug_out_ops(line_s, drug_u, drug_s, grid),
        delta0,
        tested,
        drugs,
        max_bytes,
    )
    k_index, lam_index = (int(i) for i in np.unravel_index(int(np.argmin(losses)), losses.shape))
    lam = float(grid[lam_index])
    prediction = _lodo_prediction(
        factors[k_index],
        line_u[k_index],
        line_s[k_index],
        drug_u,
        drug_s,
        T[held, drugs],
        lam,
        delta0,
        drugs,
        max_bytes,
    )
    return Fit(
        prediction=prediction,
        lam=lam,
        k=None if ks is None else ks[k_index],
        loss_min=float(losses[k_index, lam_index]),
        at_edge=lam_index in (0, grid.size - 1) or (ks is not None and k_index in (0, len(ks) - 1)),
    )


def ridge_lodo(
    K_line: np.ndarray,
    T: np.ndarray,
    delta0: np.ndarray,
    tested: np.ndarray,
    held: int,
    lambdas: np.ndarray = LAMBDAS,
    *,
    max_bytes: int = MAX_BYTES,
) -> Fit:
    """Ridge regression on chemistry and a line description, hiding drug ``held``.

    ``K_line`` is the ``[L, L]`` normalized linear kernel of a line description
    (``descriptions.linear_kernel``), or all zeros for chemistry only; ``T`` is the ``[D, D]``
    Tanimoto similarity in grid drug order. For every gene, with one penalty shared by all of
    them, each line's departures from its mean over the training drugs are regressed on the
    pair kernel ``T[d, d'] * (1 + K_line[l, l'])`` over every training (line, drug) pair. The
    penalty is the one in ``lambdas`` with the smallest exact leave-one-drug-out loss, each
    left-out fit re-estimating the line means without its left-out drug
    (``_leave_one_drug_out_ops``). The prediction, ``[L, G]``, is each line's mean over the
    training drugs plus ridge's prediction in kernel form for a drug whose similarities to the
    training drugs are ``T[held, train]`` (``_lodo_prediction``). Only the training drugs'
    answers and tested marks are read. Pass ``lambdas=np.array([λ])`` to fit at one penalty
    (a one-value grid's choice is on both its edges, so that ``Fit.at_edge`` is True).
    """
    _check_answers(delta0, tested)
    _check_square(K_line, delta0.shape[0], "K_line")
    return _ridge_lodo_fit([K_line], None, T, delta0, tested, held, lambdas, max_bytes)


def ridge_lodo_k(
    Zs: Mapping[int, np.ndarray],
    T: np.ndarray,
    delta0: np.ndarray,
    tested: np.ndarray,
    held: int,
    lambdas: np.ndarray = LAMBDAS,
    *,
    max_bytes: int = MAX_BYTES,
) -> Fit:
    """``ridge_lodo`` choosing the number of description components and the penalty together.

    ``Zs`` maps each candidate component count k to its ``[L, k]`` description (PCA or NMF with k
    components). Each is turned into its normalized linear kernel and fitted as in
    ``ridge_lodo``; the (k, penalty) pair with the smallest leave-one-drug-out loss is chosen.
    ``Fit.at_edge`` also flags the smallest or largest candidate k. Ties go to the smaller k,
    then the smaller penalty.
    """
    _check_answers(delta0, tested)
    ks, kernels = _component_kernels(Zs, delta0.shape[0])
    return _ridge_lodo_fit(kernels, ks, T, delta0, tested, held, lambdas, max_bytes)


__all__ = [
    "LAMBDAS",
    "MAX_BYTES",
    "Fit",
    "drug_average",
    "drug_blocks",
    "gene_blocks",
    "nearest_lines_lolo",
    "ridge_lodo",
    "ridge_lodo_k",
    "ridge_lolo",
    "ridge_lolo_k",
    "similarity_from_description",
]
