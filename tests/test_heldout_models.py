"""Rung 1's models when a line is hidden (task 7): drug average, ridge with closed-form
leave-one-line-out tuning, ridge over a choice of component count, and nearest lines. And when a
drug is hidden (task 8): ridge on chemistry and a line description with closed-form
leave-one-drug-out tuning, chemistry only, and ridge over a choice of component count.

Every test runs the real functions from ``fmharness.heldout.models`` on small synthetic grids.
Exactness is checked against independent computations: scikit-learn's primal ridge and a dense
Kronecker solve for the predictions, brute-force refits for the closed-form tuning losses
(invariant 6), and an explicit Python ranking for nearest lines. Invariant 1 is checked by
poisoning the hidden line's or drug's answers.
"""

# scikit-learn ships no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *its* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from fmharness.heldout import models
from fmharness.heldout.descriptions import linear_kernel, standardize
from fmharness.heldout.models import (
    LAMBDAS,
    Fit,
    drug_average,
    drug_blocks,
    gene_blocks,
    nearest_lines_lolo,
    ridge_lodo,
    ridge_lodo_k,
    ridge_lolo,
    ridge_lolo_k,
    similarity_from_description,
)

pytestmark = pytest.mark.step_fit


def _random_grid(
    n_lines: int = 12,
    n_drugs: int = 4,
    n_genes: int = 30,
    width: int = 5,
    seed: int = 0,
    untested: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(description, delta0, tested)``: random answers with a share left untested (set to 0
    in ``delta0``, as the models receive them) and a random description."""
    rng = np.random.default_rng(seed)
    description = rng.standard_normal((n_lines, width))
    delta = rng.standard_normal((n_lines, n_drugs, n_genes))
    tested = rng.random(delta.shape) >= untested
    return description, np.where(tested, delta, 0.0), tested


def _planted_grid(
    n_lines: int,
    n_drugs: int,
    n_genes: int,
    width: int,
    signal_width: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(description, delta0, tested)`` where every drug and gene's change is a drug baseline
    plus a linear function of the line's first ``signal_width`` standardized description
    columns, plus Gaussian noise of SD ``noise``; 5% of entries untested."""
    rng = np.random.default_rng(seed)
    description = rng.standard_normal((n_lines, width))
    effect = rng.standard_normal((signal_width, n_drugs, n_genes))
    baseline = rng.standard_normal((n_drugs, n_genes))
    signal = np.einsum("lc,cdg->ldg", standardize(description)[:, :signal_width], effect)
    delta = baseline + signal + noise * rng.standard_normal((n_lines, n_drugs, n_genes))
    tested = rng.random(delta.shape) >= 0.05
    return description, np.where(tested, delta, 0.0), tested


def _training_lines(n_lines: int, held: int) -> np.ndarray:
    return np.flatnonzero(np.arange(n_lines) != held)


def _assert_same_fit(a: Fit, b: Fit) -> None:
    assert a.prediction.dtype == b.prediction.dtype
    assert np.array_equal(a.prediction, b.prediction)
    assert (a.lam, a.k, a.loss_min, a.at_edge) == (b.lam, b.k, b.loss_min, b.at_edge)


# ==============================================================================================
# Ridge: the prediction and the closed-form tuning loss


@pytest.mark.parametrize("lam", [1e-2, 1.0, 30.0])
def test_ridge_lolo_equals_primal_ridge(lam: float) -> None:
    """At a fixed penalty, the kernel-form prediction is scikit-learn's primal ridge (no
    intercept) on the training departures from the drug average, added back to the average."""
    description, delta0, tested = _random_grid()
    held = 3
    train = _training_lines(12, held)
    standardized = standardize(description)
    scale = float(np.diag(standardized @ standardized.T).mean())
    features = standardized / np.sqrt(scale)
    kernel = linear_kernel(description)
    np.testing.assert_allclose(features @ features.T, kernel, rtol=1e-12, atol=1e-12)

    average = delta0[train].mean(axis=0)
    departures = (delta0[train] - average).reshape(train.size, -1)
    primal = Ridge(alpha=lam, fit_intercept=False).fit(features[train], departures)
    expected = average + primal.predict(features[[held]]).reshape(average.shape)

    fit = ridge_lolo(kernel, delta0, tested, held, lambdas=np.array([lam]))
    assert fit.lam == lam
    assert fit.k is None
    assert fit.prediction.shape == (4, 30)
    np.testing.assert_allclose(fit.prediction, expected, rtol=1e-9, atol=1e-12)


def _brute_force_lolo_losses(
    kernel: np.ndarray, delta0: np.ndarray, tested: np.ndarray, held: int, lambdas: np.ndarray
) -> np.ndarray:
    """Tuning loss per penalty by refitting everything with each training line left out.

    For each left-out training line j (invariant 6): the drug average is recomputed over the
    remaining training lines, their departures from it are formed, ridge is solved directly on
    the remaining kernel (``np.linalg.solve`` of ``(K_rest + λI) alpha = R_rest`` -- no
    eigendecomposition, no hat matrix), and line j is predicted as its re-estimated average
    plus ``k(j, rest) alpha``. Batched over penalties and left-out lines at once.
    """
    train = _training_lines(kernel.shape[0], held)
    n_train = train.size
    answers = delta0[train].reshape(n_train, -1)
    measured = tested[train].reshape(n_train, -1)
    k_train = kernel[np.ix_(train, train)]
    # row j: every training position except j
    rest = np.tile(np.arange(n_train), (n_train, 1))[~np.eye(n_train, dtype=bool)]
    rest = rest.reshape(n_train, n_train - 1)
    rest_answers = answers[rest]
    rest_average = rest_answers.mean(axis=1)
    rest_departures = rest_answers - rest_average[:, None, :]
    k_rest = k_train[rest[:, :, None], rest[:, None, :]]
    k_left_out = np.take_along_axis(k_train, rest, axis=1)
    system = k_rest[None] + lambdas[:, None, None, None] * np.eye(n_train - 1)
    alpha = np.linalg.solve(system, rest_departures[None])
    refit_prediction = rest_average[None] + np.einsum("jm,ljmn->ljn", k_left_out, alpha)
    errors = answers[None] - refit_prediction
    return (errors**2 * measured[None]).sum(axis=(1, 2)) / measured.sum()


@pytest.mark.parametrize("width", [5, 30])
def test_ridge_lolo_closed_form_loo_equals_refits(width: int) -> None:
    """Invariant 6: the closed-form leave-one-line-out loss at every penalty equals the loss
    from refitting with each training line left out of both the ridge fit and the drug average,
    to 1e-10 relative -- for a description narrower than the training lines (rank-deficient
    kernel) and one wider (full rank)."""
    description, delta0, tested = _random_grid(width=width)
    held = 7
    kernel = linear_kernel(description)
    brute = _brute_force_lolo_losses(kernel, delta0, tested, held, np.asarray(LAMBDAS))
    closed = np.array(
        [
            ridge_lolo(kernel, delta0, tested, held, lambdas=np.array([lam])).loss_min
            for lam in LAMBDAS
        ]
    )
    np.testing.assert_allclose(closed, brute, rtol=1e-10, atol=0.0)

    fit = ridge_lolo(kernel, delta0, tested, held)
    assert fit.lam == LAMBDAS[int(np.argmin(brute))]
    assert fit.loss_min == pytest.approx(float(brute.min()), rel=1e-10)


def test_lolo_tuning_does_not_reward_width_on_noise() -> None:
    """Pure-noise answers (i.i.d. normal, variance 1, independent of a full-rank 400-column
    description): tuning must not select the smallest penalty.

    What the math guarantees. Line i's leave-one-out prediction puts weights c_i on the other
    T-1 training lines' answers, summing to 1 (the re-estimated average plus a ridge term on
    departures from it). With noise independent of the description, its expected squared error
    is 1 + |c_i|², and |c_i|² is smallest, 1/(T-1), for the plain average, which the ridge term
    vanishes toward as the penalty grows (expected loss -> 1 + 1/48 here). At the smallest
    penalty ridge nearly interpolates 48 noisy lines and the weights spread. So in expectation
    the loss at the smallest penalty is above the loss at the largest, and the smallest is not
    chosen. Which of the nearly flat large penalties a finite sample picks is NOT guaranteed
    (100 or 1,000 across seeds), so the test asserts only that the chosen penalty is not
    ``LAMBDAS[0]`` and the ordering of the two ends. Holding the drug average fixed instead
    (the reversed plan) selects ``LAMBDAS[0]`` here, with a loss near 0.3: the left-out line's
    own answer leaking back.
    """
    rng = np.random.default_rng(17)
    n_lines = 50
    delta0 = rng.standard_normal((n_lines, 3, 300))
    tested = np.ones(delta0.shape, dtype=bool)
    kernel = linear_kernel(rng.standard_normal((n_lines, 400)))
    held = 0

    fit = ridge_lolo(kernel, delta0, tested, held)
    assert fit.lam != LAMBDAS[0]
    at_smallest = ridge_lolo(kernel, delta0, tested, held, lambdas=LAMBDAS[:1]).loss_min
    at_largest = ridge_lolo(kernel, delta0, tested, held, lambdas=LAMBDAS[-1:]).loss_min
    assert at_smallest > at_largest


# ==============================================================================================
# Invariant 1: nothing reads the hidden line's answers


def test_lolo_ignores_held_out_answers() -> None:
    description, delta0, tested = _random_grid(n_lines=14, n_drugs=5, n_genes=25, width=6)
    held = 5
    rng = np.random.default_rng(99)
    poisoned = delta0.copy()
    poisoned[held] = np.where(
        rng.random(delta0.shape[1:]) < 0.5, np.nan, 1e6 * rng.standard_normal(delta0.shape[1:])
    )
    poisoned_tested = tested.copy()
    poisoned_tested[held] = rng.random(tested.shape[1:]) < 0.5

    kernel = linear_kernel(description)
    by_k = {k: description[:, :k] for k in (2, 3, 6)}
    similarity = similarity_from_description(description)
    train = _training_lines(14, held)

    _assert_same_fit(
        ridge_lolo(kernel, delta0, tested, held),
        ridge_lolo(kernel, poisoned, poisoned_tested, held),
    )
    _assert_same_fit(
        ridge_lolo_k(by_k, delta0, tested, held),
        ridge_lolo_k(by_k, poisoned, poisoned_tested, held),
    )
    _assert_same_fit(
        nearest_lines_lolo(similarity, delta0, tested, held, ks=(2, 3, 5)),
        nearest_lines_lolo(similarity, poisoned, poisoned_tested, held, ks=(2, 3, 5)),
    )
    assert np.array_equal(drug_average(delta0, train), drug_average(poisoned, train))


# ==============================================================================================
# Nearest lines


def test_nearest_lines_with_every_training_line_is_the_drug_average() -> None:
    description, delta0, tested = _random_grid()
    held = 2
    n_train = 11
    fit = nearest_lines_lolo(
        similarity_from_description(description), delta0, tested, held, ks=(n_train,)
    )
    assert np.array_equal(fit.prediction, drug_average(delta0, _training_lines(12, held)))
    assert fit.k == n_train
    assert fit.lam is None
    assert fit.at_edge


def test_nearest_lines_matches_an_explicit_ranking_with_ties_to_the_lower_index() -> None:
    """The tuning loss and the prediction equal those built from Python's own sort on
    (-similarity, line index), on a similarity with three distinct values (so ties abound)."""
    n_lines, held, k = 10, 4, 3
    _, delta0, tested = _random_grid(n_lines=n_lines, n_drugs=3, n_genes=20)
    similarity = np.random.default_rng(5).choice([0.0, 0.5, 1.0], size=(n_lines, n_lines))
    train = _training_lines(n_lines, held)
    ranked = np.sort(similarity[held, train])[::-1]
    assert ranked[k - 1] == ranked[k]  # the hidden line's k-th place is tied: order decides it

    def nearest(line: int, pool: np.ndarray) -> list[int]:
        return sorted(pool.tolist(), key=lambda j: (-similarity[line, j], j))[:k]

    errors = np.stack(
        [delta0[i] - delta0[nearest(i, np.setdiff1d(train, [i]))].mean(axis=0) for i in train]
    )
    expected_loss = float((errors**2 * tested[train]).sum() / tested[train].sum())

    fit = nearest_lines_lolo(similarity, delta0, tested, held, ks=(k,))
    assert fit.loss_min == pytest.approx(expected_loss, rel=1e-12)
    np.testing.assert_allclose(
        fit.prediction, delta0[nearest(held, train)].mean(axis=0), rtol=1e-12, atol=1e-15
    )


def test_similarity_is_pearson_correlation_of_standardized_descriptions() -> None:
    description = np.random.default_rng(2).standard_normal((9, 40)) * np.arange(1, 41)
    np.testing.assert_allclose(
        similarity_from_description(description),
        np.corrcoef(standardize(description)),
        rtol=1e-12,
        atol=1e-12,
    )


# ==============================================================================================
# Drug blocks: the byte budget changes memory, never the result


def test_drug_blocks_respect_the_budget() -> None:
    assert drug_blocks(7, 39_600, 100_000) == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert drug_blocks(7, 10, 10) == [(i, i + 1) for i in range(7)]
    assert drug_blocks(7, 10, 10**9) == [(0, 7)]
    with pytest.raises(ValueError, match=r"one drug needs 11 bytes.*max_bytes budget of 10"):
        drug_blocks(7, 11, 10)


@pytest.mark.parametrize("drugs_per_block", [1, 2])
@pytest.mark.parametrize("model", ["ridge", "ridge_k", "nearest_lines"])
def test_chunked_fits_are_bit_identical(
    model: str, drugs_per_block: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each fit at the default budget (one block) equals, bit for bit, the same fit at a budget
    holding ``drugs_per_block`` drugs of its most demanding pass (7 drugs: blocks of 1, or
    uneven blocks 2, 2, 2, 1). A spy on ``drug_blocks`` reads each pass's per-drug size and
    checks the blocks really split."""
    description, delta0, tested = _random_grid(n_drugs=7)
    held = 3
    kernel = linear_kernel(description)
    by_k = {k: description[:, :k] for k in (2, 3, 5)}
    similarity = similarity_from_description(description)
    fits: dict[str, Callable[[int], Fit]] = {
        "ridge": lambda budget: ridge_lolo(kernel, delta0, tested, held, max_bytes=budget),
        "ridge_k": lambda budget: ridge_lolo_k(by_k, delta0, tested, held, max_bytes=budget),
        "nearest_lines": lambda budget: nearest_lines_lolo(
            similarity, delta0, tested, held, ks=(2, 3, 5), max_bytes=budget
        ),
    }
    passes: list[tuple[int, int]] = []  # (bytes per drug, number of blocks), per call

    def recording_blocks(n_drugs: int, bytes_per_drug: int, budget: int) -> list[tuple[int, int]]:
        blocks = drug_blocks(n_drugs, bytes_per_drug, budget)
        passes.append((bytes_per_drug, len(blocks)))
        return blocks

    monkeypatch.setattr(models, "drug_blocks", recording_blocks)
    whole = fits[model](models.MAX_BYTES)
    assert {n_blocks for _, n_blocks in passes} == {1}
    budget = drugs_per_block * max(size for size, _ in passes)
    passes.clear()
    chunked = fits[model](budget)
    assert max(n_blocks for _, n_blocks in passes) == -(-7 // drugs_per_block)
    _assert_same_fit(whole, chunked)


# ==============================================================================================
# at_edge


def test_at_edge_flags_a_choice_on_the_edge_of_its_grid() -> None:
    """A planted response on 5 of 10 description columns, with noise: tuning picks an interior
    penalty, k = 5 of {2, 5, 10}, and k = 3 of {1, 3, 8} nearest lines in clusters of 4. The
    same data with the grid cut so that choice sits on its bottom end, then on its top end, is
    flagged both times; the choice itself does not change."""
    description, delta0, tested = _planted_grid(20, 4, 100, 10, 5, noise=1.0, seed=1)
    held = 0
    kernel = linear_kernel(description[:, :5])

    interior = ridge_lolo(kernel, delta0, tested, held)
    assert interior.lam is not None
    assert LAMBDAS[0] < interior.lam < LAMBDAS[-1]
    assert not interior.at_edge
    bottom = ridge_lolo(kernel, delta0, tested, held, lambdas=LAMBDAS[interior.lam <= LAMBDAS])
    assert (bottom.lam, bottom.at_edge) == (interior.lam, True)
    top = ridge_lolo(kernel, delta0, tested, held, lambdas=LAMBDAS[interior.lam >= LAMBDAS])
    assert (top.lam, top.at_edge) == (interior.lam, True)

    def ridge_over(ks: tuple[int, ...]) -> tuple[int | None, float | None, bool]:
        fit = ridge_lolo_k({k: description[:, :k] for k in ks}, delta0, tested, held)
        return fit.k, fit.lam, fit.at_edge

    assert ridge_over((2, 5, 10)) == (5, interior.lam, False)
    assert ridge_over((5, 10)) == (5, interior.lam, True)
    assert ridge_over((2, 5)) == (5, interior.lam, True)

    rng = np.random.default_rng(3)
    cluster = np.repeat(np.arange(4), 4)
    clustered = rng.standard_normal((4, 30))[cluster] + 0.3 * rng.standard_normal((16, 30))
    responses = rng.standard_normal((4, 3, 50))[cluster] + 0.5 * rng.standard_normal((16, 3, 50))
    all_tested = np.ones(responses.shape, dtype=bool)
    similarity = similarity_from_description(clustered)

    def nearest_over(ks: tuple[int, ...]) -> tuple[int | None, bool]:
        fit = nearest_lines_lolo(similarity, responses, all_tested, held, ks=ks)
        return fit.k, fit.at_edge

    assert nearest_over((1, 3, 8)) == (3, False)
    assert nearest_over((3, 8)) == (3, True)
    assert nearest_over((1, 3)) == (3, True)


# ==============================================================================================
# A planted line-linear response


@pytest.mark.known_answer
@pytest.mark.parametrize("held", [0, 1, 2])
def test_ridge_beats_drug_average_on_a_planted_line_linear_response(held: int) -> None:
    """With a strong response linear in the line description (SD 1 per column against noise SD
    0.3), ridge's squared error on the hidden line's tested entries is below the drug average's.
    The full known-answer control, placed relative to the MDE, is task 9's."""
    description, delta0, tested = _planted_grid(30, 6, 200, 4, 4, noise=0.3, seed=7)
    fit = ridge_lolo(linear_kernel(description), delta0, tested, held)
    average = drug_average(delta0, _training_lines(30, held))
    ridge_error = float(((fit.prediction - delta0[held]) ** 2 * tested[held]).sum())
    average_error = float(((average - delta0[held]) ** 2 * tested[held]).sum())
    assert ridge_error < average_error


# ==============================================================================================
# Input contracts


def test_models_reject_malformed_inputs() -> None:
    description, delta0, tested = _random_grid()
    kernel = linear_kernel(description)
    with pytest.raises(ValueError, match="lambdas"):
        ridge_lolo(kernel, delta0, tested, 0, lambdas=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        ridge_lolo(kernel, delta0, tested, 0, lambdas=np.array([1.0, 0.1]))
    with pytest.raises(ValueError, match="strictly increasing"):
        ridge_lolo(kernel, delta0, tested, 0, lambdas=np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="max_bytes budget of 1,000"):
        ridge_lolo(kernel, delta0, tested, 0, max_bytes=1_000)
    with pytest.raises(ValueError, match="held line"):
        ridge_lolo(kernel, delta0, tested, 12)
    with pytest.raises(ValueError, match="tested"):
        ridge_lolo(kernel, delta0, tested.astype(np.float64), 0)
    with pytest.raises(ValueError, match="descriptions"):
        ridge_lolo_k({3: description}, delta0, tested, 0)
    with pytest.raises(ValueError, match="distinct"):
        drug_average(delta0, np.array([1, 1, 2]))
    with pytest.raises(ValueError, match="no tested entries"):
        ridge_lolo(kernel, delta0, np.zeros_like(tested), 0)


# ==============================================================================================
# Leave one drug out (task 8): grids


def _tanimoto(fingerprints: np.ndarray) -> np.ndarray:
    """Tanimoto similarity of boolean fingerprints, every row with at least one bit set."""
    bits = fingerprints.astype(np.float64)
    shared = bits @ bits.T
    counts = bits.sum(axis=1)
    return shared / (counts[:, None] + counts[None, :] - shared)


def _random_drug_grid(
    n_lines: int = 6,
    n_drugs: int = 8,
    n_genes: int = 5,
    width: int = 3,
    seed: int = 0,
    untested: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(description, similarity, delta0, tested)``: random answers with a share left untested
    (set to 0 in ``delta0``), a random line description, and the Tanimoto similarity of random
    24-bit fingerprints (bit 0 always set, so none is empty)."""
    rng = np.random.default_rng(seed)
    description = rng.standard_normal((n_lines, width))
    fingerprints = rng.random((n_drugs, 24)) < 0.4
    fingerprints[:, 0] = True
    delta = rng.standard_normal((n_lines, n_drugs, n_genes))
    tested = rng.random(delta.shape) >= untested
    return description, _tanimoto(fingerprints), np.where(tested, delta, 0.0), tested


def _planted_drug_grid(
    n_lines: int,
    n_drugs: int,
    n_genes: int,
    width: int,
    signal_width: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(description, similarity, delta0, tested)`` drawn from the leave-one-drug-out model's own
    kernel.

    Drugs come in six families: a random 64-bit scaffold with 10% of bits flipped per drug (bit
    0 always set); ``similarity`` is their Tanimoto similarity. Every gene's change is a line
    baseline, plus a drug effect with covariance ``similarity`` across drugs, plus
    ``signal_width`` more such effects, each weighted by one of the line's first
    ``signal_width`` standardized description columns over ``sqrt(signal_width)`` (so the
    weights' inner products are ``linear_kernel`` of those columns), plus Gaussian noise of SD
    ``noise``. 5% of entries are untested.
    """
    rng = np.random.default_rng(seed)
    description = rng.standard_normal((n_lines, width))
    scaffolds = rng.random((6, 64)) < 0.3
    fingerprints = scaffolds[np.arange(n_drugs) % 6] ^ (rng.random((n_drugs, 64)) < 0.1)
    fingerprints[:, 0] = True
    similarity = _tanimoto(fingerprints)
    values, vectors = np.linalg.eigh(similarity)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    draws = rng.standard_normal((signal_width + 1, n_drugs, n_genes))
    effects = np.einsum("dh,phg->pdg", root, draws)
    columns = standardize(description)[:, :signal_width] / np.sqrt(signal_width)
    line_weights = np.hstack([np.ones((n_lines, 1)), columns])
    baseline = rng.standard_normal((n_lines, 1, n_genes))
    signal = baseline + np.einsum("lp,pdg->ldg", line_weights, effects)
    delta = signal + noise * rng.standard_normal((n_lines, n_drugs, n_genes))
    tested = rng.random(delta.shape) >= 0.05
    return description, similarity, np.where(tested, delta, 0.0), tested


# ==============================================================================================
# Leave one drug out: the prediction and the closed-form tuning loss


@pytest.mark.parametrize("line_term", [True, False], ids=["description", "chemistry_only"])
@pytest.mark.parametrize("lam", [1e-2, 1.0, 30.0])
def test_ridge_lodo_prediction_equals_dense_solve(lam: float, line_term: bool) -> None:
    """At a fixed penalty on a 6-line x 8-drug x 5-gene grid, the eigenbasis prediction is the
    dense one: ``np.linalg.solve`` on the full pair kernel ``np.kron(1 + K_line, T_train)`` in
    (line, drug) order for the dual coefficients, and the hidden drug predicted as each line's
    training mean plus ``(1 + K_line) (alpha t)`` with ``t = T[held, train]``. With a line
    description and without one (chemistry only, ``K_line = 0``)."""
    description, similarity, delta0, tested = _random_drug_grid()
    held = 2
    n_lines, n_drugs, n_genes = delta0.shape
    line_kernel = linear_kernel(description) if line_term else np.zeros((n_lines, n_lines))
    train = np.flatnonzero(np.arange(n_drugs) != held)
    means = delta0[:, train].mean(axis=1)
    departures = delta0[:, train] - means[:, None, :]
    line_factor = 1.0 + line_kernel
    pair_kernel = np.kron(line_factor, similarity[np.ix_(train, train)])
    system = pair_kernel + lam * np.eye(pair_kernel.shape[0])
    alpha = np.linalg.solve(system, departures.reshape(-1, n_genes)).reshape(departures.shape)
    expected = means + line_factor @ np.einsum("ldg,d->lg", alpha, similarity[held, train])

    fit = ridge_lodo(line_kernel, similarity, delta0, tested, held, lambdas=np.array([lam]))
    assert fit.prediction.shape == (n_lines, n_genes)
    assert (fit.lam, fit.k, fit.at_edge) == (lam, None, True)
    np.testing.assert_allclose(fit.prediction, expected, rtol=1e-10, atol=1e-10)


def _brute_force_lodo_losses(
    line_kernel: np.ndarray,
    similarity: np.ndarray,
    delta0: np.ndarray,
    tested: np.ndarray,
    held: int,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Tuning loss per penalty by refitting everything with each training drug left out.

    For each left-out training drug j (invariant 6): each line's mean is recomputed over the
    remaining training drugs, the departures from it are formed, ridge is solved directly on
    the dense pair kernel ``np.kron(1 + K_line, T[rest][:, rest])`` in (line, drug) order
    (``np.linalg.solve`` of ``(K + λI) alpha = R`` -- no eigendecomposition, no hat matrix, no
    rotation), and drug j is predicted in every line as its re-estimated means plus
    ``(1 + K_line) (alpha T[j, rest])``. Batched over penalties and left-out drugs at once.
    """
    n_lines, n_drugs, n_genes = delta0.shape
    train = np.flatnonzero(np.arange(n_drugs) != held)
    n_train = train.size
    answers = delta0[:, train]
    measured = tested[:, train]
    line_factor = 1.0 + line_kernel
    train_similarity = similarity[np.ix_(train, train)]
    # row j: every training position except j
    rest = np.tile(np.arange(n_train), (n_train, 1))[~np.eye(n_train, dtype=bool)]
    rest = rest.reshape(n_train, n_train - 1)
    rest_answers = answers[:, rest]  # [L, j, Dt-1, G]
    rest_means = rest_answers.mean(axis=2)  # [L, j, G]
    rest_departures = rest_answers - rest_means[:, :, None, :]
    rest_similarity = train_similarity[rest[:, :, None], rest[:, None, :]]  # [j, Dt-1, Dt-1]
    n_pairs = n_lines * (n_train - 1)
    # np.kron(line_factor, rest_similarity[j]) for every j at once
    pair_kernel = np.einsum("lm,jab->jlamb", line_factor, rest_similarity)
    pair_kernel = pair_kernel.reshape(n_train, n_pairs, n_pairs)
    assert np.array_equal(pair_kernel[1], np.kron(line_factor, rest_similarity[1]))
    targets = np.moveaxis(rest_departures, 1, 0).reshape(n_train, n_pairs, n_genes)
    system = pair_kernel[None] + lambdas[:, None, None, None] * np.eye(n_pairs)
    alpha = np.linalg.solve(system, targets[None])
    alpha = alpha.reshape(lambdas.size, n_train, n_lines, n_train - 1, n_genes)
    left_out_similarity = np.take_along_axis(train_similarity, rest, axis=1)  # T[j, rest_j]
    ridge_term = np.einsum("lm,sjmag,ja->sljg", line_factor, alpha, left_out_similarity)
    errors = answers[None] - (rest_means[None] + ridge_term)
    return (errors**2 * measured[None]).sum(axis=(1, 2, 3)) / measured.sum()


@pytest.mark.parametrize("width", [2, 10, None], ids=["narrow", "wide", "chemistry_only"])
def test_ridge_lodo_block_loo_equals_refits(width: int | None) -> None:
    """Invariant 6: the closed-form leave-one-drug-out loss at every penalty equals the loss from
    refitting with each training drug left out of both the ridge fit and the line means, to
    1e-10 relative -- for a line description narrower than the lines (rank-deficient line
    factor), one wider (full rank), and chemistry only (a rank-one line factor)."""
    description, similarity, delta0, tested = _random_drug_grid(width=width or 3)
    held = 5
    n_lines = delta0.shape[0]
    line_kernel = np.zeros((n_lines, n_lines)) if width is None else linear_kernel(description)
    brute = _brute_force_lodo_losses(
        line_kernel, similarity, delta0, tested, held, np.asarray(LAMBDAS)
    )
    closed = np.array(
        [
            ridge_lodo(
                line_kernel, similarity, delta0, tested, held, lambdas=np.array([lam])
            ).loss_min
            for lam in LAMBDAS
        ]
    )
    np.testing.assert_allclose(closed, brute, rtol=1e-10, atol=0.0)

    fit = ridge_lodo(line_kernel, similarity, delta0, tested, held)
    assert fit.lam == LAMBDAS[int(np.argmin(brute))]
    assert fit.loss_min == pytest.approx(float(brute.min()), rel=1e-10)


def test_ridge_lodo_k_is_ridge_lodo_at_its_chosen_setting() -> None:
    """``ridge_lodo_k`` fits every candidate k in one pass over genes. Its choices, losses and
    prediction are those of ``ridge_lodo`` run on each k's kernel on its own (whose loss the
    refit test pins): at each single penalty, the k with the smallest loss (ties to the smaller
    k) and that loss; over the full grid, the joint argmin and the prediction under it."""
    description, similarity, delta0, tested = _planted_drug_grid(16, 24, 100, 10, 5, 1.0, 0)
    held = 0
    by_k = {k: description[:, :k] for k in (2, 5, 10)}
    ks = sorted(by_k)
    table = np.array(
        [
            ridge_lodo(
                linear_kernel(by_k[k]), similarity, delta0, tested, held, lambdas=np.array([lam])
            ).loss_min
            for k, lam in itertools.product(ks, LAMBDAS)
        ]
    ).reshape(len(ks), LAMBDAS.size)
    per_penalty = [
        ridge_lodo_k(by_k, similarity, delta0, tested, held, lambdas=np.array([lam]))
        for lam in LAMBDAS
    ]
    assert [fit.k for fit in per_penalty] == [ks[i] for i in np.argmin(table, axis=0)]
    assert len({fit.k for fit in per_penalty}) > 1  # the batch's choice moves between kernels
    np.testing.assert_allclose(
        [fit.loss_min for fit in per_penalty], table.min(axis=0), rtol=1e-12, atol=0.0
    )

    fit = ridge_lodo_k(by_k, similarity, delta0, tested, held)
    k_index, lam_index = np.unravel_index(int(np.argmin(table)), table.shape)
    assert (fit.k, fit.lam) == (ks[k_index], LAMBDAS[lam_index])
    assert fit.loss_min == pytest.approx(float(table.min()), rel=1e-12)
    assert fit.k is not None and fit.lam is not None
    single = ridge_lodo(
        linear_kernel(by_k[fit.k]), similarity, delta0, tested, held, lambdas=np.array([fit.lam])
    )
    np.testing.assert_allclose(fit.prediction, single.prediction, rtol=1e-12, atol=1e-14)


def test_lodo_tuning_does_not_reward_width_on_noise() -> None:
    """Pure-noise answers (i.i.d. normal, variance 1, independent of a full-rank 400-column
    line description and of the drugs' similarity): tuning must not select the smallest penalty,
    and the loss at the smallest penalty is above the loss at the largest.

    What the math guarantees. Drug j's leave-one-out prediction in line l weights the other
    training drugs' answers, in every line, by c: the re-estimated line mean plus a ridge term
    on departures from the re-estimated means. The weights sum to 1 over line l's own entries
    and to 0 over each other line's. With noise independent of everything, the expected squared
    error is 1 + |c|², smallest, 1 + 1/(Dt-1), for the plain line mean, which the ridge term
    vanishes toward as the penalty grows; at the smallest penalty ridge nearly interpolates and
    the weights spread. So in expectation the loss at the smallest penalty is above the loss at
    the largest, and both are above 1.

    Unlike leaving a line out, holding each line's mean fixed at its full training value does
    not tip this choice to the smallest penalty here (the similarity is not centred, so no
    direction hands the answer straight back): it shows in the level instead. The fixed mean
    contains drug j's own answer, so the error at a large penalty is about x_j - a, with
    expectation 1 - 1/Dt = 0.966 at Dt = 29, below 1. Re-estimated, the loss at the largest
    penalty is 1.032-1.039 across seeds 17-22 (expectation 1 + 1/28 = 1.036). A mean of
    116,000 squared unit-normal errors has a standard error of about sqrt(2/116,000) = 0.004,
    so 1 sits more than 7 standard errors below. Hence the last assertion.
    """
    rng = np.random.default_rng(17)
    n_lines, n_drugs = 20, 30
    delta0 = rng.standard_normal((n_lines, n_drugs, 200))
    tested = np.ones(delta0.shape, dtype=bool)
    line_kernel = linear_kernel(rng.standard_normal((n_lines, 400)))
    fingerprints = rng.random((n_drugs, 64)) < 0.3
    fingerprints[:, 0] = True
    similarity = _tanimoto(fingerprints)
    held = 0

    fit = ridge_lodo(line_kernel, similarity, delta0, tested, held)
    assert fit.lam != LAMBDAS[0]
    at_smallest = ridge_lodo(
        line_kernel, similarity, delta0, tested, held, lambdas=LAMBDAS[:1]
    ).loss_min
    at_largest = ridge_lodo(
        line_kernel, similarity, delta0, tested, held, lambdas=LAMBDAS[-1:]
    ).loss_min
    assert at_smallest > at_largest
    assert at_largest > 1.0


# ==============================================================================================
# Leave one drug out: invariant 1, the reference, blocks, at_edge, a planted response


def test_lodo_ignores_held_out_answers() -> None:
    description, similarity, delta0, tested = _random_drug_grid(
        n_lines=7, n_drugs=9, n_genes=12, width=5
    )
    held = 4
    rng = np.random.default_rng(99)
    shape = (7, 12)
    poisoned = delta0.copy()
    poisoned[:, held] = np.where(rng.random(shape) < 0.5, np.nan, 1e6 * rng.standard_normal(shape))
    poisoned_tested = tested.copy()
    poisoned_tested[:, held] = rng.random(shape) < 0.5

    kernel = linear_kernel(description)
    chemistry = np.zeros((7, 7))
    by_k = {k: description[:, :k] for k in (2, 3, 5)}

    _assert_same_fit(
        ridge_lodo(kernel, similarity, delta0, tested, held),
        ridge_lodo(kernel, similarity, poisoned, poisoned_tested, held),
    )
    _assert_same_fit(
        ridge_lodo(chemistry, similarity, delta0, tested, held),
        ridge_lodo(chemistry, similarity, poisoned, poisoned_tested, held),
    )
    _assert_same_fit(
        ridge_lodo_k(by_k, similarity, delta0, tested, held),
        ridge_lodo_k(by_k, similarity, poisoned, poisoned_tested, held),
    )


def test_lodo_identity_similarity_gives_line_means() -> None:
    """With ``K_line = 0`` and ``T`` the identity, the hidden drug resembles no training drug, so
    its predicted departure is zero and the prediction is each line's mean over the training
    drugs, whatever penalty tuning chooses."""
    _, _, delta0, tested = _random_drug_grid(n_drugs=9, n_genes=20)
    held = 3
    fit = ridge_lodo(np.zeros((6, 6)), np.eye(9), delta0, tested, held)
    train = np.flatnonzero(np.arange(9) != held)
    np.testing.assert_allclose(
        fit.prediction, delta0[:, train].mean(axis=1), rtol=1e-12, atol=1e-15
    )


def test_gene_blocks_respect_the_budget() -> None:
    assert gene_blocks(5, 10, 25) == [(0, 2), (2, 4), (4, 5)]
    assert gene_blocks(5, 10, 10**9) == [(0, 5)]
    with pytest.raises(ValueError, match=r"one gene needs 11 bytes.*max_bytes budget of 10"):
        gene_blocks(5, 11, 10)


@pytest.mark.parametrize("genes_per_block", [1, 2])
@pytest.mark.parametrize("model", ["ridge", "chemistry_only", "ridge_k"])
def test_chunked_lodo_fits_agree(
    model: str, genes_per_block: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each fit at the default budget (one gene block per pass) agrees with the same fit at a
    budget whose tuning pass holds ``genes_per_block`` genes per block: the same penalty, k and
    edge flag, and loss and prediction to 1e-12 relative. Not bit for bit: the per-block loss
    sums and matrix products change shape with the block, and BLAS may round them differently.

    A spy on ``gene_blocks`` reads the tuning pass's per-gene size and the budget left after its
    operators, and checks that both passes really split (three penalties keep the operators
    small next to a gene's working arrays, so the prediction pass splits too)."""
    description, similarity, delta0, tested = _random_drug_grid(
        n_lines=5, n_drugs=6, n_genes=30, width=4
    )
    held = 2
    n_genes = delta0.shape[2]
    lambdas = LAMBDAS[::6]
    kernel = linear_kernel(description)
    chemistry = np.zeros((5, 5))
    by_k = {k: description[:, :k] for k in (2, 3, 4)}
    fits: dict[str, Callable[[int], Fit]] = {
        "ridge": lambda budget: ridge_lodo(
            kernel, similarity, delta0, tested, held, lambdas, max_bytes=budget
        ),
        "chemistry_only": lambda budget: ridge_lodo(
            chemistry, similarity, delta0, tested, held, lambdas, max_bytes=budget
        ),
        "ridge_k": lambda budget: ridge_lodo_k(
            by_k, similarity, delta0, tested, held, lambdas, max_bytes=budget
        ),
    }
    passes: list[tuple[int, int, int]] = []  # (bytes per gene, budget, number of blocks)

    def recording_blocks(n: int, bytes_per_gene: int, budget: int) -> list[tuple[int, int]]:
        blocks = gene_blocks(n, bytes_per_gene, budget)
        passes.append((bytes_per_gene, budget, len(blocks)))
        return blocks

    monkeypatch.setattr(models, "gene_blocks", recording_blocks)
    whole = fits[model](models.MAX_BYTES)
    (tuning_bytes, tuning_budget, _), _ = passes
    assert [n_blocks for _, _, n_blocks in passes] == [1, 1]
    operators = models.MAX_BYTES - tuning_budget
    passes.clear()
    chunked = fits[model](operators + genes_per_block * tuning_bytes)
    (_, _, tuning_blocks), (_, _, prediction_blocks) = passes
    assert tuning_blocks == -(-n_genes // genes_per_block)
    assert prediction_blocks > 1

    assert (chunked.lam, chunked.k, chunked.at_edge) == (whole.lam, whole.k, whole.at_edge)
    assert chunked.loss_min == pytest.approx(whole.loss_min, rel=1e-12)
    np.testing.assert_allclose(chunked.prediction, whole.prediction, rtol=1e-12, atol=1e-14)


def test_lodo_at_edge_flags_a_choice_on_the_edge_of_its_grid() -> None:
    """A planted response on 5 of 10 description columns, with noise: tuning picks an interior
    penalty, and k = 5 of {2, 5, 10}. The same data with the penalty grid cut so that choice
    sits on its bottom end, then on its top end, is flagged both times, as is k = 5 of {5, 10}
    and of {2, 5}; the choice itself does not change."""
    description, similarity, delta0, tested = _planted_drug_grid(16, 24, 100, 10, 5, 1.0, 0)
    held = 0
    kernel = linear_kernel(description[:, :5])

    interior = ridge_lodo(kernel, similarity, delta0, tested, held)
    assert interior.lam is not None
    assert LAMBDAS[0] < interior.lam < LAMBDAS[-1]
    assert not interior.at_edge
    bottom = ridge_lodo(
        kernel, similarity, delta0, tested, held, lambdas=LAMBDAS[interior.lam <= LAMBDAS]
    )
    assert (bottom.lam, bottom.at_edge) == (interior.lam, True)
    top = ridge_lodo(
        kernel, similarity, delta0, tested, held, lambdas=LAMBDAS[interior.lam >= LAMBDAS]
    )
    assert (top.lam, top.at_edge) == (interior.lam, True)

    def ridge_over(ks: tuple[int, ...]) -> tuple[int | None, float | None, bool]:
        fit = ridge_lodo_k({k: description[:, :k] for k in ks}, similarity, delta0, tested, held)
        return fit.k, fit.lam, fit.at_edge

    assert ridge_over((2, 5, 10)) == (5, interior.lam, False)
    assert ridge_over((5, 10)) == (5, interior.lam, True)
    assert ridge_over((2, 5)) == (5, interior.lam, True)


@pytest.mark.known_answer
@pytest.mark.parametrize("held", [0, 1, 2])
def test_ridge_lodo_beats_chemistry_only_on_a_planted_response(held: int) -> None:
    """Drug effects smooth in fingerprint similarity (covariance ``T`` across drugs, six drug
    families), four of them weighted by the line description's first four standardized
    columns, noise SD 0.5: ridge with those columns has a smaller squared error on the hidden
    drug's tested entries than chemistry only, which gives every line the same departure. The
    full known-answer control, placed relative to the MDE, is task 9's."""
    description, similarity, delta0, tested = _planted_drug_grid(20, 24, 200, 8, 4, 0.5, 0)
    n_lines = delta0.shape[0]
    ridge = ridge_lodo(linear_kernel(description[:, :4]), similarity, delta0, tested, held)
    chemistry = ridge_lodo(np.zeros((n_lines, n_lines)), similarity, delta0, tested, held)
    measured = tested[:, held]
    ridge_error = float(((ridge.prediction - delta0[:, held]) ** 2 * measured).sum())
    chemistry_error = float(((chemistry.prediction - delta0[:, held]) ** 2 * measured).sum())
    assert ridge_error < chemistry_error


def test_lodo_models_reject_malformed_inputs() -> None:
    description, similarity, delta0, tested = _random_drug_grid()
    kernel = linear_kernel(description)
    with pytest.raises(ValueError, match="held drug"):
        ridge_lodo(kernel, similarity, delta0, tested, 8)
    with pytest.raises(ValueError, match=r"T must be \[8, 8\]"):
        ridge_lodo(kernel, similarity[:7, :7], delta0, tested, 0)
    with pytest.raises(ValueError, match=r"K_line must be \[6, 6\]"):
        ridge_lodo(kernel[:5, :5], similarity, delta0, tested, 0)
    with pytest.raises(ValueError, match="descriptions"):
        ridge_lodo_k({4: description}, similarity, delta0, tested, 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        ridge_lodo(kernel, similarity, delta0, tested, 0, lambdas=np.array([1.0, 0.1]))
    with pytest.raises(ValueError, match="exceed the max_bytes budget of 1,000"):
        ridge_lodo(kernel, similarity, delta0, tested, 0, max_bytes=1_000)
    with pytest.raises(ValueError, match="no tested entries"):
        ridge_lodo(kernel, similarity, delta0, np.zeros_like(tested), 0)
    with pytest.raises(ValueError, match="at least 3 drugs"):
        ridge_lodo(kernel, similarity[:2, :2], delta0[:, :2], tested[:, :2], 0)
