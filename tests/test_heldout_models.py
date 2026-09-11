"""Rung 1's models when a line is hidden (task 7): drug average, ridge with closed-form
leave-one-line-out tuning, ridge over a choice of component count, and nearest lines.

Every test runs the real functions from ``fmharness.heldout.models`` on small synthetic grids.
Exactness is checked against independent computations: scikit-learn's primal ridge for the
prediction, brute-force refits for the closed-form tuning loss (invariant 6), and an explicit
Python ranking for nearest lines. Invariant 1 is checked by poisoning the hidden line's answers.
"""

# scikit-learn ships no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *its* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

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
    nearest_lines_lolo,
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
    """Tuning loss per penalty by refitting kernel ridge with each training line left out.

    Departures are from the drug average over all training lines, held fixed (invariant 6).
    Each refit solves ``(K_rest + λI) alpha = R_rest`` directly -- no eigendecomposition, no hat
    matrix -- batched over penalties and left-out lines at once.
    """
    train = _training_lines(kernel.shape[0], held)
    n_train = train.size
    average = delta0[train].mean(axis=0)
    departures = (delta0[train] - average).reshape(n_train, -1)
    measured = tested[train].reshape(n_train, -1)
    k_train = kernel[np.ix_(train, train)]
    # row j: every training position except j
    rest = np.tile(np.arange(n_train), (n_train, 1))[~np.eye(n_train, dtype=bool)]
    rest = rest.reshape(n_train, n_train - 1)
    k_rest = k_train[rest[:, :, None], rest[:, None, :]]
    k_left_out = np.take_along_axis(k_train, rest, axis=1)
    system = k_rest[None] + lambdas[:, None, None, None] * np.eye(n_train - 1)
    alpha = np.linalg.solve(system, departures[rest][None])
    refit_prediction = np.einsum("jm,ljmn->ljn", k_left_out, alpha)
    errors = departures[None] - refit_prediction
    return (errors**2 * measured[None]).sum(axis=(1, 2)) / measured.sum()


@pytest.mark.parametrize("width", [5, 30])
def test_ridge_lolo_closed_form_loo_equals_refits(width: int) -> None:
    """Invariant 6: the closed-form leave-one-line-out loss at every penalty equals the loss
    from refitting with each training line left out, to 1e-10 relative -- for a description
    narrower than the training lines (rank-deficient kernel) and one wider (full rank)."""
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
    assert drug_blocks(7, 10, 1) == [(i, i + 1) for i in range(7)]
    assert drug_blocks(7, 10, 10**9) == [(0, 7)]


@pytest.mark.parametrize("max_bytes", [1, 100_000])
def test_chunked_fits_are_bit_identical(max_bytes: int, monkeypatch: pytest.MonkeyPatch) -> None:
    description, delta0, tested = _random_grid(n_drugs=7)
    held = 3
    kernel = linear_kernel(description)
    by_k = {k: description[:, :k] for k in (2, 3, 5)}
    similarity = similarity_from_description(description)
    whole = [
        ridge_lolo(kernel, delta0, tested, held),
        ridge_lolo_k(by_k, delta0, tested, held),
        nearest_lines_lolo(similarity, delta0, tested, held, ks=(2, 3, 5)),
    ]

    block_counts: list[int] = []

    def counting_blocks(n_drugs: int, bytes_per_drug: int, budget: int) -> list[tuple[int, int]]:
        blocks = drug_blocks(n_drugs, bytes_per_drug, budget)
        block_counts.append(len(blocks))
        return blocks

    monkeypatch.setattr(models, "drug_blocks", counting_blocks)
    chunked = [
        ridge_lolo(kernel, delta0, tested, held, max_bytes=max_bytes),
        ridge_lolo_k(by_k, delta0, tested, held, max_bytes=max_bytes),
        nearest_lines_lolo(similarity, delta0, tested, held, ks=(2, 3, 5), max_bytes=max_bytes),
    ]
    assert max(block_counts) > 1  # the budget really split the drugs
    if max_bytes == 1:
        assert set(block_counts) == {7}
    for a, b in zip(whole, chunked, strict=True):
        _assert_same_fit(a, b)


# ==============================================================================================
# at_edge


def test_at_edge_flags_a_choice_on_the_edge_of_its_grid() -> None:
    """A planted response on 5 of 10 description columns, with noise: tuning picks an interior
    penalty, k = 5 of {2, 5, 10}, and k = 3 of {1, 3, 8} nearest lines in clusters of 4. The
    same data with the grid cut so that choice sits on its end is flagged; nothing else changes."""
    description, delta0, tested = _planted_grid(20, 4, 100, 10, 5, noise=1.0, seed=1)
    held = 0
    kernel = linear_kernel(description[:, :5])

    interior = ridge_lolo(kernel, delta0, tested, held)
    assert interior.lam is not None
    assert LAMBDAS[0] < interior.lam < LAMBDAS[-1]
    assert not interior.at_edge
    cut = ridge_lolo(kernel, delta0, tested, held, lambdas=LAMBDAS[interior.lam <= LAMBDAS])
    assert cut.lam == interior.lam
    assert cut.at_edge

    middle_k = ridge_lolo_k({k: description[:, :k] for k in (2, 5, 10)}, delta0, tested, held)
    assert (middle_k.k, middle_k.lam, middle_k.at_edge) == (5, interior.lam, False)
    smallest_k = ridge_lolo_k({k: description[:, :k] for k in (5, 10)}, delta0, tested, held)
    assert (smallest_k.k, smallest_k.lam, smallest_k.at_edge) == (5, interior.lam, True)

    rng = np.random.default_rng(3)
    cluster = np.repeat(np.arange(4), 4)
    clustered = rng.standard_normal((4, 30))[cluster] + 0.3 * rng.standard_normal((16, 30))
    responses = rng.standard_normal((4, 3, 50))[cluster] + 0.5 * rng.standard_normal((16, 3, 50))
    all_tested = np.ones(responses.shape, dtype=bool)
    similarity = similarity_from_description(clustered)
    middle_nn = nearest_lines_lolo(similarity, responses, all_tested, held, ks=(1, 3, 8))
    assert (middle_nn.k, middle_nn.at_edge) == (3, False)
    edge_nn = nearest_lines_lolo(similarity, responses, all_tested, held, ks=(3, 8))
    assert (edge_nn.k, edge_nn.at_edge) == (3, True)


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
