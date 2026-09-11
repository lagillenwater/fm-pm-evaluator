"""Known-answer controls for rung 1's score, null, fit and split steps (design.md section 8,
SPEC rule 4).

Every test plants an answer with ``fmharness.heldout.controls`` and runs the REAL shipped
functions -- ``score_pairs``, ``redraw_estimates`` / ``summarize_redraws`` / ``holm``, the models
of ``fmharness.heldout.models`` -- requiring them to recover it (positive) or to find nothing
(negative). Tolerances are fixed before any test runs (invariant 9): a recovered value passes
within 3 Monte Carlo standard errors of the planted one, the standard error taken from the
synthetic draw itself; a rate over 200 repetitions passes inside the binomial 99% interval.

The fit control follows the recorded departure of 2026-09-11 (decisions.md, "design.md
(execution)"): a plant of fixed strength 0.3 at the design's responding-gene reliability, the
truth oracle's gain asserted to be at least twice its MDE (the ratio is in the failure message
and the task report), and the stand-in requirement split by scheme.
"""

# pandas and scipy ship no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *their* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pytest
from scipy import stats

from fmharness.heldout import Scheme
from fmharness.heldout.answers import Answers, scoreable
from fmharness.heldout.chemistry import tanimoto
from fmharness.heldout.comparisons import (
    N_DRAWS,
    RedrawSummary,
    holm,
    redraw_estimates,
    summarize_redraws,
)
from fmharness.heldout.controls import (
    leaky_lodo_prediction,
    leaky_lolo_prediction,
    lodo_signal_variance,
    lolo_signal_variance,
    noise_for_reliability,
    planted_reliability_pool,
    planted_signature_grid,
    synthetic_answers,
    synthetic_lodo_grid,
    synthetic_lolo_grid,
)
from fmharness.heldout.descriptions import linear_kernel, random_stand_in
from fmharness.heldout.models import (
    LAMBDAS,
    drug_average,
    nearest_lines_lolo,
    ridge_lodo,
    ridge_lolo,
    similarity_from_description,
)
from fmharness.heldout.scoring import GENE_SETS, score_pairs

pytestmark = pytest.mark.known_answer

#: The design's full-data reliabilities (section 6): responding genes, all genes.
R_RESPONDING = 0.7353
R_ALL = 0.1503

#: The fit control's planted strength (recorded departure, 2026-09-11).
STRENGTH = 0.3

#: Repetitions behind every rate, and the two-sided level a detection is read at.
REPETITIONS = 200
ALPHA = 0.05

#: The redraw seed every contrast in this file uses.
REDRAW_SEED = 20260911


# ==============================================================================================
# Shared: scoring a whole synthetic grid, and a contrast's redraws


def _grid_scores(prediction: np.ndarray, answers: Answers) -> np.ndarray:
    """Every pair's score through ``score_pairs``, as an ``[L, D]`` array (responding genes; in a
    synthetic answer every finite gene is responding, so both gene sets agree)."""
    n_lines, n_drugs, n_genes = prediction.shape
    lines, drugs = np.divmod(np.arange(n_lines * n_drugs), n_drugs)
    frame = score_pairs(
        prediction.reshape(-1, n_genes), lines, drugs, answers, scoreable(answers, ())
    )
    scores = frame.loc[frame["gene_set"] == "responding", "r"].to_numpy(dtype=np.float64)
    assert scores.size == n_lines * n_drugs
    return scores.reshape(n_lines, n_drugs)


def _contrast(scores_a: np.ndarray, scores_b: np.ndarray, scheme: Scheme) -> RedrawSummary:
    """Mean over pairs of ``scores_a - scores_b`` (both ``[L, D]``), redrawn over the held-out
    unit of ``scheme``: lines when a line is hidden, drugs when a drug is hidden."""
    diffs = scores_a - scores_b
    lines, drugs = np.indices(diffs.shape)
    units, n_units = (lines, diffs.shape[0]) if scheme == "lolo" else (drugs, diffs.shape[1])
    draws = redraw_estimates(diffs.ravel(), units.ravel(), n_units, N_DRAWS, REDRAW_SEED)
    return summarize_redraws(float(diffs.mean()), draws)


def _describe(summary: RedrawSummary) -> str:
    return (
        f"gain {summary['estimate']:.3g}, MDE {summary['mde']:.3g}, sd {summary['sd']:.3g}, "
        f"p {summary['p']:.3g}"
    )


# ==============================================================================================
# score: the true change scores √R, a second measurement R, an unrelated prediction 0


def _pair_indices(n_lines: int, n_drugs: int) -> tuple[np.ndarray, np.ndarray]:
    lines, drugs = np.divmod(np.arange(n_lines * n_drugs), n_drugs)
    return lines, drugs


def _within_three_se(scores: np.ndarray, planted: float) -> tuple[bool, str]:
    """Whether the mean of ``scores`` is within 3 Monte Carlo SEs (their SD over √n) of
    ``planted``, and a message saying by how much."""
    se = float(scores.std(ddof=1) / np.sqrt(scores.size))
    mean = float(scores.mean())
    return abs(mean - planted) <= 3 * se, f"mean {mean:.5f} vs planted {planted:.5f}, SE {se:.2g}"


@pytest.mark.step_score
@pytest.mark.parametrize("reliability", [R_RESPONDING, R_ALL], ids=["responding", "all"])
def test_score_square_root_known_answer(reliability: float) -> None:
    """600 pairs x 2,000 genes at reliability R, a random half of each pair's genes responding.
    Through ``score_pairs``, on both gene sets: the true change scores √R against the
    measurement, and a second, independent measurement scores R.

    Why this is the right target. A measurement is X = T + e with R = var(T) / var(X). Across
    genes, corr(T, X) = √R and corr(X, X') = R for two measurements with independent noise.
    A pair's r is biased toward 0 by about c(1 - c²)/(2n) for a correlation c over n genes: at
    most 1.1e-4 here (c = 0.8575, n = 1,000), a tenth of the tolerance of 3 SEs (about 1.0e-3)."""
    n_lines, n_drugs, n_genes = 20, 30, 2000
    truth, measurement, second = planted_reliability_pool(
        reliability, n_lines * n_drugs, n_genes, seed=41
    )
    responding = np.random.default_rng(42).random((n_lines, n_drugs, n_genes)) < 0.5
    answers = synthetic_answers(measurement.reshape(n_lines, n_drugs, n_genes), responding)
    lines, drugs = _pair_indices(n_lines, n_drugs)
    scored = {
        "truth": score_pairs(truth, lines, drugs, answers, scoreable(answers, ())),
        "second": score_pairs(second, lines, drugs, answers, scoreable(answers, ())),
    }
    planted = {"truth": float(np.sqrt(reliability)), "second": reliability}

    for name, gene_set in itertools.product(scored, GENE_SETS):
        frame = scored[name]
        scores = frame.loc[frame["gene_set"] == gene_set, "r"].to_numpy(dtype=np.float64)
        assert scores.size == n_lines * n_drugs
        ok, message = _within_three_se(scores, planted[name])
        assert ok, f"{name} on {gene_set} genes: {message}"


@pytest.mark.step_score
def test_score_independent_prediction_is_zero() -> None:
    """A prediction drawn independently of the answer scores 0 within 3 SEs, on both gene sets."""
    n_lines, n_drugs, n_genes = 20, 30, 2000
    _, measurement, _ = planted_reliability_pool(R_RESPONDING, n_lines * n_drugs, n_genes, seed=43)
    responding = np.random.default_rng(44).random((n_lines, n_drugs, n_genes)) < 0.5
    answers = synthetic_answers(measurement.reshape(n_lines, n_drugs, n_genes), responding)
    lines, drugs = _pair_indices(n_lines, n_drugs)
    unrelated = np.random.default_rng(45).standard_normal((n_lines * n_drugs, n_genes))
    frame = score_pairs(unrelated, lines, drugs, answers, scoreable(answers, ()))

    for gene_set in GENE_SETS:
        scores = frame.loc[frame["gene_set"] == gene_set, "r"].to_numpy(dtype=np.float64)
        ok, message = _within_three_se(scores, 0.0)
        assert ok, f"{gene_set} genes: {message}"


# ==============================================================================================
# null: a contrast at its MDE is detected 80% of the time; nothing planted, at most 5%


def _unit_structured_diffs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair differences for one comparison when a line is hidden, with nothing planted:
    50 lines holding 98-107 pairs each (as masks leave them), a line effect of SD 0.02 shared by
    the line's pairs, and pair noise of SD 0.05. Returns ``(diffs, line_index)``."""
    n_units = 50
    counts = 107 - rng.integers(0, 10, n_units)
    units = np.repeat(np.arange(n_units), counts)
    diffs = rng.normal(0.0, 0.02, n_units)[units] + rng.normal(0.0, 0.05, units.size)
    return diffs, units


@dataclass(frozen=True)
class NullRepetitions:
    """p-values over ``REPETITIONS`` synthetic comparisons: with nothing planted, and with the
    same differences shifted by the MDE that repetition's null redraws estimate."""

    p_null: np.ndarray
    p_planted: np.ndarray


@pytest.fixture(scope="module")
def null_repetitions() -> NullRepetitions:
    p_null = np.empty(REPETITIONS)
    p_planted = np.empty(REPETITIONS)
    for repetition in range(REPETITIONS):
        diffs, units = _unit_structured_diffs(np.random.default_rng([61, repetition]))
        null = summarize_redraws(
            float(diffs.mean()), redraw_estimates(diffs, units, 50, N_DRAWS, repetition)
        )
        shifted = diffs + null["mde"]
        planted = summarize_redraws(
            float(shifted.mean()), redraw_estimates(shifted, units, 50, N_DRAWS, repetition)
        )
        p_null[repetition], p_planted[repetition] = null["p"], planted["p"]
    return NullRepetitions(p_null=p_null, p_planted=p_planted)


@pytest.mark.step_null
def test_null_detection_at_mde(null_repetitions: NullRepetitions) -> None:
    """A contrast planted at its MDE, 2.8 x the null redraw SD, is detected (p < 0.05) in a
    share of 200 repetitions inside the binomial 99% interval around 0.80.

    Why 0.80. The estimate is the null mean plus the MDE; the redraws spread about it with the
    estimate's standard error s, so p < 0.05 when the estimate clears 1.96 s, i.e. when the null
    mean exceeds (1.96 - 2.8) s = -0.84 s: probability 0.80."""
    detected = int(np.count_nonzero(null_repetitions.p_planted < ALPHA))
    low, high = stats.binom.interval(0.99, REPETITIONS, 0.80)
    assert low <= detected <= high, f"{detected}/{REPETITIONS} detected; 99% interval {low}-{high}"


@pytest.mark.step_null
def test_null_false_positive_rate(null_repetitions: NullRepetitions) -> None:
    """With nothing planted, p < 0.05 in no more of 200 repetitions than the top of the binomial
    99% interval around 0.05."""
    false_positives = int(np.count_nonzero(null_repetitions.p_null < ALPHA))
    _, high = stats.binom.interval(0.99, REPETITIONS, ALPHA)
    assert false_positives <= high, f"{false_positives}/{REPETITIONS} false positives; limit {high}"


@pytest.mark.step_null
def test_holm_controls_familywise() -> None:
    """Six null comparisons per repetition (the leave-one-line-out family), p from the real
    redraws, adjusted by ``holm``: in no more of 200 repetitions than the top of the binomial
    99% interval around 0.05 does any adjusted p fall below 0.05 (Holm holds the familywise
    rate at or below 0.05 under any dependence)."""
    family = 6
    raw = np.empty(REPETITIONS * family)
    for index in range(raw.size):
        diffs, units = _unit_structured_diffs(np.random.default_rng([62, index]))
        draws = redraw_estimates(diffs, units, 50, N_DRAWS, index)
        raw[index] = summarize_redraws(float(diffs.mean()), draws)["p"]
    adjusted = np.vstack([holm(row) for row in raw.reshape(REPETITIONS, family)])
    familywise = int(np.count_nonzero((adjusted < ALPHA).any(axis=1)))
    _, high = stats.binom.interval(0.99, REPETITIONS, ALPHA)
    assert familywise <= high, f"{familywise}/{REPETITIONS} families with a rejection; limit {high}"


# ==============================================================================================
# fit: a planted line- or drug-specific response is recovered; nothing planted, nothing gained


@dataclass(frozen=True)
class Rounds:
    """Every round's prediction per model, stacked into ``[L, D, G]``, and each ridge model's
    chosen penalty per round."""

    predictions: dict[str, np.ndarray]
    lambdas: dict[str, np.ndarray]


def _lolo_rounds(
    delta: np.ndarray,
    kernels: dict[str, np.ndarray],
    similarity: np.ndarray | None = None,
) -> Rounds:
    """All leave-one-line-out rounds: the drug average, ridge on each kernel, and (when a
    similarity is given) nearest lines."""
    n_lines = delta.shape[0]
    tested = np.ones(delta.shape, dtype=bool)
    predictions = {name: np.empty(delta.shape) for name in (*kernels, "drug_average")}
    lambdas = {name: np.empty(n_lines) for name in kernels}
    if similarity is not None:
        predictions["nearest_lines"] = np.empty(delta.shape)
    for held in range(n_lines):
        train = np.flatnonzero(np.arange(n_lines) != held)
        predictions["drug_average"][held] = drug_average(delta, train)
        fits = {name: ridge_lolo(kernel, delta, tested, held) for name, kernel in kernels.items()}
        for name, fit in fits.items():
            predictions[name][held] = fit.prediction
            lambdas[name][held] = fit.lam
        if similarity is not None:
            predictions["nearest_lines"][held] = nearest_lines_lolo(
                similarity, delta, tested, held
            ).prediction
    return Rounds(predictions=predictions, lambdas=lambdas)


def _lodo_rounds(
    delta: np.ndarray, similarity: np.ndarray, kernels: dict[str, np.ndarray]
) -> Rounds:
    """All leave-one-drug-out rounds of ridge on each line kernel (all zeros: chemistry only)."""
    n_drugs = delta.shape[1]
    tested = np.ones(delta.shape, dtype=bool)
    predictions = {name: np.empty(delta.shape) for name in kernels}
    lambdas = {name: np.empty(n_drugs) for name in kernels}
    for held in range(n_drugs):
        fits = {
            name: ridge_lodo(kernel, similarity, delta, tested, held)
            for name, kernel in kernels.items()
        }
        for name, fit in fits.items():
            predictions[name][:, held] = fit.prediction
            lambdas[name][held] = fit.lam
    return Rounds(predictions=predictions, lambdas=lambdas)


@dataclass(frozen=True)
class FitControl:
    """A fit control's contrasts, each against the scheme's reference unless named otherwise."""

    oracle: RedrawSummary
    description: RedrawSummary
    description_minus_oracle: RedrawSummary
    random: RedrawSummary
    description_minus_random: RedrawSummary


def _fit_control(scheme: Scheme, seed: int) -> FitControl:
    """The planted fit control on a grid of the screen's size, 50 lines x 107 drugs x 300 genes
    with a width-20 description, strength ``STRENGTH`` at reliability ``R_RESPONDING``: every
    round of the scheme's reference, ridge on the matching description, and ridge on a random
    stand-in of the same width; the oracle predicts the noiseless change."""
    stand_in = linear_kernel(random_stand_in(50, 20, seed + 1))
    if scheme == "lolo":
        noise = noise_for_reliability(R_RESPONDING, lolo_signal_variance(STRENGTH))
        grid = synthetic_lolo_grid(strength=STRENGTH, noise=noise, seed=seed)
        kernels = {"description": linear_kernel(grid.description), "random": stand_in}
        rounds = _lolo_rounds(grid.delta, kernels)
        reference = "drug_average"
    else:
        noise = noise_for_reliability(R_RESPONDING, lodo_signal_variance(STRENGTH))
        grid = synthetic_lodo_grid(strength=STRENGTH, noise=noise, seed=seed)
        assert grid.fingerprints is not None
        kernels = {
            "chemistry_only": np.zeros((50, 50)),
            "description": linear_kernel(grid.description),
            "random": stand_in,
        }
        rounds = _lodo_rounds(grid.delta, tanimoto(grid.fingerprints), kernels)
        reference = "chemistry_only"
    assert not np.allclose(stand_in, kernels["description"])
    answers = synthetic_answers(grid.delta)
    scores = {name: _grid_scores(p, answers) for name, p in rounds.predictions.items()}
    scores["oracle"] = _grid_scores(grid.truth, answers)
    return FitControl(
        oracle=_contrast(scores["oracle"], scores[reference], scheme),
        description=_contrast(scores["description"], scores[reference], scheme),
        description_minus_oracle=_contrast(scores["description"], scores["oracle"], scheme),
        random=_contrast(scores["random"], scores[reference], scheme),
        description_minus_random=_contrast(scores["description"], scores["random"], scheme),
    )


def _assert_planted_and_recovered(control: FitControl) -> None:
    """The requirements both schemes share: the plant is at least twice the oracle's MDE, the
    matching description's gain is detected, and it does not beat the oracle's gain by more than
    3 standard errors of their difference."""
    oracle, description = control.oracle, control.description
    ratio = oracle["estimate"] / oracle["mde"]
    assert ratio >= 2.0, f"oracle gain is {ratio:.1f} x its MDE ({_describe(oracle)})"
    assert description["estimate"] > 0 and description["p"] < ALPHA, _describe(description)
    bound = oracle["estimate"] + 3 * control.description_minus_oracle["sd"]
    assert description["estimate"] <= bound, (
        f"description {_describe(description)} above the oracle's {_describe(oracle)} "
        f"by more than 3 SE ({control.description_minus_oracle['sd']:.3g})"
    )


@pytest.mark.step_fit
def test_fit_recovers_planted_line_response() -> None:
    """A line hidden: a line-specific response linear in a width-20 description, planted at
    strength 0.3. The oracle's gain over the drug average is at least twice its MDE; ridge on the
    matching description gains detectably (p < 0.05) and no more than the oracle (+3 SE); ridge on
    a random stand-in of the same width gains nothing beyond its MDE. The stand-in learns, per
    drug, how training lines' departures follow random numbers; the hidden line's random numbers
    say nothing about its response.

    The check is one-sided, gain <= MDE (decisions.md, 2026-09-11 amendment): at the top penalty
    the stand-in predicts the drug average plus a tiny fit to noise, a small steady loss with a
    near-zero redraw spread, and a leak through a random description could only show as a gain.
    A penalty chosen too small is caught by the null test's requirement that the stand-in's
    penalty sit at the top."""
    control = _fit_control("lolo", seed=71)
    _assert_planted_and_recovered(control)
    random = control.random
    assert random["estimate"] <= random["mde"], _describe(random)


@pytest.mark.step_fit
def test_fit_recovers_planted_drug_response() -> None:
    """A drug hidden: effects shared by chemically similar drugs, part of them scaled line by line
    by a width-20 description (strength 0.3). The oracle's gain over chemistry only is at least
    twice its MDE; ridge on the matching description gains detectably (p < 0.05) and no more than
    the oracle (+3 SE).

    The stand-in requirement differs from a hidden line's (recorded departure, 2026-09-11). When a
    drug is hidden, every line is in training: a line description only supplies a basis for line
    space along which chemically similar drugs share effects, and a random basis of 20 columns
    spans about 20/49 of the 49 directions lines vary in. So the random stand-in recovers part of
    any planted line effect, and cannot be required to gain nothing. On this grid (seed 72) it
    recovers +0.0039 to the matching description's +0.0096, measured from their shared level at
    strength 0 (0.41, against 20/49 = 0.41); both sit 0.005 below chemistry only at strength 0,
    fitting noise at the small penalty chemistry needs, so the stand-in's net gain here is a small
    loss. The control requires the matching description to beat its stand-in (p < 0.05) and
    reports the stand-in's gain."""
    control = _fit_control("lodo", seed=72)
    _assert_planted_and_recovered(control)
    versus_random = control.description_minus_random
    assert versus_random["estimate"] > 0 and versus_random["p"] < ALPHA, (
        f"description minus stand-in: {_describe(versus_random)}; "
        f"stand-in over chemistry only: {_describe(control.random)}"
    )


@pytest.mark.step_fit
@pytest.mark.parametrize("scheme", ["lolo", "lodo"])
def test_fit_null_shrinks_to_reference(scheme: Scheme) -> None:
    """Nothing planted beyond what the reference predicts: no model gains more than its MDE over
    the reference (read one-sided, gain <= MDE: a model fitting noise loses a little, steadily),
    and ridge chooses the largest penalty in at least half the rounds.

    * A line hidden: strength 0, so answers are each drug's shared response plus noise; models
      are ridge on the description and on a random stand-in, and nearest lines, against the drug
      average.
    * A drug hidden: strength 0 and no chemistry effect, so answers are line baselines plus
      noise; models are ridge on the description and on a random stand-in against chemistry
      only, itself a ridge fit whose penalty is checked too. (With the chemistry effect kept,
      every ridge needs a small penalty for the drugs' shared effects, which the penalty shared
      by every component then applies to the line part too.)"""
    stand_in = linear_kernel(random_stand_in(50, 20, 74))
    if scheme == "lolo":
        noise = noise_for_reliability(R_RESPONDING, lolo_signal_variance(0.0))
        grid = synthetic_lolo_grid(strength=0.0, noise=noise, seed=73)
        kernels = {"description": linear_kernel(grid.description), "random": stand_in}
        rounds = _lolo_rounds(grid.delta, kernels, similarity_from_description(grid.description))
        reference = "drug_average"
    else:
        noise = noise_for_reliability(R_RESPONDING, lodo_signal_variance(0.0, chemistry=0.0))
        grid = synthetic_lodo_grid(strength=0.0, chemistry=0.0, noise=noise, seed=73)
        assert grid.fingerprints is not None
        kernels = {
            "chemistry_only": np.zeros((50, 50)),
            "description": linear_kernel(grid.description),
            "random": stand_in,
        }
        rounds = _lodo_rounds(grid.delta, tanimoto(grid.fingerprints), kernels)
        reference = "chemistry_only"
    answers = synthetic_answers(grid.delta)
    scores = {name: _grid_scores(p, answers) for name, p in rounds.predictions.items()}

    gains = {
        name: _contrast(scores[name], scores[reference], scheme)
        for name in scores
        if name != reference
    }
    beyond = {name: _describe(g) for name, g in gains.items() if g["estimate"] > g["mde"]}
    assert not beyond, f"gains beyond the MDE with nothing planted: {beyond}"
    at_top = {name: float(np.mean(lams == LAMBDAS[-1])) for name, lams in rounds.lambdas.items()}
    assert min(at_top.values()) >= 0.5, f"share of rounds at the largest penalty: {at_top}"


# ==============================================================================================
# split: a signature in a unit's own answers leaks through a broken split, not the shipped one


SPLIT_LINES, SPLIT_DRUGS, SPLIT_GENES, SPLIT_WIDTH = 20, 24, 200, 8


@dataclass(frozen=True)
class SplitRun:
    """One scheme's split control on a 20-line x 24-drug x 200-gene grid carrying a planted
    signature per unit: each round's shipped and leaky predictions, the planted answers, and the
    signatures wrapped as an answer to score predictions against."""

    shipped: np.ndarray
    leaky: np.ndarray
    planted: np.ndarray
    signature_answers: Answers
    kernel: np.ndarray
    similarity: np.ndarray | None


def _split_run(scheme: Scheme) -> SplitRun:
    """Rounds of the shipped ridge fit (the matching description; chemistry and the description
    when a drug is hidden) and of the leaky fit at the penalty the shipped fit chose, so the two
    differ only in whether the hidden unit's answers were fitted."""
    shape = (SPLIT_LINES, SPLIT_DRUGS, SPLIT_GENES)
    tested = np.ones(shape, dtype=bool)
    shipped = np.empty(shape)
    leaky = np.empty(shape)
    if scheme == "lolo":
        noise = noise_for_reliability(R_RESPONDING, lolo_signal_variance(STRENGTH))
        grid = synthetic_lolo_grid(*shape, SPLIT_WIDTH, strength=STRENGTH, noise=noise, seed=81)
        planted, signatures = planted_signature_grid(grid.delta, "lolo", 1.0, seed=82)
        kernel = linear_kernel(grid.description)
        similarity = None
        for held in range(SPLIT_LINES):
            fit = ridge_lolo(kernel, planted, tested, held)
            assert fit.lam is not None
            shipped[held] = fit.prediction
            leaky[held] = leaky_lolo_prediction(kernel, planted, held, fit.lam)
        signature_answer = np.broadcast_to(signatures[:, None, :], shape)
    else:
        noise = noise_for_reliability(R_RESPONDING, lodo_signal_variance(STRENGTH))
        grid = synthetic_lodo_grid(*shape, SPLIT_WIDTH, strength=STRENGTH, noise=noise, seed=83)
        assert grid.fingerprints is not None
        planted, signatures = planted_signature_grid(grid.delta, "lodo", 1.0, seed=84)
        kernel = linear_kernel(grid.description)
        similarity = tanimoto(grid.fingerprints)
        for held in range(SPLIT_DRUGS):
            fit = ridge_lodo(kernel, similarity, planted, tested, held)
            assert fit.lam is not None
            shipped[:, held] = fit.prediction
            leaky[:, held] = leaky_lodo_prediction(kernel, similarity, planted, held, fit.lam)
        signature_answer = np.broadcast_to(signatures[None, :, :], shape)
    return SplitRun(
        shipped=shipped,
        leaky=leaky,
        planted=planted,
        signature_answers=synthetic_answers(signature_answer),
        kernel=kernel,
        similarity=similarity,
    )


def _signature_score(run: SplitRun, prediction: np.ndarray, scheme: Scheme) -> RedrawSummary:
    """Mean over pairs of the correlation, across genes, between a prediction and the hidden
    unit's own signature (through ``score_pairs``), redrawn over the held-out units."""
    scores = _grid_scores(prediction, run.signature_answers)
    return _contrast(scores, np.zeros_like(scores), scheme)


@pytest.mark.step_split
@pytest.mark.parametrize("scheme", ["lolo", "lodo"])
def test_split_leaky_recovers_signature(scheme: Scheme) -> None:
    """A random gene signature planted in each line's (or drug's) own answers only: a broken
    split that fits ridge with the hidden unit kept in training puts that signature into the
    hidden unit's prediction, so the prediction correlates with it clearly above 0 -- detected
    (p < 0.05) and above its MDE."""
    run = _split_run(scheme)
    leaked = _signature_score(run, run.leaky, scheme)
    assert leaked["estimate"] > leaked["mde"] and leaked["p"] < ALPHA, _describe(leaked)


@pytest.mark.step_split
@pytest.mark.parametrize("scheme", ["lolo", "lodo"])
def test_split_shipped_does_not(scheme: Scheme) -> None:
    """Under the shipped splits the same signature scores 0 within its MDE: nothing about the
    hidden unit's answers reaches its prediction. Checked directly as well (invariant 1): with the
    hidden unit's answers and tested marks replaced by arbitrary values, the shipped fit is bit
    for bit the same, while the leaky fit changes."""
    run = _split_run(scheme)
    kept_out = _signature_score(run, run.shipped, scheme)
    assert abs(kept_out["estimate"]) <= kept_out["mde"], _describe(kept_out)

    held = 3
    rng = np.random.default_rng(85)
    poisoned = run.planted.copy()
    tested = np.ones(poisoned.shape, dtype=bool)
    poisoned_tested = tested.copy()
    if scheme == "lolo":
        poisoned[held] = 1e6 * rng.standard_normal(poisoned.shape[1:])
        poisoned_tested[held] = rng.random(poisoned.shape[1:]) < 0.5
        clean = ridge_lolo(run.kernel, run.planted, tested, held)
        dirty = ridge_lolo(run.kernel, poisoned, poisoned_tested, held)
        assert clean.lam is not None
        leak_moves = not np.array_equal(
            leaky_lolo_prediction(run.kernel, run.planted, held, clean.lam),
            leaky_lolo_prediction(run.kernel, poisoned, held, clean.lam),
        )
    else:
        assert run.similarity is not None
        column = (poisoned.shape[0], poisoned.shape[2])
        poisoned[:, held] = 1e6 * rng.standard_normal(column)
        poisoned_tested[:, held] = rng.random(column) < 0.5
        clean = ridge_lodo(run.kernel, run.similarity, run.planted, tested, held)
        dirty = ridge_lodo(run.kernel, run.similarity, poisoned, poisoned_tested, held)
        assert clean.lam is not None
        leak_moves = not np.array_equal(
            leaky_lodo_prediction(run.kernel, run.similarity, run.planted, held, clean.lam),
            leaky_lodo_prediction(run.kernel, run.similarity, poisoned, held, clean.lam),
        )
    assert np.array_equal(clean.prediction, dirty.prediction)
    assert (clean.lam, clean.loss_min, clean.at_edge) == (dirty.lam, dirty.loss_min, dirty.at_edge)
    assert leak_moves
