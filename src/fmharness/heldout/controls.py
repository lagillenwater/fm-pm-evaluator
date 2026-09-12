"""Synthetic grids with planted answers, for rung 1's known-answer controls (design.md section 8).

Each builder plants an answer the shipped code must recover, or must not:

* ``planted_reliability_pool`` -- the score control: a true change and two noisy measurements
  of it at a known reliability R, so the true change scores √R against a measurement and the
  two measurements score R against each other.
* ``synthetic_lolo_grid`` -- the fit control when a line is hidden: every drug has a response
  shared by all lines, plus a line-specific part linear in the line's description.
* ``synthetic_lodo_grid`` -- the fit control when a drug is hidden: each line has a baseline,
  drugs with similar fingerprints have similar effects, and part of each drug's effect is
  modulated by the line's description.
* ``planted_signature_grid`` -- the split control: a random gene signature added to one
  unit's (line's or drug's) own answers only, unrelated to any description.
* ``leaky_lolo_prediction`` and ``leaky_lodo_prediction`` -- deliberately broken splits that
  fit ridge regression with the hidden unit left IN the training set. They are dense solves
  written independently of ``fmharness.heldout.models`` (no eigendecomposition, no leave-one-out
  shortcut), so they cannot share a defect with the shipped split.
* ``synthetic_answers`` -- a synthetic array wrapped as ``Answers``, so every control is scored
  through the real ``score_pairs``.

Everything is numpy and seeded: the same arguments give the same arrays.

The second half of this module is the **shared control harness** (ruling 38): the rounds each
control runs, the contrast it reads off them, and the four control bodies themselves --
``fit_control``, ``fit_null_control``, ``split_control_run`` and ``null_repetition_pvalues`` --
together with the sizes and seeds they use. ``tests/test_rung1_controls.py`` asserts on these and
``scripts/heldout_combine.py`` publishes them, and both call THIS code. They were once two
near-identical copies that happened to agree; a copy that agrees today is one edit away from a
published evidence table that quietly keeps the old behaviour while the test asserts the new.
The plan's file map always designated this module "synthetic grids with planted answers, shared
by tests and the combine job".
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from fmharness.heldout import Scheme
from fmharness.heldout.answers import Answers, scoreable
from fmharness.heldout.chemistry import tanimoto
from fmharness.heldout.comparisons import (
    N_DRAWS,
    RedrawSummary,
    redraw_estimates,
    summarize_redraws,
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
from fmharness.heldout.scoring import score_pairs


@dataclass(frozen=True)
class SyntheticGrid:
    """A synthetic grid and what was planted in it.

    ``delta`` -- the measured change ``[L, D, G]``, float64: ``truth`` plus independent
    Gaussian noise.
    ``truth`` -- the noiseless change ``[L, D, G]``: what an oracle predicts.
    ``description`` -- the line description ``[L, width]`` the line-specific part is built from.
    ``fingerprints`` -- bool ``[D, n_bits]`` drug fingerprints (leave-one-drug-out grids only;
    ``None`` otherwise).
    """

    delta: np.ndarray
    truth: np.ndarray
    description: np.ndarray
    fingerprints: np.ndarray | None


def noise_for_reliability(reliability: float, signal_variance: float) -> float:
    """The noise SD at which a measurement with true-change variance ``signal_variance`` has
    reliability ``reliability``: ``R = s / (s + noise²)``, so ``noise = sqrt(s (1 - R) / R)``."""
    if not 0.0 < reliability < 1.0:
        raise ValueError(f"reliability must lie strictly between 0 and 1; got {reliability}")
    return float(np.sqrt(signal_variance * (1.0 - reliability) / reliability))


def planted_reliability_pool(
    R: float, n_pairs: int, n_genes: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(truth, measurement, second_measurement)``, each ``[n_pairs, n_genes]`` float64.

    Per gene, the true change has variance ``R`` and each measurement adds independent noise of
    variance ``1 - R``, so a measurement's reliability is ``R``: across genes, the truth
    correlates with a measurement at ``√R`` and two measurements correlate at ``R``.
    """
    if not 0.0 < R < 1.0:
        raise ValueError(f"R must lie strictly between 0 and 1; got {R}")
    rng = np.random.default_rng(seed)
    shape = (n_pairs, n_genes)
    truth = np.sqrt(R) * rng.standard_normal(shape)
    measurement = truth + np.sqrt(1.0 - R) * rng.standard_normal(shape)
    second_measurement = truth + np.sqrt(1.0 - R) * rng.standard_normal(shape)
    return truth, measurement, second_measurement


def lolo_signal_variance(strength: float) -> float:
    """The expected per-gene variance of a pair's true change in ``synthetic_lolo_grid``: 1 for
    the drug's shared response, ``strength²`` for the line-specific part."""
    return 1.0 + strength**2


def synthetic_lolo_grid(
    n_lines: int = 50,
    n_drugs: int = 107,
    n_genes: int = 300,
    width: int = 20,
    *,
    strength: float,
    noise: float,
    seed: int,
) -> SyntheticGrid:
    """A grid for the fit control when a line is hidden.

    With ``rng = np.random.default_rng(seed)``, drawn in this order whatever ``strength`` and
    ``noise`` are (so grids differing only in those share every draw):

    * the description ``Z`` ``[L, width]``, standard normal;
    * each drug's shared response ``g_d`` ``[D, G]``, standard normal;
    * each drug's line loadings ``B_d`` ``[D, width, G]``, normal with variance ``1/width``;
    * standard normal noise ``[L, D, G]``.

    ``truth[l, d] = g_d + strength · Z_l · B_d``, so the line-specific part is linear in the
    description with per-gene variance ``strength² |Z_l|² / width`` (``strength²`` on average);
    ``delta = truth + noise · noise_draw``. The drug average carries the shared response; only a
    model that uses the description can predict the rest.
    """
    rng = np.random.default_rng(seed)
    description = rng.standard_normal((n_lines, width))
    shared = rng.standard_normal((n_drugs, n_genes))
    loadings = rng.standard_normal((n_drugs, width, n_genes)) / np.sqrt(width)
    noise_draw = rng.standard_normal((n_lines, n_drugs, n_genes))
    truth = shared[None, :, :] + strength * np.tensordot(description, loadings, axes=(1, 1))
    return SyntheticGrid(
        delta=truth + noise * noise_draw,
        truth=truth,
        description=description,
        fingerprints=None,
    )


def lodo_signal_variance(strength: float, chemistry: float = 1.0) -> float:
    """The expected per-gene variance of a pair's true change in ``synthetic_lodo_grid``: 1 for
    the line baseline, ``chemistry²`` for the drug's shared effect, ``strength²`` for the part
    modulated by the line description."""
    return 1.0 + chemistry**2 + strength**2


def synthetic_lodo_grid(
    n_lines: int = 50,
    n_drugs: int = 107,
    n_genes: int = 300,
    width: int = 20,
    *,
    strength: float,
    noise: float,
    seed: int,
    chemistry: float = 1.0,
    n_bits: int = 256,
    n_families: int = 12,
) -> SyntheticGrid:
    """A grid for the fit control when a drug is hidden.

    Fingerprints: ``n_families`` random scaffolds (each bit on with probability 0.15); drug ``d``
    copies scaffold ``d % n_families`` and flips each bit with probability 0.05, and bit 0 is
    always on, so drugs in a family share most bits (Tanimoto similarity carries the family) and
    no fingerprint is empty. With ``u_d = f_d / |f_d|`` a drug's unit-length fingerprint, an
    effect ``u_d · V`` with ``V`` ``[n_bits, G]`` standard normal has per-gene variance 1 and
    covariance ``u_d · u_d'`` between drugs: smooth in fingerprint space.

    With ``rng = np.random.default_rng(seed)``, drawn in this order whatever ``strength``,
    ``chemistry`` and ``noise`` are: the description ``Z`` ``[L, width]``; the scaffolds and the
    flips; each line's baseline ``a_l`` ``[L, G]``; the shared effect's ``V``; ``width`` more
    such effects ``V'_c``; standard normal noise ``[L, D, G]``. Then

    ``truth[l, d] = a_l + chemistry · u_d · V + strength · Σ_c Z_lc (u_d · V'_c) / √width``

    -- a line baseline (each line's mean over drugs carries it), an effect shared by chemically
    similar drugs in every line (chemistry only predicts it), and an effect shared by chemically
    similar drugs and scaled, line by line, by the description (a model needs both the
    chemistry and the description). ``delta = truth + noise · noise_draw``.
    """
    rng = np.random.default_rng(seed)
    description = rng.standard_normal((n_lines, width))
    scaffolds = rng.random((n_families, n_bits)) < 0.15
    flips = rng.random((n_drugs, n_bits)) < 0.05
    fingerprints = scaffolds[np.arange(n_drugs) % n_families] ^ flips
    fingerprints[:, 0] = True
    baseline = rng.standard_normal((n_lines, n_genes))
    shared_bits = rng.standard_normal((n_bits, n_genes))
    modulated_bits = rng.standard_normal((width, n_bits, n_genes))
    noise_draw = rng.standard_normal((n_lines, n_drugs, n_genes))

    unit = fingerprints / np.linalg.norm(fingerprints.astype(np.float64), axis=1, keepdims=True)
    shared = unit @ shared_bits  # [D, G]
    modulated = np.matmul(unit, modulated_bits)  # [width, D, G]
    line_part = np.tensordot(description / np.sqrt(width), modulated, axes=(1, 0))  # [L, D, G]
    truth = baseline[:, None, :] + chemistry * shared[None, :, :] + strength * line_part
    return SyntheticGrid(
        delta=truth + noise * noise_draw,
        truth=truth,
        description=description,
        fingerprints=fingerprints,
    )


def planted_signature_grid(
    delta: np.ndarray, scheme: Scheme, amplitude: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(planted, signatures)``: ``delta`` ``[L, D, G]`` with a random gene signature added to
    each unit's own answers, and the signatures ``[U, G]``.

    The unit is a line when a line is hidden (``scheme="lolo"``: ``planted[l, d] = delta[l, d] +
    amplitude · signatures[l]`` for every drug) and a drug when a drug is hidden (``"lodo"``:
    ``planted[l, d] = delta[l, d] + amplitude · signatures[d]`` for every line). Signatures are
    standard normal from ``np.random.default_rng(seed)``, independent of everything else, so a
    hidden unit's signature is in its own answers and nowhere a correct split lets a model see.
    """
    n_lines, n_drugs, n_genes = delta.shape
    rng = np.random.default_rng(seed)
    if scheme == "lolo":
        signatures = rng.standard_normal((n_lines, n_genes))
        added = signatures[:, None, :]
    else:
        signatures = rng.standard_normal((n_drugs, n_genes))
        added = signatures[None, :, :]
    return np.asarray(delta, dtype=np.float64) + amplitude * added, signatures


def _solve(system: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """``np.linalg.solve(system, targets)`` for a 2-D ``targets`` (one column per target); numpy's
    stubs leave the result's type unknown for a plain ``ndarray``, hence the cast."""
    return cast(np.ndarray, np.linalg.solve(system, targets))


def leaky_lolo_prediction(
    kernel: np.ndarray, delta0: np.ndarray, held: int, lam: float
) -> np.ndarray:
    """A broken leave-one-line-out split: ridge on every line, the hidden one included. ``[D, G]``.

    The same model as ``models.ridge_lolo`` at penalty ``lam`` -- each drug's mean change as the
    reference, departures from it regressed on the line kernel -- but the reference is the mean
    over all ``L`` lines and the dual coefficients ``alpha = (K + λI)⁻¹ R`` solve over all ``L``
    lines, line ``held`` among them. The hidden line's own answer therefore enters its own
    prediction ``mean + K[held] alpha``. A dense ``np.linalg.solve``, for small grids.
    """
    held = operator.index(held)
    n_lines = delta0.shape[0]
    answers = np.asarray(delta0, dtype=np.float64).reshape(n_lines, -1)
    reference = answers.mean(axis=0)
    alpha = _solve(kernel + lam * np.eye(n_lines), answers - reference)
    return (reference + kernel[held] @ alpha).reshape(delta0.shape[1:])


def leaky_lodo_prediction(
    K_line: np.ndarray, T: np.ndarray, delta0: np.ndarray, held: int, lam: float
) -> np.ndarray:
    """A broken leave-one-drug-out split: ridge on every drug, the hidden one included. ``[L, G]``.

    The same model as ``models.ridge_lodo`` at penalty ``lam`` -- each line's mean change over
    drugs as the reference, departures regressed on the pair kernel ``T[d, d'] (1 +
    K_line[l, l'])`` -- but the line means are taken over all ``D`` drugs and the dual
    coefficients ``alpha = (np.kron(1 + K_line, T) + λI)⁻¹ R`` solve over every (line, drug)
    pair, drug ``held``'s among them. The hidden drug is predicted in every line as the line mean
    plus ``(1 + K_line) (alpha T[held])``, so its own answer enters its own prediction. A dense
    ``np.linalg.solve`` on the full ``L·D`` pair kernel, for small grids.
    """
    held = operator.index(held)
    n_lines, n_drugs, n_genes = delta0.shape
    answers = np.asarray(delta0, dtype=np.float64)
    reference = answers.mean(axis=1)
    departures = (answers - reference[:, None, :]).reshape(n_lines * n_drugs, n_genes)
    line_factor = 1.0 + K_line
    system = np.kron(line_factor, T) + lam * np.eye(n_lines * n_drugs)
    alpha = _solve(system, departures).reshape(n_lines, n_drugs, n_genes)
    return reference + line_factor @ np.einsum("ldg,d->lg", alpha, T[held])


def synthetic_answers(delta: np.ndarray, responding: np.ndarray | None = None) -> Answers:
    """A synthetic ``[L, D, G]`` answer wrapped as ``Answers`` for ``score_pairs``.

    Lines, drugs and genes are named ``line_000``, ``drug_000``, ``gene_0000`` in index order;
    ``delta`` is stored as float32, like the real answer. ``responding`` defaults to every gene
    with a finite answer, so the two gene sets coincide; when given, it is restricted to finite
    answers.
    """
    answer = np.asarray(delta, dtype=np.float32)
    if answer.ndim != 3:
        raise ValueError(f"delta must be [lines, drugs, genes]; got shape {answer.shape}")
    finite = np.isfinite(answer)
    marks = finite if responding is None else np.asarray(responding, dtype=bool) & finite
    n_lines, n_drugs, n_genes = answer.shape
    return Answers(
        lines=tuple(f"line_{i:03d}" for i in range(n_lines)),
        drugs=tuple(f"drug_{j:03d}" for j in range(n_drugs)),
        genes=tuple(f"gene_{g:04d}" for g in range(n_genes)),
        delta=answer,
        responding=marks,
    )


# ==============================================================================================
# The shared control harness (ruling 38): what tests/test_rung1_controls.py asserts on and what
# scripts/heldout_combine.py publishes, in one place so the two cannot drift apart.

#: The design's full-data reliabilities (section 6): responding genes, all genes.
R_RESPONDING = 0.7353
R_ALL = 0.1503

#: The fit control's planted strength (recorded departure, 2026-09-11), and its grid: the
#: screen's own size, which is also ``synthetic_lolo_grid``'s and ``synthetic_lodo_grid``'s
#: default. It is passed explicitly wherever a grid is built, so the size a control table records
#: is the size that actually ran rather than a constant that merely happens to match (ruling 41).
STRENGTH = 0.3
FIT_GRID: dict[str, int] = {"n_lines": 50, "n_drugs": 107, "n_genes": 300, "width": 20}
FIT_SEEDS: dict[str, int] = {"lolo": 71, "lodo": 72}
FIT_NULL_GRID_SEED = 73
FIT_NULL_STAND_IN_SEED = 74

#: The split control's grid. Deliberately small: the leaky fit is a dense solve over every
#: (line, drug) pair, so the broken split cannot be run at the screen's size (task 9's note).
SPLIT_GRID: dict[str, int] = {"n_lines": 20, "n_drugs": 24, "n_genes": 200, "width": 8}
SPLIT_AMPLITUDE = 1.0
SPLIT_SEEDS: dict[str, tuple[int, int]] = {"lolo": (81, 82), "lodo": (83, 84)}

#: The score control's pool, and the seeds of its two parts.
SCORE_GRID: dict[str, int] = {"n_lines": 20, "n_drugs": 30, "n_genes": 2000}
SCORE_POOL_SEED, SCORE_RESPONDING_SEED = 41, 42
SCORE_UNRELATED_SEEDS = (43, 44, 45)

#: Repetitions behind every rate, the level a detection is read at, and the rate an effect
#: planted at its own MDE is detected at (invariant 9).
REPETITIONS = 200
ALPHA = 0.05
DETECTION_RATE = 0.80
NULL_UNITS = 50
NULL_UNIT_SEED = 61

#: The redraw seed every control contrast uses.
REDRAW_SEED = 20260911

#: A recovered mean passes within this many Monte Carlo standard errors of the planted value.
SE_MULTIPLE = 3.0


@dataclass(frozen=True)
class Rounds:
    """Every round's prediction per model, stacked into ``[L, D, G]``, and each ridge model's
    chosen penalty per round."""

    predictions: dict[str, np.ndarray]
    lambdas: dict[str, np.ndarray]


@dataclass(frozen=True)
class FitControl:
    """A fit control's contrasts, each against the scheme's reference unless named otherwise."""

    oracle: RedrawSummary
    description: RedrawSummary
    description_minus_oracle: RedrawSummary
    random: RedrawSummary
    description_minus_random: RedrawSummary


@dataclass(frozen=True)
class FitNull:
    """The fit control with nothing planted: each model's gain over the scheme's reference, and
    the share of rounds each ridge model chose the largest penalty in."""

    reference: str
    gains: dict[str, RedrawSummary]
    lambda_at_top: dict[str, float]


@dataclass(frozen=True)
class SplitRun:
    """One scheme's split control: each round's shipped and leaky predictions, the planted
    answers, the signatures wrapped as an answer, and the kernels the fits used."""

    shipped: np.ndarray
    leaky: np.ndarray
    planted: np.ndarray
    signature_answers: Answers
    kernel: np.ndarray
    similarity: np.ndarray | None


@dataclass(frozen=True)
class NullRepetitions:
    """p-values over ``REPETITIONS`` synthetic comparisons: with nothing planted, and with the
    same differences shifted by the MDE that repetition's null redraws estimate."""

    p_null: np.ndarray
    p_planted: np.ndarray


def grid_scores(prediction: np.ndarray, answers: Answers) -> np.ndarray:
    """Every pair's score through the real ``score_pairs``, as an ``[L, D]`` array.

    Responding genes; in a synthetic answer every finite gene is responding, so both gene sets
    agree. Raises if the scoring did not return one row per pair, which would silently reshape.
    """
    n_lines, n_drugs, n_genes = prediction.shape
    lines, drugs = np.divmod(np.arange(n_lines * n_drugs), n_drugs)
    frame = score_pairs(
        prediction.reshape(-1, n_genes), lines, drugs, answers, scoreable(answers, ())
    )
    scores = frame.loc[frame["gene_set"] == "responding", "r"].to_numpy(dtype=np.float64)
    if scores.size != n_lines * n_drugs:
        raise ValueError(
            f"scored {scores.size} pairs, not the grid's {n_lines * n_drugs}; a control cannot be "
            "read off a partial grid"
        )
    return scores.reshape(n_lines, n_drugs)


def contrast_summary(
    scores_a: np.ndarray, scores_b: np.ndarray, scheme: Scheme, seed: int = REDRAW_SEED
) -> RedrawSummary:
    """Mean over pairs of ``scores_a - scores_b`` (both ``[L, D]``), redrawn over the held-out
    unit of ``scheme``: lines when a line is hidden, drugs when a drug is hidden."""
    diffs = scores_a - scores_b
    lines, drugs = np.indices(diffs.shape)
    units, n_units = (lines, diffs.shape[0]) if scheme == "lolo" else (drugs, diffs.shape[1])
    draws = redraw_estimates(diffs.ravel(), units.ravel(), n_units, N_DRAWS, seed)
    return summarize_redraws(float(diffs.mean()), draws)


def lolo_rounds(
    delta: np.ndarray,
    kernels: Mapping[str, np.ndarray],
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
        for name, kernel in kernels.items():
            fit = ridge_lolo(kernel, delta, tested, held)
            predictions[name][held] = fit.prediction
            lambdas[name][held] = np.nan if fit.lam is None else float(fit.lam)
        if similarity is not None:
            predictions["nearest_lines"][held] = nearest_lines_lolo(
                similarity, delta, tested, held
            ).prediction
    return Rounds(predictions=predictions, lambdas=lambdas)


def lodo_rounds(
    delta: np.ndarray, similarity: np.ndarray, kernels: Mapping[str, np.ndarray]
) -> Rounds:
    """All leave-one-drug-out rounds of ridge on each line kernel (all zeros: chemistry only)."""
    n_drugs = delta.shape[1]
    tested = np.ones(delta.shape, dtype=bool)
    predictions = {name: np.empty(delta.shape) for name in kernels}
    lambdas = {name: np.empty(n_drugs) for name in kernels}
    for held in range(n_drugs):
        for name, kernel in kernels.items():
            fit = ridge_lodo(kernel, similarity, delta, tested, held)
            predictions[name][:, held] = fit.prediction
            lambdas[name][held] = np.nan if fit.lam is None else float(fit.lam)
    return Rounds(predictions=predictions, lambdas=lambdas)


def _fit_kernels(
    scheme: Scheme, grid: SyntheticGrid, stand_in: np.ndarray
) -> tuple[dict[str, np.ndarray], str]:
    """The kernels one fit-control scheme runs, and the name of its reference model.

    The stand-in must not coincide with the description's own kernel: if it did, the control
    would publish ``description_minus_random`` as an exact zero, which reads exactly like "the
    description adds nothing over noise" -- the defect class task 10a shipped.
    """
    described = linear_kernel(grid.description)
    if np.allclose(stand_in, described):
        raise ValueError(
            "the random stand-in's kernel coincides with the description's, so their contrast "
            "would be an exact zero indistinguishable from a true null"
        )
    if scheme == "lolo":
        return {"description": described, "random": stand_in}, "drug_average"
    n_lines = grid.description.shape[0]
    return (
        {
            "chemistry_only": np.zeros((n_lines, n_lines)),
            "description": described,
            "random": stand_in,
        },
        "chemistry_only",
    )


def fit_control(scheme: Scheme, seed: int) -> FitControl:
    """The planted fit control on a grid of the screen's size (``FIT_GRID``), strength
    ``STRENGTH`` at reliability ``R_RESPONDING``: every round of the scheme's reference, ridge on
    the matching description, and ridge on a random stand-in of the same width; the oracle
    predicts the noiseless change."""
    stand_in = linear_kernel(random_stand_in(FIT_GRID["n_lines"], FIT_GRID["width"], seed + 1))
    if scheme == "lolo":
        noise = noise_for_reliability(R_RESPONDING, lolo_signal_variance(STRENGTH))
        grid = synthetic_lolo_grid(
            FIT_GRID["n_lines"],
            FIT_GRID["n_drugs"],
            FIT_GRID["n_genes"],
            FIT_GRID["width"],
            strength=STRENGTH,
            noise=noise,
            seed=seed,
        )
        kernels, reference = _fit_kernels(scheme, grid, stand_in)
        rounds = lolo_rounds(grid.delta, kernels)
    else:
        noise = noise_for_reliability(R_RESPONDING, lodo_signal_variance(STRENGTH))
        grid = synthetic_lodo_grid(
            FIT_GRID["n_lines"],
            FIT_GRID["n_drugs"],
            FIT_GRID["n_genes"],
            FIT_GRID["width"],
            strength=STRENGTH,
            noise=noise,
            seed=seed,
        )
        if grid.fingerprints is None:
            raise ValueError("the leave-one-drug-out control grid carries no fingerprints")
        kernels, reference = _fit_kernels(scheme, grid, stand_in)
        rounds = lodo_rounds(grid.delta, tanimoto(grid.fingerprints), kernels)

    answers = synthetic_answers(grid.delta)
    scores = {name: grid_scores(p, answers) for name, p in rounds.predictions.items()}
    scores["oracle"] = grid_scores(grid.truth, answers)
    return FitControl(
        oracle=contrast_summary(scores["oracle"], scores[reference], scheme),
        description=contrast_summary(scores["description"], scores[reference], scheme),
        description_minus_oracle=contrast_summary(scores["description"], scores["oracle"], scheme),
        random=contrast_summary(scores["random"], scores[reference], scheme),
        description_minus_random=contrast_summary(scores["description"], scores["random"], scheme),
    )


def fit_null_control(scheme: Scheme) -> FitNull:
    """The fit control with nothing planted: strength 0 (and, when a drug is hidden, no chemistry
    effect either), so no model may gain beyond its MDE and ridge must sit at the top of its
    penalty grid in most rounds."""
    stand_in = linear_kernel(
        random_stand_in(FIT_GRID["n_lines"], FIT_GRID["width"], FIT_NULL_STAND_IN_SEED)
    )
    if scheme == "lolo":
        noise = noise_for_reliability(R_RESPONDING, lolo_signal_variance(0.0))
        grid = synthetic_lolo_grid(
            FIT_GRID["n_lines"],
            FIT_GRID["n_drugs"],
            FIT_GRID["n_genes"],
            FIT_GRID["width"],
            strength=0.0,
            noise=noise,
            seed=FIT_NULL_GRID_SEED,
        )
        kernels, reference = _fit_kernels(scheme, grid, stand_in)
        rounds = lolo_rounds(grid.delta, kernels, similarity_from_description(grid.description))
    else:
        noise = noise_for_reliability(R_RESPONDING, lodo_signal_variance(0.0, chemistry=0.0))
        grid = synthetic_lodo_grid(
            FIT_GRID["n_lines"],
            FIT_GRID["n_drugs"],
            FIT_GRID["n_genes"],
            FIT_GRID["width"],
            strength=0.0,
            chemistry=0.0,
            noise=noise,
            seed=FIT_NULL_GRID_SEED,
        )
        if grid.fingerprints is None:
            raise ValueError("the leave-one-drug-out null grid carries no fingerprints")
        kernels, reference = _fit_kernels(scheme, grid, stand_in)
        rounds = lodo_rounds(grid.delta, tanimoto(grid.fingerprints), kernels)

    answers = synthetic_answers(grid.delta)
    scores = {name: grid_scores(p, answers) for name, p in rounds.predictions.items()}
    return FitNull(
        reference=reference,
        gains={
            name: contrast_summary(scores[name], scores[reference], scheme)
            for name in scores
            if name != reference
        },
        lambda_at_top={
            name: float(np.mean(lams == LAMBDAS[-1])) for name, lams in rounds.lambdas.items()
        },
    )


def split_control_run(scheme: Scheme) -> SplitRun:
    """Rounds of the shipped ridge fit and of the leaky fit at the penalty the shipped fit chose,
    on a grid carrying a planted signature per unit, so the two differ only in whether the hidden
    unit's answers were fitted."""
    shape = (SPLIT_GRID["n_lines"], SPLIT_GRID["n_drugs"], SPLIT_GRID["n_genes"])
    tested = np.ones(shape, dtype=bool)
    shipped = np.empty(shape)
    leaky = np.empty(shape)
    grid_seed, signature_seed = SPLIT_SEEDS[scheme]
    if scheme == "lolo":
        noise = noise_for_reliability(R_RESPONDING, lolo_signal_variance(STRENGTH))
        grid = synthetic_lolo_grid(
            *shape, SPLIT_GRID["width"], strength=STRENGTH, noise=noise, seed=grid_seed
        )
        planted, signatures = planted_signature_grid(
            grid.delta, "lolo", SPLIT_AMPLITUDE, seed=signature_seed
        )
        kernel = linear_kernel(grid.description)
        similarity = None
        for held in range(SPLIT_GRID["n_lines"]):
            fit = ridge_lolo(kernel, planted, tested, held)
            if fit.lam is None:
                raise ValueError("the shipped leave-one-line-out fit chose no penalty")
            shipped[held] = fit.prediction
            leaky[held] = leaky_lolo_prediction(kernel, planted, held, fit.lam)
        signature_answer = np.broadcast_to(signatures[:, None, :], shape)
    else:
        noise = noise_for_reliability(R_RESPONDING, lodo_signal_variance(STRENGTH))
        grid = synthetic_lodo_grid(
            *shape, SPLIT_GRID["width"], strength=STRENGTH, noise=noise, seed=grid_seed
        )
        if grid.fingerprints is None:
            raise ValueError("the leave-one-drug-out split grid carries no fingerprints")
        planted, signatures = planted_signature_grid(
            grid.delta, "lodo", SPLIT_AMPLITUDE, seed=signature_seed
        )
        kernel = linear_kernel(grid.description)
        similarity = tanimoto(grid.fingerprints)
        for held in range(SPLIT_GRID["n_drugs"]):
            fit = ridge_lodo(kernel, similarity, planted, tested, held)
            if fit.lam is None:
                raise ValueError("the shipped leave-one-drug-out fit chose no penalty")
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


def unit_structured_diffs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair differences for one comparison when a line is hidden, with nothing planted:
    50 lines holding 98-107 pairs each (as masks leave them), a line effect of SD 0.02 shared by
    the line's pairs, and pair noise of SD 0.05. Returns ``(diffs, line_index)``."""
    counts = 107 - rng.integers(0, 10, NULL_UNITS)
    units = np.repeat(np.arange(NULL_UNITS), counts)
    diffs = rng.normal(0.0, 0.02, NULL_UNITS)[units] + rng.normal(0.0, 0.05, units.size)
    return diffs, units


def null_repetition_pvalues() -> NullRepetitions:
    """``REPETITIONS`` synthetic comparisons, each read twice: with nothing planted, and shifted
    by exactly the MDE its own null redraws estimate."""
    p_null = np.empty(REPETITIONS)
    p_planted = np.empty(REPETITIONS)
    for repetition in range(REPETITIONS):
        diffs, units = unit_structured_diffs(np.random.default_rng([NULL_UNIT_SEED, repetition]))
        null = summarize_redraws(
            float(diffs.mean()), redraw_estimates(diffs, units, NULL_UNITS, N_DRAWS, repetition)
        )
        shifted = diffs + null["mde"]
        planted = summarize_redraws(
            float(shifted.mean()),
            redraw_estimates(shifted, units, NULL_UNITS, N_DRAWS, repetition),
        )
        p_null[repetition], p_planted[repetition] = null["p"], planted["p"]
    return NullRepetitions(p_null=p_null, p_planted=p_planted)


__all__ = [
    "ALPHA",
    "DETECTION_RATE",
    "FIT_GRID",
    "FIT_NULL_GRID_SEED",
    "FIT_NULL_STAND_IN_SEED",
    "FIT_SEEDS",
    "NULL_UNITS",
    "NULL_UNIT_SEED",
    "REDRAW_SEED",
    "REPETITIONS",
    "R_ALL",
    "R_RESPONDING",
    "SCORE_GRID",
    "SCORE_POOL_SEED",
    "SCORE_RESPONDING_SEED",
    "SCORE_UNRELATED_SEEDS",
    "SE_MULTIPLE",
    "SPLIT_AMPLITUDE",
    "SPLIT_GRID",
    "SPLIT_SEEDS",
    "STRENGTH",
    "FitControl",
    "FitNull",
    "NullRepetitions",
    "Rounds",
    "SplitRun",
    "SyntheticGrid",
    "contrast_summary",
    "fit_control",
    "fit_null_control",
    "grid_scores",
    "leaky_lodo_prediction",
    "leaky_lolo_prediction",
    "lodo_rounds",
    "lodo_signal_variance",
    "lolo_rounds",
    "lolo_signal_variance",
    "noise_for_reliability",
    "null_repetition_pvalues",
    "planted_reliability_pool",
    "planted_signature_grid",
    "split_control_run",
    "synthetic_answers",
    "synthetic_lodo_grid",
    "synthetic_lolo_grid",
    "unit_structured_diffs",
]
