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

Everything is numpy and seeded: the same arguments give the same arrays. The tests in
``tests/test_rung1_controls.py`` use these now; the combine job's control tables reuse them.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import cast

import numpy as np

from fmharness.heldout import Scheme
from fmharness.heldout.answers import Answers


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


__all__ = [
    "SyntheticGrid",
    "leaky_lodo_prediction",
    "leaky_lolo_prediction",
    "lodo_signal_variance",
    "lolo_signal_variance",
    "noise_for_reliability",
    "planted_reliability_pool",
    "planted_signature_grid",
    "synthetic_answers",
    "synthetic_lodo_grid",
    "synthetic_lolo_grid",
]
