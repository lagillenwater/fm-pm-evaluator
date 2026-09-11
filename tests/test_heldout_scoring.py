"""Rung 1's per-pair score and the redraw statistics its comparisons are read with (task 9).

Every test runs the real functions from ``fmharness.heldout.scoring`` and
``fmharness.heldout.comparisons`` against an independent computation: ``np.corrcoef`` on each
pair's selected genes for the score, and brute-force resampling for the redraws -- each held-out
unit's pairs repeated as many times as the unit was drawn, then averaged -- from weights drawn
with the same seed. Values in the resampling fixtures are multiples of 1/8, so every sum is exact
in float64 and the vectorized and brute-force estimates agree bit for bit. The known-answer
controls for the score and the redraws are in ``tests/test_rung1_controls.py``.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from fmharness.heldout.answers import Answers, scoreable
from fmharness.heldout.comparisons import (
    DRAWS_PER_BLOCK,
    MDE_FACTOR,
    N_BLOCKS,
    N_DRAWS,
    concatenate_blocks,
    design_effect,
    holm,
    in_blocks,
    mean_score_ci,
    redraw_block_seed,
    redraw_estimates,
    summarize_redraws,
    two_way_estimates,
)
from fmharness.heldout.scoring import (
    GENE_SETS,
    MIN_GENES,
    SCORE_COLUMNS,
    fraction_of_ceiling,
    score_pairs,
)

N_LINES, N_DRUGS, N_GENES = 4, 5, 120
EXCLUDED = (("L3", "D0"),)


def _fixture_answers() -> Answers:
    """A 4-line x 5-drug x 120-gene answer with about 10% of entries untested and about 75% of
    the tested genes responding, and three pairs with a known place in the masks:

    * (L1, D2) keeps only 30 responding genes, so it is scored on all genes only;
    * (L2, D4) is tested on only 40 genes, so it is scored on neither gene set;
    * (L3, D0) is an excluded pair, scored on neither whatever its genes.
    """
    rng = np.random.default_rng(3)
    delta = rng.standard_normal((N_LINES, N_DRUGS, N_GENES)).astype(np.float32)
    delta[rng.random(delta.shape) < 0.1] = np.nan
    responding = np.isfinite(delta) & (rng.random(delta.shape) < 0.75)
    responding[1, 2, np.flatnonzero(responding[1, 2])[30:]] = False
    delta[2, 4, 40:] = np.nan
    responding[2, 4, 40:] = False
    return Answers(
        lines=tuple(f"L{i}" for i in range(N_LINES)),
        drugs=tuple(f"D{j}" for j in range(N_DRUGS)),
        genes=tuple(f"G{g:03d}" for g in range(N_GENES)),
        delta=delta,
        responding=responding,
    )


def _every_pair() -> tuple[np.ndarray, np.ndarray]:
    lines, drugs = np.divmod(np.arange(N_LINES * N_DRUGS), N_DRUGS)
    return lines, drugs


def _rows(frame: pd.DataFrame, gene_set: str) -> list[tuple[str, str]]:
    """The (line, drug) names of ``frame``'s rows for one gene set, in row order."""
    chosen = frame["gene_set"] == gene_set
    return list(zip(frame.loc[chosen, "line"], frame.loc[chosen, "drug"], strict=True))


# ==============================================================================================
# The score


@pytest.mark.step_score
def test_score_pairs_equals_corrcoef_on_each_scored_pair() -> None:
    """Each scored pair's r is ``np.corrcoef`` of the prediction and the answer over that pair's
    responding genes, or over every gene with a finite answer; ``n_genes`` is that gene count.
    Rows come gene set by gene set, in the order the pairs were passed, and only for pairs the
    masks keep."""
    answers = _fixture_answers()
    masks = scoreable(answers, EXCLUDED)
    lines, drugs = _every_pair()
    prediction = np.random.default_rng(4).standard_normal((lines.size, N_GENES))

    frame = score_pairs(prediction, lines, drugs, answers, masks)

    assert tuple(frame.columns) == SCORE_COLUMNS
    expected: list[tuple[str, str, str, float, int]] = []
    for gene_set, pair in itertools.product(GENE_SETS, range(lines.size)):
        line, drug = int(lines[pair]), int(drugs[pair])
        if not masks[gene_set][line, drug]:
            continue
        answer = answers.delta[line, drug].astype(np.float64)
        select = answers.responding[line, drug] if gene_set == "responding" else np.isfinite(answer)
        r = float(np.corrcoef(prediction[pair, select], answer[select])[0, 1])
        expected.append((f"L{line}", f"D{drug}", gene_set, r, int(select.sum())))

    assert len(frame) == len(expected) == sum(int(masks[g].sum()) for g in GENE_SETS)
    assert list(zip(frame["line"], frame["drug"], frame["gene_set"], strict=True)) == [
        row[:3] for row in expected
    ]
    np.testing.assert_allclose(
        frame["r"].to_numpy(), [row[3] for row in expected], rtol=1e-12, atol=1e-14
    )
    assert frame["n_genes"].tolist() == [row[4] for row in expected]
    assert (frame["n_genes"] >= MIN_GENES).all()


@pytest.mark.step_score
def test_score_pairs_leaves_out_excluded_and_thin_pairs() -> None:
    """With ``scoreable``'s masks, the excluded pair has no row in either gene set, a pair with
    30 responding genes has a row on all genes only, and a pair tested on 40 genes has none."""
    answers = _fixture_answers()
    masks = scoreable(answers, EXCLUDED)
    lines, drugs = _every_pair()
    prediction = np.random.default_rng(5).standard_normal((lines.size, N_GENES))

    frame = score_pairs(prediction, lines, drugs, answers, masks)

    responding, every_gene = _rows(frame, "responding"), _rows(frame, "all")
    assert ("L3", "D0") not in responding and ("L3", "D0") not in every_gene
    assert ("L1", "D2") not in responding and ("L1", "D2") in every_gene
    assert ("L2", "D4") not in responding and ("L2", "D4") not in every_gene
    assert len(responding) == N_LINES * N_DRUGS - 3
    assert len(every_gene) == N_LINES * N_DRUGS - 2
    assert np.isfinite(frame["r"].to_numpy(dtype=np.float64)).all()


@pytest.mark.step_score
def test_score_pairs_gives_nan_to_a_kept_pair_below_min_genes() -> None:
    """A pair the masks keep but that has fewer than 50 genes to correlate gets a row with r NaN
    (the combine drops NaN scores); a pair the masks leave out gets no row. Both ways a pair
    falls short: masks that keep every pair, including the 30-responding-gene pair, and a
    prediction finite on only 45 of a pair's genes."""
    answers = _fixture_answers()
    keep_all = {g: np.ones((N_LINES, N_DRUGS), dtype=bool) for g in GENE_SETS}
    lines, drugs = _every_pair()
    prediction = np.random.default_rng(6).standard_normal((lines.size, N_GENES))
    prediction[0, 45:] = np.nan

    frame = (
        score_pairs(prediction, lines, drugs, answers, keep_all)
        .set_index(["gene_set", "line", "drug"])
        .sort_index()
    )

    assert len(frame) == 2 * N_LINES * N_DRUGS
    thin = frame.loc[("responding", "L1", "D2")]
    assert np.isnan(thin["r"]) and thin["n_genes"] == 30
    assert np.isnan(frame.loc[("all", "L2", "D4"), "r"])
    assert np.isnan(frame.loc[("responding", "L0", "D0"), "r"])
    assert frame.loc[("all", "L0", "D0"), "n_genes"] <= 45
    assert np.isfinite(frame.loc[("responding", "L3", "D0"), "r"])


@pytest.mark.step_score
def test_score_pairs_scores_one_round_as_part_of_the_grid() -> None:
    """A round passes only its own pairs -- one line's drugs when a line is hidden, one drug's
    lines when a drug is hidden -- and each pair's r is the one it gets scored with the whole
    grid, in the round's order."""
    answers = _fixture_answers()
    masks = scoreable(answers, EXCLUDED)
    lines, drugs = _every_pair()
    prediction = np.random.default_rng(7).standard_normal((lines.size, N_GENES))
    whole = (
        score_pairs(prediction, lines, drugs, answers, masks)
        .set_index(["gene_set", "line", "drug"])
        .sort_index()
    )

    rounds = {
        "line": (np.array([2, 2, 2, 2, 2]), np.array([4, 3, 2, 1, 0])),
        "drug": (np.array([3, 1, 0, 2]), np.array([0, 0, 0, 0])),
    }
    frames = {
        name: score_pairs(prediction[i * N_DRUGS + j], i, j, answers, masks)
        for name, (i, j) in rounds.items()
    }
    for name, gene_set in itertools.product(rounds, GENE_SETS):
        round_lines, round_drugs = rounds[name]
        kept = masks[gene_set][round_lines, round_drugs]
        names = [
            (f"L{i}", f"D{j}") for i, j in zip(round_lines[kept], round_drugs[kept], strict=True)
        ]
        frame = frames[name]
        assert _rows(frame, gene_set) == names
        chosen = frame.loc[frame["gene_set"] == gene_set, "r"].to_numpy(dtype=np.float64)
        reference = whole.loc[[(gene_set, *pair) for pair in names], "r"]
        np.testing.assert_array_equal(chosen, reference.to_numpy(dtype=np.float64))


@pytest.mark.step_score
def test_score_pairs_rejects_misshapen_inputs() -> None:
    answers = _fixture_answers()
    masks = scoreable(answers, EXCLUDED)
    lines, drugs = _every_pair()
    prediction = np.zeros((lines.size, N_GENES))
    with pytest.raises(ValueError, match="genes"):
        score_pairs(prediction[:, :-1], lines, drugs, answers, masks)
    with pytest.raises(ValueError, match="one line and one drug index per"):
        score_pairs(prediction, lines[:-1], drugs, answers, masks)
    with pytest.raises(ValueError, match="outside"):
        score_pairs(prediction, lines + 1, drugs, answers, masks)
    with pytest.raises(ValueError, match="masks"):
        score_pairs(prediction, lines, drugs, answers, {"all": masks["all"]})


@pytest.mark.step_score
def test_fraction_of_ceiling_divides_by_the_square_root_ceiling() -> None:
    assert fraction_of_ceiling(0.3876 / 2, 0.3876) == 0.5
    assert fraction_of_ceiling(0.8575, 0.8575) == 1.0
    with pytest.raises(ValueError, match="positive"):
        fraction_of_ceiling(0.1, 0.0)


# ==============================================================================================
# Redraws of the held-out units


def _unit_weights(n_units: int, n_draws: int, seed: int) -> np.ndarray:
    """The multinomial unit counts the redraws use, drawn afresh from ``seed``."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_draws)


def _dyadic_pairs(n_units: int, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(diffs, unit_index)``: per-pair differences in multiples of 1/8 (every sum exact in
    float64) and a unit per pair, every unit holding at least one pair except the last."""
    rng = np.random.default_rng(seed)
    diffs = rng.integers(-40, 41, size=n_pairs) / 8.0
    units = np.concatenate(
        [np.arange(n_units - 1), rng.integers(0, n_units - 1, n_pairs - n_units + 1)]
    )
    return diffs, units


@pytest.mark.step_null
def test_redraw_estimates_equal_brute_force_resampling() -> None:
    """Each draw is the mean over every pair of the redrawn units, a unit drawn k times
    contributing its pairs k times: ``np.repeat`` of the pairs by their unit's count, then the
    mean. Exact on dyadic values (a unit with no pairs included); to 1e-12 on arbitrary ones."""
    n_units, n_draws, seed = 9, 300, 21
    diffs, units = _dyadic_pairs(n_units, 40, seed=1)
    weights = _unit_weights(n_units, n_draws, seed)
    brute = np.array(
        [diffs[np.repeat(np.arange(diffs.size), weights[b, units])].mean() for b in range(n_draws)]
    )
    np.testing.assert_array_equal(redraw_estimates(diffs, units, n_units, n_draws, seed), brute)

    noisy = np.random.default_rng(2).standard_normal(diffs.size)
    brute_noisy = np.array(
        [noisy[np.repeat(np.arange(noisy.size), weights[b, units])].mean() for b in range(n_draws)]
    )
    np.testing.assert_allclose(
        redraw_estimates(noisy, units, n_units, n_draws, seed), brute_noisy, rtol=1e-12
    )


@pytest.mark.step_null
def test_a_draw_with_no_pairs_is_nan_and_dropped_from_the_summary() -> None:
    """Only one of three units holds pairs, so a draw that never picks it averages nothing: its
    estimate is NaN, and the summary leaves it out and counts it."""
    n_units, n_draws, seed = 3, 400, 5
    diffs = np.array([0.5, 1.5, 1.0])
    units = np.zeros(3, dtype=np.int64)
    weights = _unit_weights(n_units, n_draws, seed)
    empty = weights[:, 0] == 0
    assert 0 < empty.sum() < n_draws

    draws = redraw_estimates(diffs, units, n_units, n_draws, seed)
    assert np.array_equal(np.isnan(draws), empty)
    assert (draws[~empty] == 1.0).all()
    summary = summarize_redraws(1.0, draws)
    assert summary["n_dropped"] == int(empty.sum())
    assert summary["n_draws"] == n_draws - int(empty.sum())


@pytest.mark.step_null
def test_summarize_redraws_on_a_hand_example() -> None:
    """Ten draws -0.2, -0.1, ..., 0.7 and one NaN.

    * interval: numpy's linear percentile, positions 0.025 x 9 = 0.225 and 0.975 x 9 = 8.775,
      so -0.2 + 0.225 x 0.1 = -0.1775 and 0.6 + 0.775 x 0.1 = 0.6775;
    * p: 3 draws are <= 0 and 8 are >= 0, so 2 x min(4/11, 9/11) = 8/11;
    * sd: an evenly spaced run of n = 10 with step 0.1 has sd 0.1 x sqrt(n(n + 1)/12);
    * mde = 2.8 x sd; the NaN is dropped and counted.
    """
    draws = np.append(np.arange(-2, 8) / 10.0, np.nan)
    summary = summarize_redraws(0.25, draws)
    sd = 0.1 * np.sqrt(10 * 11 / 12)
    assert summary["estimate"] == 0.25
    assert summary["ci_lo"] == pytest.approx(-0.1775, abs=1e-12)
    assert summary["ci_hi"] == pytest.approx(0.6775, abs=1e-12)
    assert summary["p"] == pytest.approx(8 / 11, rel=1e-12)
    assert summary["sd"] == pytest.approx(sd, rel=1e-12)
    assert summary["mde"] == pytest.approx(MDE_FACTOR * sd, rel=1e-12)
    assert (summary["n_draws"], summary["n_dropped"]) == (10, 1)

    assert summarize_redraws(1.0, np.linspace(0.1, 1.0, 10))["p"] == pytest.approx(2 / 11)
    assert summarize_redraws(-1.0, -np.linspace(0.1, 1.0, 10))["p"] == pytest.approx(2 / 11)
    assert summarize_redraws(0.0, np.zeros(10))["p"] == 1.0
    with pytest.raises(ValueError, match="at least 2"):
        summarize_redraws(0.0, np.array([0.1, np.nan]))


@pytest.mark.step_null
def test_holm_on_a_hand_example() -> None:
    """p = (0.01, 0.04, 0.03, 0.005). Sorted: 0.005, 0.01, 0.03, 0.04, scaled by 4, 3, 2, 1 to
    0.02, 0.03, 0.06, 0.04; the running maximum makes the last 0.06. Back in input order:
    (0.03, 0.06, 0.06, 0.02). Scaled values above 1 are capped at 1."""
    np.testing.assert_allclose(
        holm(np.array([0.01, 0.04, 0.03, 0.005])), [0.03, 0.06, 0.06, 0.02], rtol=1e-12
    )
    np.testing.assert_array_equal(holm(np.array([0.5, 0.6])), [1.0, 1.0])
    np.testing.assert_array_equal(holm(np.array([0.2])), [0.2])
    assert holm(np.array([])).size == 0
    with pytest.raises(ValueError, match="between 0 and 1"):
        holm(np.array([0.1, np.nan]))


@pytest.mark.step_null
def test_two_way_estimates_equal_brute_force_resampling() -> None:
    """Lines and drugs redrawn together: a pair's weight is its line's count times its drug's
    count (line counts drawn first, then drug counts, from one generator). Brute force repeats
    each pair by that product and averages. On a 6 x 7 grid with pairs missing, exact on dyadic
    values."""
    n_lines, n_drugs, n_draws, seed = 6, 7, 250, 8
    rng = np.random.default_rng(9)
    present = np.flatnonzero(rng.random(n_lines * n_drugs) < 0.8)
    lines, drugs = np.divmod(present, n_drugs)
    diffs = rng.integers(-40, 41, size=present.size) / 8.0
    sampler = np.random.default_rng(seed)
    line_weights = sampler.multinomial(n_lines, np.full(n_lines, 1 / n_lines), size=n_draws)
    drug_weights = sampler.multinomial(n_drugs, np.full(n_drugs, 1 / n_drugs), size=n_draws)

    brute = np.full(n_draws, np.nan)
    for b in range(n_draws):
        repeats = line_weights[b, lines] * drug_weights[b, drugs]
        if repeats.sum() > 0:
            brute[b] = diffs[np.repeat(np.arange(diffs.size), repeats)].mean()

    np.testing.assert_array_equal(
        two_way_estimates(diffs, lines, drugs, n_lines, n_drugs, n_draws, seed), brute
    )


@pytest.mark.step_null
def test_design_effect_is_the_ratio_of_redraw_variances() -> None:
    assert design_effect(np.array([1.0, 3.0, np.nan]), np.array([1.0, 2.0])) == pytest.approx(4.0)


@pytest.mark.step_null
def test_redraws_are_seeded() -> None:
    """Invariant 4: the same seed gives the same draws, one way and two way, and a different seed
    different ones. Run in blocks, each block seeded ``base + block``, the blocks computed in any
    order and concatenated in block order equal one in-process call over every block."""
    diffs, units = _dyadic_pairs(12, 60, seed=3)
    drugs = np.arange(diffs.size) % 5

    def one_way(n_draws: int, seed: int) -> np.ndarray:
        return redraw_estimates(diffs, units, 12, n_draws, seed)

    def two_way(n_draws: int, seed: int) -> np.ndarray:
        return two_way_estimates(diffs, units, drugs, 12, 5, n_draws, seed)

    assert (N_BLOCKS, DRAWS_PER_BLOCK, N_DRAWS) == (8, 250, 2000)
    assert redraw_block_seed(40, 3) == 43
    for draw in (one_way, two_way):
        np.testing.assert_array_equal(draw(100, 7), draw(100, 7))
        assert not np.array_equal(draw(100, 7), draw(100, 8))
        separately = {b: draw(25, redraw_block_seed(40, b)) for b in reversed(range(4))}
        combined = in_blocks(draw, n_blocks=4, draws_per_block=25, base_seed=40)
        np.testing.assert_array_equal(
            combined, concatenate_blocks([separately[b] for b in range(4)])
        )
        assert combined.shape == (100,)


@pytest.mark.step_null
def test_mean_score_ci_redraws_the_units_of_a_model_mean() -> None:
    scores, units = _dyadic_pairs(10, 50, seed=4)
    summary = mean_score_ci(scores, units, 10, 500, 3)
    expected = summarize_redraws(float(scores.mean()), redraw_estimates(scores, units, 10, 500, 3))
    assert summary == expected


@pytest.mark.step_null
def test_redraws_reject_misshapen_inputs() -> None:
    diffs, units = _dyadic_pairs(5, 20, seed=5)
    with pytest.raises(ValueError, match="one unit index per"):
        redraw_estimates(diffs, units[:-1], 5, 10, 0)
    with pytest.raises(ValueError, match="outside"):
        redraw_estimates(diffs, units + 1, 4, 10, 0)
    with pytest.raises(ValueError, match="outside"):
        two_way_estimates(diffs, units, units + 1, 5, 4, 10, 0)
