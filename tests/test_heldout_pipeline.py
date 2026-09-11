"""Rung 1's run: one round of fitting and scoring, and one block of redraws (task 10a).

The two runners are ``scripts/heldout_fit.py`` (a round: every model declared for the scheme,
fitted, scored on both gene sets, with the settings its tuning chose) and
``scripts/heldout_redraws.py`` (a block of redraws of every comparison). Both are loaded by
path, as ``scripts`` is not a package, and exercised on a 6-line x 8-drug fixture cache built
here -- the real cache's files, at a size that fits in a test.

What is checked:

* **Invariant 8, staged equals one process.** 6 leave-one-line-out rounds + 8 leave-one-drug-out
  rounds + 4 redraw blocks, each run through its command line one at a time, produce files
  byte-identical to a single process that loads the inputs once and runs all of them
  (``test_staged_run_equals_one_process``).
* **A finished round, or block, skips.** A second call recomputes nothing: the stage function is
  replaced by one that raises, and the outputs are untouched.
* **Invariant 4, determinism.** A redraw block is bit-identical on a rerun, and the blocks are
  independent: running them in reverse order gives the same files.
* **The wiring is right, not merely reproducible.** One round's drug-average scores are
  recomputed here from the answer array with ``numpy`` alone, and one comparison's block-0
  redraws from the scored pairs with ``redraw_estimates``; the excluded pair is scored for no
  model; every declared model has scores and a settings row; every declared comparison and every
  description-versus-its-stand-in contrast is redrawn on both gene sets; and the stand-ins the
  fit draws are the ones ``descriptions.npz`` holds (one seed source, ``RANDOM_SEEDS``).

Tolerances are the module constants below, set before any test runs (invariant 9). The parity
and determinism checks need none: they compare bytes.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from fmharness.heldout import COMPARISONS, MODELS
from fmharness.heldout.chemistry import tanimoto
from fmharness.heldout.comparisons import DRAWS_PER_BLOCK, redraw_block_seed, redraw_estimates
from fmharness.heldout.controls import synthetic_lolo_grid
from fmharness.heldout.descriptions import (
    RANDOM_SEEDS,
    pca_components,
    random_stand_in,
    standardize,
)
from fmharness.heldout.grid import load_grid
from fmharness.heldout.records import sha256_file
from fmharness.heldout.scoring import GENE_SETS

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hf = _load_module("heldout_fit", "scripts/heldout_fit.py")
hr = _load_module("heldout_redraws", "scripts/heldout_redraws.py")

#: The fixture grid: small enough to fit 14 rounds in a test, wide enough that every pair clears
#: the 50-gene minimum on both gene sets.
N_LINES, N_DRUGS, N_GENES = 6, 8, 120
N_RESPONDING = 80
EXPRESSION_WIDTH, STACK_WIDTH, N_BITS = 40, 12, 64
NMF_KS: tuple[int, ...] = (2, 5)
N_BLOCKS_TESTED = 4

LINES: tuple[str, ...] = tuple(f"ACH-{i:06d}" for i in range(N_LINES))
#: One drug name carries a trailing space, as the real grid's screen names do.
DRUGS: tuple[str, ...] = tuple(f"Drug-{j}" + (" " if j == 5 else "") for j in range(N_DRUGS))
GENES: tuple[str, ...] = tuple(f"G{g:03d}" for g in range(N_GENES))
EXCLUDED_PAIR = (LINES[2], DRUGS[3])
#: (LINES[0], DRUGS[0]) is untested from this gene on, so one pair carries untested genes.
UNTESTED_FROM = 100

STRENGTH, NOISE, GRID_SEED, FIXTURE_SEED = 0.8, 0.5, 11, 17

#: A per-pair score recomputed here with ``np.corrcoef`` may differ from the shipped
#: ``masked_rowwise_pearson`` only by floating-point association.
R_TOLERANCE = 1e-12

#: A description and its random stand-in are different fits, so their per-pair scores must differ
#: by more than this somewhere on the fixture. (The reviewer measured 0.0151 for expression and
#: 0.0062 for stack_base, against exactly 0.0 for the two whose stand-in was misdispatched.)
MIN_STAND_IN_DIFFERENCE = 1e-6

#: How many distinct per-pair score vectors the 14 leave-one-line-out models give on this
#: fixture, fixed before the test runs. It is 13, not 14, for a reason particular to a 6-line
#: grid: nearest lines has 5 training lines to rank, and its tuning picks k = 5 in every round,
#: so "the mean over the k most similar training lines" is the mean over all of them -- the drug
#: average exactly, a documented property of ``nearest_lines_lolo``. At the design's 50-line grid
#: there are 49 training lines and no candidate k (3, 5, 10, 20) can span them.
EXPECTED_DISTINCT_LOLO_MODELS = 13

#: The one pair of leave-one-line-out models allowed to coincide on this fixture, for that
#: reason. Any other pair coinciding is two models collapsed onto one fit.
COINCIDING_LOLO_MODELS = ("drug_average", "nearest_lines")


# ==============================================================================================
# The fixture cache: exactly the files a round reads, at 6 lines x 8 drugs.


def _write_fixture(cache: Path) -> Path:
    """Write ``rung1_grid.json``, ``answers.npz``, ``descriptions.npz``, ``tanimoto.npz`` and the
    three ``embedding_{version}.parquet`` files into ``cache``; return the grid's path."""
    cache.mkdir(parents=True, exist_ok=True)
    grid_path = cache / "rung1_grid.json"
    grid_path.write_text(
        json.dumps(
            {
                "dose": 5.0,
                "lines": list(LINES),
                "drugs": list(DRUGS),
                "metadata_name": {drug: drug.strip() for drug in DRUGS},
                "excluded_pairs": [list(EXCLUDED_PAIR)],
            },
            indent=2,
        )
        + "\n"
    )

    planted = synthetic_lolo_grid(
        n_lines=N_LINES,
        n_drugs=N_DRUGS,
        n_genes=N_GENES,
        width=EXPRESSION_WIDTH,
        strength=STRENGTH,
        noise=NOISE,
        seed=GRID_SEED,
    )
    delta = planted.delta.astype(np.float32)
    delta[0, 0, UNTESTED_FROM:] = np.nan
    responding = np.zeros(delta.shape, dtype=bool)
    responding[:, :, :N_RESPONDING] = True
    responding &= np.isfinite(delta)
    np.savez(
        cache / "answers.npz",
        lines=np.asarray(LINES),
        drugs=np.asarray(DRUGS),
        genes=np.asarray(GENES),
        delta=delta,
        responding=responding,
    )

    rng = np.random.default_rng(FIXTURE_SEED)
    expression = standardize(planted.description)
    pca_scores, pca_loadings = pca_components(expression, k_max=20)
    pca = standardize(pca_scores)
    descriptions: dict[str, np.ndarray] = {
        "lines": np.asarray(LINES),
        "expression": expression,
        "expression_genes": np.asarray([f"panel_{g}" for g in range(EXPRESSION_WIDTH)]),
        "pca": pca,
        "pca_loadings": pca_loadings,
        **{f"nmf_{k}": standardize(rng.random((N_LINES, k))) for k in NMF_KS},
        "random_expression": random_stand_in(N_LINES, EXPRESSION_WIDTH, RANDOM_SEEDS["expression"]),
        "random_pca": random_stand_in(N_LINES, pca.shape[1], RANDOM_SEEDS["pca"]),
        "random_nmf": random_stand_in(N_LINES, max(NMF_KS), RANDOM_SEEDS["nmf"]),
    }
    np.savez(cache / "descriptions.npz", **cast(dict[str, Any], descriptions))

    fingerprints = rng.random((N_DRUGS, N_BITS)) < 0.3
    fingerprints[:, 0] = True
    np.savez(
        cache / "tanimoto.npz",
        drugs=np.asarray(DRUGS),
        similarity=tanimoto(fingerprints),
        fingerprints=fingerprints,
    )

    columns = [f"dim_{i}" for i in range(STACK_WIDTH)]
    for version in ("base", "cytokine", "drug"):
        embedding = pd.DataFrame(
            rng.standard_normal((N_LINES, STACK_WIDTH)),
            index=pd.Index(LINES, name="line"),
            columns=columns,
        )
        embedding.to_parquet(cache / f"embedding_{version}.parquet")

    return grid_path


# ==============================================================================================
# Running the two stages: one round (or block) per command line, or all of them in one process.


def _run_cli(mp: pytest.MonkeyPatch, module: ModuleType, args: list[str]) -> None:
    mp.setattr(sys, "argv", ["x", *args])
    module.main()


def _stage_by_stage(cache: Path, grid_path: Path) -> None:
    """Every round and redraw block through its own command line, one at a time."""
    with pytest.MonkeyPatch.context() as mp:
        for scheme, n_rounds in (("lolo", N_LINES), ("lodo", N_DRUGS)):
            for index in range(n_rounds):
                _run_cli(
                    mp,
                    hf,
                    [
                        "--scheme",
                        scheme,
                        "--round",
                        str(index),
                        "--grid",
                        str(grid_path),
                        "--cache",
                        str(cache),
                    ],
                )
        for block in range(N_BLOCKS_TESTED):
            _run_cli(
                mp,
                hr,
                ["--block", str(block), "--grid", str(grid_path), "--cache", str(cache)],
            )


def _in_one_process(cache: Path, grid_path: Path) -> None:
    """The same rounds and blocks in a single process, loading the inputs once."""
    grid = load_grid(grid_path)
    inputs = hf.load_inputs(grid, cache)
    component_ks = hf.realized_component_ks(inputs)
    for scheme, n_rounds in (("lolo", N_LINES), ("lodo", N_DRUGS)):
        for index in range(n_rounds):
            scores, settings = hf.run_round(scheme, index, inputs)
            hf.write_round(cache, scheme, index, scores, settings, component_ks)
    pair_scores = hr.load_pair_scores(cache, grid)
    for block in range(N_BLOCKS_TESTED):
        hr.write_block(
            cache,
            block,
            hr.run_redraw_block(pair_scores, block, grid=grid),
            redraw_block_seed(hr.REDRAW_BASE_SEED, block),
        )


def _output_names() -> list[str]:
    """Every file the two stages write, with its completion record, in a fixed order."""
    names: list[str] = []
    for scheme, n_rounds in (("lolo", N_LINES), ("lodo", N_DRUGS)):
        for index in range(n_rounds):
            names.append(f"scores_{scheme}_{index:03d}.parquet")
            names.append(f"settings_{scheme}_{index:03d}.csv")
    names.extend(f"redraws_{block}.parquet" for block in range(N_BLOCKS_TESTED))
    return names + [f"{name}.done.json" for name in names]


def _all_scores(cache: Path) -> pd.DataFrame:
    parts = [
        pd.read_parquet(cache / f"scores_{scheme}_{index:03d}.parquet")
        for scheme, n_rounds in (("lolo", N_LINES), ("lodo", N_DRUGS))
        for index in range(n_rounds)
    ]
    return pd.concat(parts, ignore_index=True)


def _all_settings(cache: Path) -> pd.DataFrame:
    parts = [
        pd.read_csv(cache / f"settings_{scheme}_{index:03d}.csv", float_precision="round_trip")
        for scheme, n_rounds in (("lolo", N_LINES), ("lodo", N_DRUGS))
        for index in range(n_rounds)
    ]
    return pd.concat(parts, ignore_index=True)


def _all_redraws(cache: Path) -> pd.DataFrame:
    parts = [
        pd.read_parquet(cache / f"redraws_{block}.parquet") for block in range(N_BLOCKS_TESTED)
    ]
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def run_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One fixture cache run twice: stage by stage, and in a single process."""
    root = tmp_path_factory.mktemp("rung1_pipeline")
    staged, single = root / "staged", root / "single"
    grid_path = _write_fixture(staged)
    shutil.copytree(staged, single)
    _stage_by_stage(staged, grid_path)
    _in_one_process(single, single / "rung1_grid.json")
    return {"staged": staged, "single": single, "grid": grid_path}


# ==============================================================================================
# Invariant 8: staged equals one process


@pytest.mark.step_fit
def test_staged_run_equals_one_process(run_dirs: dict[str, Path]) -> None:
    """6 + 8 rounds and 4 redraw blocks, one command line each, write the same bytes as one
    process that runs all of them -- every table and every completion record."""
    staged, single = run_dirs["staged"], run_dirs["single"]
    names = _output_names()
    assert len(names) == 2 * (2 * (N_LINES + N_DRUGS) + N_BLOCKS_TESTED)
    for name in names:
        staged_path, single_path = staged / name, single / name
        assert staged_path.exists(), f"the staged run did not write {name}"
        assert single_path.exists(), f"the one-process run did not write {name}"
        assert sha256_file(staged_path) == sha256_file(single_path), f"{name} differs"

    scores = _all_scores(staged)
    assert list(scores.columns) == [
        "scheme",
        "round",
        "model",
        "line",
        "drug",
        "gene_set",
        "r",
        "n_genes",
    ]
    pd.testing.assert_frame_equal(scores, _all_scores(single), check_exact=True)
    pd.testing.assert_frame_equal(_all_settings(staged), _all_settings(single), check_exact=True)
    pd.testing.assert_frame_equal(_all_redraws(staged), _all_redraws(single), check_exact=True)


@pytest.mark.step_fit
def test_a_finished_round_is_not_recomputed(run_dirs: dict[str, Path]) -> None:
    """A round whose outputs are already done skips: the fit is never called, and neither output
    nor its record is rewritten."""
    staged, grid_path = run_dirs["staged"], run_dirs["grid"]
    names = ["scores_lolo_000.parquet", "settings_lolo_000.csv"]
    before = {
        name: (sha256_file(staged / name), (staged / name).stat().st_mtime_ns) for name in names
    }

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a finished round must not be recomputed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hf, "run_round", refuse)
        _run_cli(
            mp,
            hf,
            ["--scheme", "lolo", "--round", "0", "--grid", str(grid_path), "--cache", str(staged)],
        )

    for name in names:
        assert (sha256_file(staged / name), (staged / name).stat().st_mtime_ns) == before[name]


@pytest.mark.step_null
def test_a_finished_redraw_block_is_not_recomputed(run_dirs: dict[str, Path]) -> None:
    """A redraw block whose output is already done skips, the same way."""
    staged, grid_path = run_dirs["staged"], run_dirs["grid"]
    name = "redraws_0.parquet"
    before = (sha256_file(staged / name), (staged / name).stat().st_mtime_ns)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a finished redraw block must not be recomputed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hr, "run_redraw_block", refuse)
        _run_cli(mp, hr, ["--block", "0", "--grid", str(grid_path), "--cache", str(staged)])

    assert (sha256_file(staged / name), (staged / name).stat().st_mtime_ns) == before


# ==============================================================================================
# Invariant 4: the redraws are seeded, and the blocks do not depend on the order they run in


@pytest.mark.step_null
def test_redraw_blocks_are_seeded_and_independent_of_order(
    run_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """Rerunning the blocks -- in reverse order, in a fresh copy of the same scores -- writes
    byte-identical files: every draw comes from its block's seed alone."""
    staged = run_dirs["staged"]
    reverse = tmp_path / "reverse"
    shutil.copytree(staged, reverse)
    for block in range(N_BLOCKS_TESTED):
        (reverse / f"redraws_{block}.parquet").unlink()
        (reverse / f"redraws_{block}.parquet.done.json").unlink()

    grid_path = reverse / "rung1_grid.json"
    with pytest.MonkeyPatch.context() as mp:
        for block in reversed(range(N_BLOCKS_TESTED)):
            _run_cli(
                mp,
                hr,
                ["--block", str(block), "--grid", str(grid_path), "--cache", str(reverse)],
            )

    for block in range(N_BLOCKS_TESTED):
        name = f"redraws_{block}.parquet"
        assert sha256_file(reverse / name) == sha256_file(staged / name)


@pytest.mark.step_null
def test_block_seeds_come_from_the_base_seed(run_dirs: dict[str, Path]) -> None:
    """Block ``b``'s draws are ``redraw_estimates`` at ``redraw_block_seed(REDRAW_BASE_SEED, b)``:
    recomputed here for one comparison from the scored pairs, bit for bit. ``draw`` numbers the
    draws across blocks, so the blocks concatenate into 0..N-1."""
    staged = run_dirs["staged"]
    scores = _all_scores(staged)
    wide = _wide(scores, "lolo", "responding")
    diffs = (wide["stack_base"] - wide["drug_average"]).to_numpy(dtype=np.float64)
    unit = np.array([LINES.index(line) for line, _ in wide.index], dtype=np.int64)

    redraws = _all_redraws(staged)
    chosen = (redraws["comparison"] == "lolo_stack_base_vs_drug_average") & (
        redraws["gene_set"] == "responding"
    )
    table = redraws.loc[chosen].sort_values("draw")
    assert table["draw"].tolist() == list(range(N_BLOCKS_TESTED * DRAWS_PER_BLOCK))

    expected = np.concatenate(
        [
            redraw_estimates(
                diffs,
                unit,
                N_LINES,
                DRAWS_PER_BLOCK,
                redraw_block_seed(hr.REDRAW_BASE_SEED, block),
            )
            for block in range(N_BLOCKS_TESTED)
        ]
    )
    np.testing.assert_array_equal(table["estimate"].to_numpy(dtype=np.float64), expected)


# ==============================================================================================
# The wiring: what a round scores, and what a block redraws


def _wide(scores: pd.DataFrame, scheme: str, gene_set: str) -> pd.DataFrame:
    chosen = (scores["scheme"] == scheme) & (scores["gene_set"] == gene_set)
    return scores.loc[chosen].pivot(index=["line", "drug"], columns="model", values="r")


@pytest.mark.step_fit
def test_every_declared_model_is_fitted_scored_and_recorded(run_dirs: dict[str, Path]) -> None:
    """Each scheme runs exactly the models declared for it in ``MODELS``: every one has scored
    pairs on both gene sets in every round, and a settings row per round."""
    staged = run_dirs["staged"]
    scores = _all_scores(staged)
    for scheme, n_rounds in (("lolo", N_LINES), ("lodo", N_DRUGS)):
        declared = [spec.id for spec in MODELS.values() if scheme in spec.schemes]
        assert declared, f"no models declared for {scheme}"
        scheme_scores = scores.loc[scores["scheme"] == scheme]
        assert set(scheme_scores["model"]) == set(declared)
        for index in range(n_rounds):
            settings = pd.read_csv(
                staged / f"settings_{scheme}_{index:03d}.csv", float_precision="round_trip"
            )
            assert settings["model"].tolist() == declared
        for gene_set in GENE_SETS:
            counts = (
                scheme_scores.loc[scheme_scores["gene_set"] == gene_set]
                .groupby(["round", "model"])
                .size()
            )
            assert set(counts.index.get_level_values("model")) == set(declared)
            assert counts.index.get_level_values("round").nunique() == n_rounds


@pytest.mark.step_fit
def test_settings_hold_one_row_per_model_and_round(run_dirs: dict[str, Path]) -> None:
    """``settings_{scheme}_{i}.csv`` holds the contract's columns, one row per model fitted in
    that round, with the penalty and component count its tuning chose."""
    staged = run_dirs["staged"]
    table = pd.read_csv(staged / "settings_lolo_000.csv", float_precision="round_trip")
    assert list(table.columns) == ["model", "lambda", "k", "loss_min", "lambda_at_edge"]
    declared = [spec.id for spec in MODELS.values() if "lolo" in spec.schemes]
    assert table["model"].tolist() == declared
    ridge = table.set_index("model").loc["stack_base"]
    assert np.isfinite(float(ridge["lambda"])) and float(ridge["lambda"]) > 0
    assert np.isfinite(float(ridge["loss_min"]))
    assert isinstance(bool(ridge["lambda_at_edge"]), bool)
    assert np.isfinite(float(table.set_index("model").loc["nmf", "k"]))
    # Ruling 35: a stand-in of a tuned description is tuned over the same counts, so it reports a
    # chosen k of its own -- drawn from its own stand-in, not from the description's components.
    assert np.isfinite(float(table.set_index("model").loc["random_nmf", "k"]))
    assert not np.isfinite(float(table.set_index("model").loc["drug_average", "lambda"]))

    # The chosen k means nothing without the set it was chosen from, so the round records it.
    record = json.loads((staged / "settings_lolo_000.csv.done.json").read_text())
    assert record["component_ks"] == {
        model: list(NMF_KS) for model in ("nmf", "pca", "random_nmf", "random_pca")
    }


@pytest.mark.step_score
def test_drug_average_scores_match_an_independent_recomputation(
    run_dirs: dict[str, Path],
) -> None:
    """One leave-one-line-out round's drug-average scores, recomputed here from ``answers.npz``
    with numpy alone: the mean over the 5 training lines, correlated with the hidden line's
    answer over its responding genes."""
    staged = run_dirs["staged"]
    held = 1
    with np.load(staged / "answers.npz") as npz:
        delta = npz["delta"].astype(np.float64)
        responding = npz["responding"]
    train = np.array([i for i in range(N_LINES) if i != held])
    prediction = np.nan_to_num(delta, nan=0.0)[train].mean(axis=0)

    scores = _all_scores(staged)
    rows = scores.loc[
        (scores["scheme"] == "lolo")
        & (scores["round"] == held)
        & (scores["model"] == "drug_average")
        & (scores["gene_set"] == "responding")
    ]
    assert len(rows) == N_DRUGS
    for drug, r in zip(rows["drug"], rows["r"], strict=True):
        d = DRUGS.index(drug)
        select = responding[held, d] & np.isfinite(delta[held, d])
        expected = float(np.corrcoef(prediction[d, select], delta[held, d, select])[0, 1])
        assert abs(float(r) - expected) <= R_TOLERANCE, f"{drug}: {r} against {expected}"


@pytest.mark.step_score
def test_the_excluded_pair_is_scored_for_no_model(run_dirs: dict[str, Path]) -> None:
    """The leaked pair is removed for every model, in both schemes and on both gene sets."""
    scores = _all_scores(run_dirs["staged"])
    line, drug = EXCLUDED_PAIR
    assert not ((scores["line"] == line) & (scores["drug"] == drug)).any()
    assert len(scores) > 0


@pytest.mark.step_null
def test_every_comparison_and_stand_in_contrast_is_redrawn(run_dirs: dict[str, Path]) -> None:
    """A block covers every comparison declared in ``COMPARISONS`` and, for both schemes, each
    description against its own random stand-in, on both gene sets."""
    block = pd.read_parquet(run_dirs["staged"] / "redraws_0.parquet")
    assert list(block.columns) == [
        "comparison",
        "gene_set",
        "draw",
        "estimate",
        "estimate_two_way",
    ]
    ids = set(block["comparison"])
    for comparison in COMPARISONS:
        assert comparison.id in ids, f"{comparison.id} was not redrawn"
    for scheme in ("lolo", "lodo"):
        for spec in MODELS.values():
            if spec.kind == "random" or scheme not in spec.schemes or spec.description is None:
                continue
            if spec.kind != "ridge":
                continue
            assert f"{scheme}_{spec.id}_vs_random_{spec.id}" in ids
    counts = block.groupby(["comparison", "gene_set"]).size()
    assert set(counts.unique().tolist()) == {DRAWS_PER_BLOCK}
    assert len(counts) == len(ids) * len(GENE_SETS)
    assert np.isfinite(block["estimate"].to_numpy(dtype=np.float64)).all()
    assert np.isfinite(block["estimate_two_way"].to_numpy(dtype=np.float64)).all()


@pytest.mark.step_fit
def test_stand_ins_come_from_the_declared_seeds(run_dirs: dict[str, Path]) -> None:
    """Every random stand-in the fit uses is drawn through one helper from
    ``descriptions.RANDOM_SEEDS`` -- the same numbers ``descriptions.npz`` holds for the three
    descriptions it stores a stand-in for, and the declared seed for each Stack version."""
    staged = run_dirs["staged"]
    with np.load(staged / "descriptions.npz") as npz:
        stored = {
            "expression": npz["random_expression"],
            "pca": npz["random_pca"],
            "nmf": npz["random_nmf"],
        }
    for name, array in stored.items():
        drawn = hf.stand_in(name, N_LINES, array.shape[1])
        np.testing.assert_array_equal(drawn, array)
    for version in ("base", "cytokine", "drug"):
        name = f"stack_{version}"
        np.testing.assert_array_equal(
            hf.stand_in(name, N_LINES, STACK_WIDTH),
            random_stand_in(N_LINES, STACK_WIDTH, RANDOM_SEEDS[name]),
        )


@pytest.mark.step_fit
def test_a_description_and_its_stand_in_are_different_fits(run_dirs: dict[str, Path]) -> None:
    """Every description is fitted on what it describes, and its stand-in on random numbers of
    the same shape -- including PCA and NMF, whose stand-in is tuned over the same candidate
    component counts (ruling 35) but from its own matrices.

    Fitted on the description's own components a stand-in would score identically on every pair,
    and its contrast would be an exact zero reported as "the description adds nothing over random
    noise".
    """
    scores = _all_scores(run_dirs["staged"])
    for scheme in ("lolo", "lodo"):
        wide = _wide(scores, scheme, "responding")
        for spec in MODELS.values():
            if spec.kind != "ridge" or scheme not in spec.schemes:
                continue
            twin = f"random_{spec.id}"
            assert twin in wide.columns, f"{scheme}: {twin} was not scored"
            difference = float((wide[spec.id] - wide[twin]).abs().max())
            assert difference > MIN_STAND_IN_DIFFERENCE, (
                f"{scheme}: {spec.id} and {twin} score identically on every pair (largest "
                f"difference {difference}), so the stand-in was not fitted on its own matrices"
            )


@pytest.mark.step_null
def test_stand_in_contrasts_are_not_identically_zero(run_dirs: dict[str, Path]) -> None:
    """The consequence, in the table that gets reported: a description-versus-stand-in contrast
    has a spread and a non-zero estimate, not the exact zero a misdispatched stand-in gives."""
    block = pd.read_parquet(run_dirs["staged"] / "redraws_0.parquet")
    for scheme in ("lolo", "lodo"):
        for description in ("pca", "nmf"):
            chosen = (block["comparison"] == f"{scheme}_{description}_vs_random_{description}") & (
                block["gene_set"] == "responding"
            )
            estimates = block.loc[chosen, "estimate"].to_numpy(dtype=np.float64)
            assert estimates.size == DRAWS_PER_BLOCK
            assert np.abs(estimates).max() > MIN_STAND_IN_DIFFERENCE
            assert float(estimates.std(ddof=1)) > 0.0


@pytest.mark.step_fit
def test_the_lolo_models_give_distinct_predictions(run_dirs: dict[str, Path]) -> None:
    """No two leave-one-line-out models collapse onto the same fit. A mistyped model id, a kernel
    reused for two models, or a stand-in dispatched to its description's matrices all show up
    here as two identical per-pair score vectors."""
    wide = _wide(_all_scores(run_dirs["staged"]), "lolo", "responding")
    declared = [spec.id for spec in MODELS.values() if "lolo" in spec.schemes]
    assert sorted(wide.columns) == sorted(declared)
    vectors = {tuple(wide[model].to_numpy(dtype=np.float64).tolist()) for model in declared}
    assert len(vectors) == EXPECTED_DISTINCT_LOLO_MODELS, (
        f"{len(declared)} leave-one-line-out models gave {len(vectors)} distinct per-pair score "
        f"vectors, not {EXPECTED_DISTINCT_LOLO_MODELS}: models collapsed onto one fit"
    )

    # Which pair coincides is pinned too, so the one coincidence this fixture's size forces
    # cannot stand in for a different pair collapsing.
    coinciding = {
        (a, b)
        for a, b in itertools.combinations(declared, 2)
        if np.array_equal(wide[a].to_numpy(dtype=np.float64), wide[b].to_numpy(dtype=np.float64))
    }
    assert coinciding == {COINCIDING_LOLO_MODELS}, (
        f"the models that score identically are {sorted(coinciding)}, not "
        f"{[COINCIDING_LOLO_MODELS]}"
    )


def _write_full_size_cache(cache: Path, nmf_ks: tuple[int, ...]) -> Path:
    """A cache at the design's 50 x 107 grid holding only what ``load_inputs`` reads, with
    ``nmf_{k}`` arrays for ``nmf_ks`` alone -- enough to exercise the full-size guard on the
    candidate component counts. 60 genes: past the 50 a pair needs, small enough to stay fast."""
    n_lines, n_drugs, n_genes = hf.FULL_GRID_LINES, hf.FULL_GRID_DRUGS, 60
    lines = tuple(f"L{i:03d}" for i in range(n_lines))
    drugs = tuple(f"D{j:03d}" for j in range(n_drugs))
    cache.mkdir(parents=True, exist_ok=True)
    grid_path = cache / "rung1_grid.json"
    grid_path.write_text(
        json.dumps(
            {
                "dose": 5.0,
                "lines": list(lines),
                "drugs": list(drugs),
                "metadata_name": {drug: drug for drug in drugs},
                "excluded_pairs": [],
            }
        )
        + "\n"
    )

    rng = np.random.default_rng(FIXTURE_SEED)
    delta = rng.standard_normal((n_lines, n_drugs, n_genes)).astype(np.float32)
    np.savez(
        cache / "answers.npz",
        lines=np.asarray(lines),
        drugs=np.asarray(drugs),
        genes=np.asarray([f"G{g:03d}" for g in range(n_genes)]),
        delta=delta,
        responding=np.isfinite(delta),
    )

    widest = max(hf.COMPONENT_KS)
    descriptions: dict[str, np.ndarray] = {
        "lines": np.asarray(lines),
        "expression": standardize(rng.standard_normal((n_lines, EXPRESSION_WIDTH))),
        "pca": standardize(rng.standard_normal((n_lines, widest))),
        **{f"nmf_{k}": standardize(rng.random((n_lines, k))) for k in nmf_ks},
        "random_expression": random_stand_in(n_lines, EXPRESSION_WIDTH, RANDOM_SEEDS["expression"]),
        "random_pca": random_stand_in(n_lines, widest, RANDOM_SEEDS["pca"]),
        "random_nmf": random_stand_in(n_lines, max(nmf_ks), RANDOM_SEEDS["nmf"]),
    }
    np.savez(cache / "descriptions.npz", **cast(dict[str, Any], descriptions))

    fingerprints = rng.random((n_drugs, N_BITS)) < 0.3
    fingerprints[:, 0] = True
    np.savez(
        cache / "tanimoto.npz",
        drugs=np.asarray(drugs),
        similarity=tanimoto(fingerprints),
        fingerprints=fingerprints,
    )

    columns = [f"dim_{i}" for i in range(STACK_WIDTH)]
    for version in ("base", "cytokine", "drug"):
        pd.DataFrame(
            rng.standard_normal((n_lines, STACK_WIDTH)),
            index=pd.Index(lines, name="line"),
            columns=columns,
        ).to_parquet(cache / f"embedding_{version}.parquet")

    return grid_path


@pytest.mark.step_fit
def test_the_designs_candidate_counts_are_required_at_full_size(tmp_path: Path) -> None:
    """On the design's 50 x 107 grid every tuned model must offer the declared candidate
    component counts, stand-ins included; a cache missing one ``nmf_{k}`` array is refused rather
    than quietly running a model with fewer candidates than the design declares. The fixture's
    6-line grid is not full size, so it keeps the leniency it needs."""
    complete = tmp_path / "complete"
    inputs = hf.load_inputs(load_grid(_write_full_size_cache(complete, hf.COMPONENT_KS)), complete)
    assert hf.realized_component_ks(inputs) == {
        model: list(hf.COMPONENT_KS) for model in ("nmf", "pca", "random_nmf", "random_pca")
    }

    narrowed = tmp_path / "narrowed"
    grid_path = _write_full_size_cache(narrowed, tuple(k for k in hf.COMPONENT_KS if k != 15))
    with pytest.raises(ValueError, match="candidate component counts"):
        hf.load_inputs(load_grid(grid_path), narrowed)


@pytest.mark.step_null
def test_a_blocks_record_pins_the_seed_its_draws_came_from(
    run_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """A block's completion record holds the seed its draws were actually made with, not one
    recomputed from the module's base seed: a block drawn from another base would otherwise be
    recorded with provenance it does not have."""
    grid = load_grid(run_dirs["grid"])
    pair_scores = hr.load_pair_scores(run_dirs["staged"], grid)
    other_base = hr.REDRAW_BASE_SEED + 500
    block_seed = redraw_block_seed(other_base, 0)
    draws = hr.run_redraw_block(pair_scores, 0, seed=other_base, grid=grid)
    hr.write_block(tmp_path, 0, draws, block_seed)

    record = json.loads((tmp_path / "redraws_0.parquet.done.json").read_text())
    assert block_seed != redraw_block_seed(hr.REDRAW_BASE_SEED, 0)
    assert record["block_seed"] == block_seed
