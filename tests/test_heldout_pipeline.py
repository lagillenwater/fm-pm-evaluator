"""Rung 1's run: rounds, redraw blocks, and the combine that gathers them (tasks 10a and 10b).

The three runners are ``scripts/heldout_fit.py`` (a round: every model declared for the scheme,
fitted, scored on both gene sets, with the settings its tuning chose),
``scripts/heldout_redraws.py`` (a block of redraws of every comparison) and
``scripts/heldout_combine.py`` (the run's result tables, control tables, figures, leakage
records and parameter sidecar). All three are loaded by path, as ``scripts`` is not a package,
and exercised on a 6-line x 8-drug fixture cache built here -- the real cache's files, at a size
that fits in a test.

What is checked:

* **Invariant 8, staged equals one process.** 6 leave-one-line-out rounds + 8 leave-one-drug-out
  rounds + 8 redraw blocks + the combine, each run through its command line one at a time,
  produce files byte-identical to a single process that loads the inputs once and runs all of
  them (``test_staged_run_equals_one_process``).
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
* **The combine gathers, it does not invent.** Every table of the data contracts is written with
  the columns it declares; each model's fraction of the ceiling recomputes from the pair scores
  and ``rung1_ceiling.csv``; Holm is adjusted within each test and left off everything the design
  reports unadjusted; the five control tables carry the claims task 9's known-answer tests
  assert; the leakage records validate against ``LeakageProfile`` and name the removed pairs; the
  parameter sidecar pins every input's sha256, every seed, and the realized component counts.
* **A contrast between two models that are secretly one fit is refused**, not reported as a
  clean null: identical per-pair scores raise rather than writing an exact zero.

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
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from fmharness.heldout import COMPARISONS, MODELS
from fmharness.heldout.chemistry import tanimoto
from fmharness.heldout.comparisons import (
    DRAWS_PER_BLOCK,
    N_BLOCKS,
    N_DRAWS,
    holm,
    redraw_block_seed,
    redraw_estimates,
)
from fmharness.heldout.controls import synthetic_lolo_grid
from fmharness.heldout.descriptions import (
    NMF_SEED,
    RANDOM_SEEDS,
    pca_components,
    random_stand_in,
    standardize,
)
from fmharness.heldout.figures import null_panel_comparisons
from fmharness.heldout.grid import (
    SCIPLEX_CIDS,
    SCIPLEX_LINE,
    Grid,
    load_drug_metadata,
    load_grid,
)
from fmharness.heldout.leakage import LeakageProfile, leakage_profiles
from fmharness.heldout.records import is_done, sha256_file
from fmharness.heldout.scoring import GENE_SETS, fraction_of_ceiling

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
hc = _load_module("heldout_combine", "scripts/heldout_combine.py")

#: The fixture grid: small enough to fit 14 rounds in a test, wide enough that every pair clears
#: the 50-gene minimum on both gene sets.
N_LINES, N_DRUGS, N_GENES = 6, 8, 120
N_RESPONDING = 80
EXPRESSION_WIDTH, STACK_WIDTH, N_BITS = 40, 12, 64
NMF_KS: tuple[int, ...] = (2, 5)
#: Every declared block runs here, not a subset: the combine gathers the whole 2,000 redraws and
#: refuses a partial set, so a fixture that ran four blocks would exercise nothing downstream.
N_BLOCKS_TESTED = N_BLOCKS

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

#: Design section 6's ceilings, the values the fixture's ``rung1_ceiling.csv`` carries: the
#: combine divides every model's mean score by these, so the fixture uses the real numbers.
CEILING_SQRT_SB = {"responding": 0.8575, "all": 0.3876}

#: A fraction of the ceiling, or a comparison estimate, recomputed here from the written tables
#: may differ from the combine's own arithmetic only by floating-point association.
TABLE_TOLERANCE = 1e-12

#: The descriptions the build control covers: the three built from expression, and the three
#: Stack versions, whose identity-match rows the embed stage writes in the same shape.
BUILD_DESCRIPTIONS = ("expression", "pca", "nmf", "stack_base", "stack_cytokine", "stack_drug")

#: The fixture's build-stage numbers: a description matches its own other half far above the
#: shuffled null (task 5's control), and the drug fine-tune's encoder differs from the base's.
FIXTURE_IDENTITY_SHARE, FIXTURE_NULL_MEAN, FIXTURE_NULL_P99 = 0.8333, 0.1667, 0.4


# ==============================================================================================
# The fixture cache: exactly the files a round reads, at 6 lines x 8 drugs.


def write_fixture(cache: Path) -> Path:
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

    _write_drug_metadata(cache)
    return grid_path


def _write_drug_metadata(cache: Path) -> Path:
    """The Tahoe drug table for the fixture's drugs, in the real table's three columns.

    The names are the grid's screen names stripped, as the real table's are, so the combine's
    leakage records go through the same ``str.strip()`` crosswalk the grid was built with. None
    of these PubChem ids is one of the five the sci-Plex fine-tune saw.
    """
    path = cache / "drug_metadata.csv"
    pd.DataFrame(
        {
            "drug": [drug.strip() for drug in DRUGS],
            "pubchem_cid": [str(90000 + j) for j in range(N_DRUGS)],
            "canonical_smiles": ["CCO"] * N_DRUGS,
        }
    ).to_csv(path, index=False)
    return path


def write_build_tables(out_dir: Path) -> None:
    """The tables earlier stages leave in ``--out-dir``, which the combine's build control and
    build figure read: the ceiling, the cells per line and plate, each description's
    half-versus-half identity match with its shuffled null, the correlation grids behind them,
    and the drug fine-tune's weights check.

    The ceiling holds design section 6's real values, because the combine divides real scores by
    it; the rest are the fixture's own small stand-ins for what the cluster stages write.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "gene_set": gene_set,
                "pairs_scored_rung0": pairs,
                "split_half_r": split_half_r,
                "sb": sb,
                "sqrt_sb": CEILING_SQRT_SB[gene_set],
                "promoted_r": promoted_r,
                "promoted_sb": promoted_sb,
                "promoted_sqrt_sb": promoted_sqrt_sb,
            }
            for gene_set, pairs, split_half_r, sb, promoted_r, promoted_sb, promoted_sqrt_sb in (
                ("responding", 4593, 0.5815, 0.7353, 0.5768, 0.7316, 0.8553),
                ("all", 5350, 0.0812, 0.1503, 0.0808, 0.1494, 0.3865),
            )
        ]
    ).to_csv(out_dir / "rung1_ceiling.csv", index=False)

    rng = np.random.default_rng(FIXTURE_SEED)
    pd.DataFrame(
        {
            "line": np.repeat(np.asarray(LINES), 2),
            "plate": np.tile(np.asarray(["plate_0", "plate_1"]), N_LINES),
            "n_dmso_seen": rng.integers(600, 1200, 2 * N_LINES),
            "n_superset": rng.integers(200, 400, 2 * N_LINES),
            "n_selected": rng.integers(100, 300, 2 * N_LINES),
        }
    ).to_csv(out_dir / "rung1_cells.csv", index=False)

    def match_row(description: str) -> dict[str, object]:
        return {
            "description": description,
            "identity_share": FIXTURE_IDENTITY_SHARE,
            "null_mean": FIXTURE_NULL_MEAN,
            "null_p99": FIXTURE_NULL_P99,
            "n_lines": N_LINES,
            "n_shuffles": 200,
        }

    pd.DataFrame([match_row(name) for name in ("expression", "pca", "nmf")]).to_csv(
        out_dir / "rung1_identity_match.csv", index=False
    )
    for version in ("base", "cytokine", "drug"):
        pd.DataFrame([match_row(f"stack_{version}")]).to_csv(
            out_dir / f"rung1_identity_match_stack_{version}.csv", index=False
        )

    lines = np.asarray(LINES)
    for description in BUILD_DESCRIPTIONS:
        real = 0.8 * np.eye(N_LINES) + 0.2 * rng.standard_normal((N_LINES, N_LINES))
        shuffled = 0.2 * rng.standard_normal((N_LINES, N_LINES))
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "line_a": np.repeat(lines, N_LINES),
                        "line_b": np.tile(lines, N_LINES),
                        "r": grid_values.ravel(),
                        "source": source,
                    }
                )
                for grid_values, source in ((real, "real"), (shuffled, "shuffled"))
            ],
            ignore_index=True,
        ).to_csv(out_dir / f"rung1_identity_grid_{description}.csv", index=False)

    (out_dir / "rung1_weights_check.json").write_text(
        json.dumps(
            {
                "drug_checkpoint": "epoch=5-val_loss=6.1078.ckpt",
                "base_checkpoint": "bc_large.ckpt",
                "head_suffixes": [".decoder.weight"],
                "n_shared": 10,
                "n_identical": 2,
                "n_different": 8,
                "identical_all": False,
                "only_in_a": [],
                "only_in_b": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# ==============================================================================================
# Running the two stages: one round (or block) per command line, or all of them in one process.


def _run_cli(mp: pytest.MonkeyPatch, module: ModuleType, args: list[str]) -> None:
    mp.setattr(sys, "argv", ["x", *args])
    module.main()


def stage_by_stage(cache: Path, grid_path: Path, out_dir: Path) -> None:
    """Every round, redraw block and finally the combine, through its own command line."""
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
        _run_cli(
            mp,
            hc,
            [
                "--grid",
                str(grid_path),
                "--cache",
                str(cache),
                "--out-dir",
                str(out_dir),
                "--drug-metadata",
                str(cache / "drug_metadata.csv"),
            ],
        )


def _in_one_process(cache: Path, grid_path: Path, out_dir: Path) -> None:
    """The same rounds, blocks and combine in a single process, loading the inputs once."""
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
    hc.combine(
        out_dir,
        cache,
        grid=grid,
        drug_metadata=load_drug_metadata(cache / "drug_metadata.csv"),
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
    """One fixture cache run twice: stage by stage, and in a single process.

    Each run writes its result tables into its own ``--out-dir``, seeded with the tables the
    earlier cluster stages leave there (the ceiling, the cells, the identity matches and their
    grids, the weights check) -- the combine reads those and must not invent them.
    """
    root = tmp_path_factory.mktemp("rung1_pipeline")
    staged, single = root / "staged", root / "single"
    out_staged, out_single = root / "out_staged", root / "out_single"
    grid_path = write_fixture(staged)
    shutil.copytree(staged, single)
    for out_dir in (out_staged, out_single):
        write_build_tables(out_dir)
    stage_by_stage(staged, grid_path, out_staged)
    _in_one_process(single, single / "rung1_grid.json", out_single)
    return {
        "staged": staged,
        "single": single,
        "grid": grid_path,
        "out_staged": out_staged,
        "out_single": out_single,
    }


# ==============================================================================================
# Invariant 8: staged equals one process


@pytest.mark.step_fit
def test_staged_run_equals_one_process(run_dirs: dict[str, Path]) -> None:
    """6 + 8 rounds, 8 redraw blocks and the combine, one command line each, write the same
    bytes as one process that runs all of them -- every table and every completion record.

    The combine's own outputs are compared too: the result tables, the control tables, the
    leakage records and the figures byte for byte, and the parameter sidecar with its arguments
    dropped, since the two runs write the same run from different directories.
    """
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

    out_staged, out_single = run_dirs["out_staged"], run_dirs["out_single"]
    combine_outputs = _relative_outputs(out_staged)
    assert combine_outputs == _relative_outputs(out_single)
    assert len(combine_outputs) > len(hc.FIGURES)
    for name in combine_outputs:
        staged_path, single_path = out_staged / name, out_single / name
        assert staged_path.exists(), f"the staged combine did not write {name}"
        assert single_path.exists(), f"the one-process combine did not write {name}"
        if name == hc.PARAMS_JSON:
            assert _params_without_args(staged_path) == _params_without_args(single_path)
            continue
        assert sha256_file(staged_path) == sha256_file(single_path), f"{name} differs"


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


# ==============================================================================================
# The combine: result tables, control tables, figures, leakage records and the provenance record


def _relative_outputs(out_dir: Path) -> list[str]:
    """Every file the combine writes, named relative to its output directory."""
    return [str(Path(path).relative_to(out_dir)) for path in hc.output_paths(out_dir)]


def _params_without_args(path: Path) -> dict[str, Any]:
    """The parameter sidecar with its arguments dropped: the two runs write the same run from
    different directories, so every recorded path differs and nothing else may."""
    params = cast(dict[str, Any], json.loads(path.read_text()))
    return {key: value for key, value in params.items() if key != "args"}


def _read_pair_scores(out_dir: Path) -> pd.DataFrame:
    """The gathered per-pair scores, read so that every float is the one the run wrote.

    ``read_csv``'s default float parser is not correctly rounded and can move the last bit of a
    double, which would make a score recomputed here differ from the run's own in the 16th digit
    for no reason but the reader. Task 12's battery reads this table the same way.
    """
    return pd.read_csv(out_dir / "rung1_pair_scores.csv.gz", float_precision="round_trip")


def _only(rows: pd.DataFrame, column: str) -> float:
    """One column's single value, as a float: the tables are read back from CSV, where every
    number is a string until something asks for it."""
    return float(rows[column].to_numpy(dtype=np.float64)[0])


def _strings(table: pd.DataFrame, column: str) -> list[str]:
    """One column as plain strings, row for row."""
    return [str(value) for value in table[column].tolist()]


def _wide_by_scheme_and_gene_set(scores: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """The scored population the combine reports on, rebuilt here from the written pair scores:
    one model per column, only the pairs every model scored."""
    prepared: dict[tuple[str, str], pd.DataFrame] = {}
    for scheme in ("lolo", "lodo"):
        for gene_set in GENE_SETS:
            chosen = (scores["scheme"] == scheme) & (scores["gene_set"] == gene_set)
            wide = scores.loc[chosen].pivot(index=["line", "drug"], columns="model", values="r")
            prepared[(scheme, gene_set)] = wide.dropna(axis=0, how="any")
    return prepared


@pytest.mark.step_score
def test_combine_writes_every_table_of_the_data_contracts(run_dirs: dict[str, Path]) -> None:
    """Every file the contracts name is written, with its completion record and its declared
    columns: the four result tables, the five control tables, the leakage records, the parameter
    sidecar, and one figure per design section 8 bullet."""
    out = run_dirs["out_staged"]
    assert _relative_outputs(out) == [
        "rung1_pair_scores.csv.gz",
        "rung1_model_summary.csv",
        "rung1_comparisons.csv",
        "rung1_settings.csv",
        "rung1_control_build.csv",
        "rung1_control_split.csv",
        "rung1_control_fit.csv",
        "rung1_control_score.csv",
        "rung1_control_null.csv",
        "rung1_leakage_profiles.json",
        "rung1_run.params.json",
        *(str(Path("figures") / name) for name in hc.FIGURES),
    ]
    for path in hc.output_paths(out):
        assert Path(path).exists(), f"the combine did not write {path}"
        assert is_done(Path(path)), f"{Path(path).name} has no valid completion record"

    assert list(pd.read_csv(out / "rung1_pair_scores.csv.gz").columns) == list(
        hf.SCORE_FILE_COLUMNS
    )
    assert list(pd.read_csv(out / "rung1_settings.csv").columns) == [
        "scheme",
        "round",
        "model",
        "lambda",
        "k",
        "loss_min",
        "lambda_at_edge",
    ]
    assert list(pd.read_csv(out / "rung1_model_summary.csv").columns) == list(
        hc.MODEL_SUMMARY_COLUMNS
    )
    assert list(pd.read_csv(out / "rung1_comparisons.csv").columns) == list(hc.COMPARISON_COLUMNS)


@pytest.mark.step_score
def test_the_gathered_tables_hold_every_round_and_every_model(run_dirs: dict[str, Path]) -> None:
    """The gathered pair scores and settings are the rounds' own files, concatenated: nothing
    dropped, nothing duplicated, and each round's scheme and index carried through."""
    out, staged = run_dirs["out_staged"], run_dirs["staged"]
    gathered = _read_pair_scores(out)
    pd.testing.assert_frame_equal(
        gathered.sort_values(list(hf.SCORE_FILE_COLUMNS)).reset_index(drop=True),
        _all_scores(staged).sort_values(list(hf.SCORE_FILE_COLUMNS)).reset_index(drop=True),
        check_exact=True,
    )

    settings = pd.read_csv(out / "rung1_settings.csv", float_precision="round_trip")
    assert len(settings) == len(_all_settings(staged))
    for scheme, n_rounds in (("lolo", N_LINES), ("lodo", N_DRUGS)):
        declared = [spec.id for spec in MODELS.values() if scheme in spec.schemes]
        rows = settings.loc[settings["scheme"] == scheme]
        assert sorted(rows["round"].unique().tolist()) == list(range(n_rounds))
        assert set(rows["model"]) == set(declared)


@pytest.mark.step_score
def test_model_summary_means_and_fractions_recompute_from_the_pair_scores(
    run_dirs: dict[str, Path],
) -> None:
    """Each model's mean score is the mean over the pairs every model scored, and its fraction of
    the ceiling is that mean over ``rung1_ceiling.csv``'s square root of SB -- both recomputed
    here from the written tables alone."""
    out = run_dirs["out_staged"]
    summary = pd.read_csv(out / "rung1_model_summary.csv")
    prepared = _wide_by_scheme_and_gene_set(_read_pair_scores(out))
    assert len(summary) == sum(len(wide.columns) for wide in prepared.values())

    for (scheme, gene_set), wide in prepared.items():
        for model in wide.columns:
            rows = summary.loc[
                (summary["scheme"] == scheme)
                & (summary["gene_set"] == gene_set)
                & (summary["model"] == model)
            ]
            assert len(rows) == 1, f"{scheme}/{gene_set}/{model} has {len(rows)} summary rows"
            mean_r = float(wide[model].to_numpy(dtype=np.float64).mean())
            assert int(_only(rows, "n_pairs")) == len(wide)
            assert abs(_only(rows, "mean_r") - mean_r) <= TABLE_TOLERANCE
            assert _only(rows, "sqrt_sb") == CEILING_SQRT_SB[gene_set]
            expected = fraction_of_ceiling(mean_r, CEILING_SQRT_SB[gene_set])
            assert abs(_only(rows, "fraction_of_ceiling") - expected) <= TABLE_TOLERANCE
            assert _only(rows, "ci_lo") <= mean_r <= _only(rows, "ci_hi")


@pytest.mark.step_null
def test_comparisons_cover_every_contrast_with_holm_within_each_test(
    run_dirs: dict[str, Path],
) -> None:
    """Every declared comparison and every description-versus-stand-in contrast is reported on
    both gene sets; each estimate is the mean paired difference over the shared pairs; and Holm
    is applied within each test on responding genes and nowhere else (design section 7)."""
    out = run_dirs["out_staged"]
    table = pd.read_csv(out / "rung1_comparisons.csv")
    declared = {comparison.id for comparison in COMPARISONS}
    stand_in = {contrast.id for contrast in hr.stand_in_contrasts()}
    assert set(table["comparison"]) == declared | stand_in
    assert len(table) == (len(declared) + len(stand_in)) * len(GENE_SETS)

    prepared = _wide_by_scheme_and_gene_set(_read_pair_scores(out))
    ids = _strings(table, "comparison")
    schemes = _strings(table, "scheme")
    gene_sets = _strings(table, "gene_set")
    model_a = _strings(table, "model_a")
    model_b = _strings(table, "model_b")
    estimate = table["estimate"].to_numpy(dtype=np.float64)
    n_pairs = table["n_pairs"].to_numpy(dtype=np.int64)

    assert (
        table["n_draws"].to_numpy(dtype=np.int64) + table["n_dropped"].to_numpy(dtype=np.int64)
        == N_DRAWS
    ).all()
    assert (table["ci_lo"].to_numpy(dtype=np.float64) <= table["ci_hi"].to_numpy(np.float64)).all()
    assert (table["mde"].to_numpy(dtype=np.float64) > 0.0).all()

    for index in range(len(table)):
        wide = prepared[(schemes[index], gene_sets[index])]
        difference = wide[model_a[index]].to_numpy(dtype=np.float64) - wide[
            model_b[index]
        ].to_numpy(dtype=np.float64)
        expected = float(difference.mean())
        assert abs(float(estimate[index]) - expected) <= TABLE_TOLERANCE, ids[index]
        assert int(n_pairs[index]) == len(wide)

    adjusted = table.loc[table["p_holm"].notna()]
    assert set(adjusted["gene_set"]) == {"responding"}
    assert set(adjusted["comparison"]) == declared
    assert {str(k): int(v) for k, v in adjusted["holm_family"].value_counts().items()} == {
        "lolo": 6,
        "lodo": 5,
    }
    for _, family in adjusted.groupby("holm_family"):
        ordered = family.sort_values("comparison")
        np.testing.assert_allclose(
            holm(ordered["p"].to_numpy(dtype=np.float64)),
            ordered["p_holm"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=TABLE_TOLERANCE,
        )


@pytest.mark.step_null
def test_every_comparison_reports_its_two_way_design_effect(run_dirs: dict[str, Path]) -> None:
    """Pairs share lines and drugs, so each comparison also carries the variance ratio of the
    two-way redraws over the one-way redraws, and its square root -- the interval-width ratio."""
    table = pd.read_csv(run_dirs["out_staged"] / "rung1_comparisons.csv")
    effect = table["design_effect"].to_numpy(dtype=np.float64)
    width = table["width_ratio"].to_numpy(dtype=np.float64)
    assert np.isfinite(effect).all() and (effect > 0).all()
    np.testing.assert_allclose(width, np.sqrt(effect), rtol=0.0, atol=TABLE_TOLERANCE)


@pytest.mark.known_answer
@pytest.mark.step_fit
def test_the_five_control_tables_hold_their_known_answers(run_dirs: dict[str, Path]) -> None:
    """The controls the combine records are the ones task 9's tests assert on, at the design's
    sizes and seeds: the build match beats its shuffled null and the fine-tune's weights differ;
    a broken split finds the planted signature and the shipped split does not; the planted fit is
    recovered by the matching description and not by its stand-in; synthetic answers of known
    reliability score the square root of R; and the redraws detect a plant at its MDE about 80%
    of the time and almost nothing when nothing is planted."""
    out = run_dirs["out_staged"]

    build = pd.read_csv(out / "rung1_control_build.csv")
    identity = build.loc[build["check"] == "half_vs_half_identity"]
    assert set(identity["subject"]) == set(BUILD_DESCRIPTIONS)
    assert bool(identity["above_null_p99"].all()), "a description did not beat its shuffled null"
    weights = build.loc[build["check"] == "drug_finetune_weights_differ"]
    assert len(weights) == 1
    assert not bool(weights["encoders_identical"].iloc[0])
    assert int(weights["n_different_tensors"].iloc[0]) == 8

    split = pd.read_csv(out / "rung1_control_split.csv")
    assert set(split["scheme"]) == {"lolo", "lodo"}
    assert set(split["split"]) == {"leaky", "shipped"}
    leaky = split.loc[split["split"] == "leaky"]
    assert bool(leaky["detected"].all()), "the broken split did not recover the planted signature"
    shipped = split.loc[split["split"] == "shipped"]
    assert bool(shipped["within_mde"].all()), "the shipped split leaked the planted signature"

    fit = pd.read_csv(out / "rung1_control_fit.csv")
    planted = fit.loc[fit["planted"]]
    oracle = planted.loc[planted["contrast"] == "oracle"]
    assert set(oracle["scheme"]) == {"lolo", "lodo"}
    assert (oracle["ratio_to_mde"] >= 2.0).all(), "the plant is not twice the oracle's MDE"
    description = planted.loc[planted["contrast"] == "description"]
    assert (description["estimate"] > 0).all() and (description["p"] < 0.05).all()
    lolo_random = planted.loc[(planted["scheme"] == "lolo") & (planted["contrast"] == "random")]
    assert (lolo_random["estimate"] <= lolo_random["mde"]).all(), "a random stand-in gained"
    beats_random = planted.loc[
        (planted["scheme"] == "lodo") & (planted["contrast"] == "description_minus_random")
    ]
    assert (beats_random["estimate"] > 0).all() and (beats_random["p"] < 0.05).all()
    # Task 9's note: with a drug hidden the stand-in's own gain is reported, not required to be 0.
    assert (
        planted.loc[(planted["scheme"] == "lodo") & (planted["contrast"] == "random"), "estimate"]
        .notna()
        .all()
    )
    nothing_planted = fit.loc[~fit["planted"]]
    assert len(nothing_planted) > 0
    assert (nothing_planted["estimate"] <= nothing_planted["mde"]).all()
    at_top = nothing_planted["lambda_at_top_share"].dropna()
    assert len(at_top) > 0 and (at_top >= 0.5).all()

    score = pd.read_csv(out / "rung1_control_score.csv")
    assert set(score["prediction"]) == {"truth", "second_measurement", "unrelated"}
    assert set(score["gene_set"]) == set(GENE_SETS)
    assert bool(score["within_3_se"].all()), "a synthetic answer did not score its planted value"
    truth = score.loc[(score["prediction"] == "truth") & (score["reliability"] == 0.7353)]
    assert np.allclose(truth["planted"].to_numpy(dtype=np.float64), np.sqrt(0.7353))

    null = pd.read_csv(out / "rung1_control_null.csv")
    assert set(null["check"]) == {"detection_at_mde", "false_positive_rate"}
    assert (null["repetitions"] == 200).all()
    assert bool(null["inside_interval"].all()), "the redraws' detection rates left their interval"


@pytest.mark.step_score
def test_the_combine_writes_one_non_empty_png_per_design_figure(
    run_dirs: dict[str, Path],
) -> None:
    """One figure per design section 8 bullet -- build, split, fit, score, null -- each a real
    PNG rather than a stub or a truncated write."""
    out = run_dirs["out_staged"]
    assert len(hc.FIGURES) == 5
    for name in hc.FIGURES:
        blob = (out / "figures" / name).read_bytes()
        assert blob[:4] == b"\x89PNG", f"{name} is not a PNG"
        assert len(blob) > 1000, f"{name} is only {len(blob)} bytes, too small to hold panels"


@pytest.mark.step_document
def test_the_leakage_records_validate_and_say_what_was_removed(
    run_dirs: dict[str, Path],
) -> None:
    """One record per Stack version, each validating against ``LeakageProfile``: no version saw
    Tahoe's treated cells, the drug fine-tune names the line and the five PubChem ids it saw, and
    a grid without that line says so rather than reporting a bare zero."""
    records = json.loads((run_dirs["out_staged"] / "rung1_leakage_profiles.json").read_text())
    profiles = [LeakageProfile.model_validate(record) for record in records]
    assert [profile.model_version for profile in profiles] == [
        "stack_base",
        "stack_cytokine",
        "stack_drug",
    ]
    assert not any(profile.saw_tahoe_treated_cells for profile in profiles)
    assert all(profile.untreated_cells_may_be_in_pretraining for profile in profiles)
    assert all(profile.excluded_pairs_removed_for_every_model for profile in profiles)

    fine_tuned = [profile for profile in profiles if profile.sciplex_fine_tuned]
    assert [profile.model_version for profile in fine_tuned] == ["stack_drug"]
    drug = fine_tuned[0]
    assert drug.exposed_line == SCIPLEX_LINE
    assert tuple(drug.exposed_cids) == SCIPLEX_CIDS
    # The fixture's six lines are not A549, and the record distinguishes "the line is not in this
    # grid" from "the line is here and nothing was found" -- a bare 0 reads the same either way.
    assert drug.sciplex_line_in_grid is False
    assert drug.n_exposed_pairs == 0
    assert drug.exposed_pairs == ()


@pytest.mark.known_answer
@pytest.mark.step_document
def test_leakage_profiles_name_the_two_removed_pairs() -> None:
    """On a grid that does hold A549, the record names exactly the two pairs design section 7
    removes, matched to the real drug table by PubChem id through the grid's own name crosswalk.

    And a grid whose excluded pairs disagree with the drug table raises: a leakage record that
    quietly reported no exposure would look exactly like a model that had none.
    """
    drug_metadata = load_drug_metadata(REPO / "tests" / "fixtures" / "tahoe_drug_metadata.csv")
    pairs = ((SCIPLEX_LINE, "Temsirolimus"), (SCIPLEX_LINE, "Trametinib "))
    grid = Grid(
        lines=("ACH-000001", SCIPLEX_LINE),
        drugs=("Temsirolimus", "Trametinib "),
        metadata_name={"Temsirolimus": "Temsirolimus", "Trametinib ": "Trametinib"},
        excluded_pairs=pairs,
    )
    profiles = leakage_profiles(grid, drug_metadata)
    drug = next(profile for profile in profiles if profile.model_version == "stack_drug")
    assert drug.sciplex_line_in_grid is True
    assert drug.exposed_pairs == pairs
    assert drug.n_exposed_pairs == 2

    with pytest.raises(ValueError, match="excluded"):
        leakage_profiles(replace(grid, excluded_pairs=()), drug_metadata)


@pytest.mark.step_promote
def test_the_params_sidecar_pins_inputs_seeds_and_component_counts(
    run_dirs: dict[str, Path],
) -> None:
    """The provenance record of project rule 1: the commit the run was made at, its arguments,
    every seed the run used, the realized component counts (ruling 36), and the sha256 of every
    input file -- each checked here against the file on disk."""
    out, cache = run_dirs["out_staged"], run_dirs["staged"]
    params = json.loads((out / hc.PARAMS_JSON).read_text())
    # A full commit hash, not merely something truthy: the producing commit is one of the three
    # things project rule 1 says cannot be recovered afterwards, and a sidecar recording
    # "unknown" would pass a truthiness check while naming no commit at all.
    assert re.fullmatch(r"[0-9a-f]{40}", str(params["git_sha"])), params["git_sha"]
    assert params["args"]["cache"] == str(cache)
    assert params["seeds"]["redraw_base_seed"] == hr.REDRAW_BASE_SEED == 7000
    assert params["seeds"]["random_stand_ins"] == dict(RANDOM_SEEDS)
    assert params["seeds"]["nmf"] == NMF_SEED
    assert params["n_draws"] == N_DRAWS
    assert params["n_blocks"] == N_BLOCKS
    assert params["ceiling"] == CEILING_SQRT_SB
    assert params["component_ks"] == {
        model: list(NMF_KS) for model in ("nmf", "pca", "random_nmf", "random_pca")
    }

    roots = {"cache": cache, "out_dir": out}
    assert params["inputs"], "the sidecar pinned no inputs"
    for name, digest in params["inputs"].items():
        root, relative = str(name).split("/", 1)
        assert sha256_file(roots[root] / relative) == digest, f"{name} does not match its sha256"
    for expected in (
        "cache/answers.npz",
        "cache/scores_lolo_000.parquet",
        "cache/settings_lodo_007.csv",
        f"cache/redraws_{N_BLOCKS - 1}.parquet",
        "out_dir/rung1_ceiling.csv",
        "out_dir/rung1_identity_match.csv",
        "out_dir/rung1_weights_check.json",
    ):
        assert expected in params["inputs"], f"{expected} is not pinned in the sidecar"


@pytest.mark.step_score
def test_a_finished_combine_is_not_recomputed(run_dirs: dict[str, Path]) -> None:
    """A combine whose outputs are all done skips: the gather is never called, and no output or
    completion record is rewritten."""
    out, cache, grid_path = run_dirs["out_staged"], run_dirs["staged"], run_dirs["grid"]
    paths = [Path(path) for path in hc.output_paths(out)]
    before = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in paths}

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a finished combine must not be recomputed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hc, "gather_pair_scores", refuse)
        _run_cli(
            mp,
            hc,
            [
                "--grid",
                str(grid_path),
                "--cache",
                str(cache),
                "--out-dir",
                str(out),
                "--drug-metadata",
                str(cache / "drug_metadata.csv"),
            ],
        )

    for path in paths:
        assert (sha256_file(path), path.stat().st_mtime_ns) == before[path], f"{path} was rewritten"


@pytest.mark.step_null
def test_a_contrast_between_two_identical_fits_is_refused() -> None:
    """Two models that scored every pair identically are one fit under two names, and their
    contrast is an exact zero with no spread -- indistinguishable, in the reported table, from a
    correct null. The combine raises instead of writing it."""
    scored = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="identical"):
        hc.paired_differences(scored, "a", "b", "lolo_a_vs_b")

    moved = scored.assign(b=[0.1, 0.2, 0.4])
    np.testing.assert_allclose(
        hc.paired_differences(moved, "a", "b", "lolo_a_vs_b"), [0.0, 0.0, -0.1]
    )


@pytest.mark.step_null
def test_the_null_figure_draws_one_panel_per_hypothesis(run_dirs: dict[str, Path]) -> None:
    """Design section 8 asks the null figure for "redraws for H1(a) and H2", one panel each.

    The trap this pins: H1(a) names a comparison under BOTH schemes -- Stack base against the
    drug average when a line is hidden, against chemistry only when a drug is hidden -- so a
    selection that takes "the first two responding-gene comparisons" fills both panels with
    H1(a) and never reaches H2, one of the rung's two hypotheses. The panels are chosen per
    hypothesis instead, and the titles the figure draws are these names.
    """
    assert sum(1 for comparison in COMPARISONS if comparison.hypothesis == "H1(a)") == 2

    table = pd.read_csv(run_dirs["out_staged"] / "rung1_comparisons.csv")
    drawn = null_panel_comparisons(table)
    assert len(drawn) == 2, drawn
    hypotheses = [
        str(table.loc[table["comparison"] == name, "hypothesis"].to_numpy()[0]) for name in drawn
    ]
    assert hypotheses == ["H1(a)", "H2"], dict(zip(drawn, hypotheses, strict=True))


@pytest.mark.step_build
def test_a_weights_check_missing_a_key_is_refused(
    run_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """The build control's weights row is H2's premise -- that the drug fine-tune is a different
    model from the base checkpoint. Read with ``dict.get`` defaults, an empty or renamed
    ``rung1_weights_check.json`` would publish "encoders differ, 0 tensors differ": that claim
    manufactured from silence. Every key is required instead."""
    out_dir = tmp_path / "out"
    shutil.copytree(run_dirs["out_staged"], out_dir)
    complete = json.loads((out_dir / "rung1_weights_check.json").read_text())
    assert all(key in complete for key in hc.WEIGHTS_CHECK_KEYS)

    for missing in hc.WEIGHTS_CHECK_KEYS:
        partial = {key: value for key, value in complete.items() if key != missing}
        (out_dir / "rung1_weights_check.json").write_text(json.dumps(partial) + "\n")
        with pytest.raises(SystemExit, match=missing):
            hc.read_build_tables(out_dir)

    (out_dir / "rung1_weights_check.json").write_text(json.dumps({}) + "\n")
    with pytest.raises(SystemExit, match="weights"):
        hc.read_build_tables(out_dir)


@pytest.mark.step_score
def test_the_combine_reports_the_population_the_redraws_were_taken_over(
    run_dirs: dict[str, Path],
) -> None:
    """The combine's scored population is the redraws' own: the pairs every model scored, per
    scheme and gene set. If the two ever drifted apart, every comparison's estimate would be
    taken over one set of pairs and its interval over another."""
    grid = load_grid(run_dirs["grid"])
    prepared = hr._scored_pairs(hr.load_pair_scores(run_dirs["staged"], grid), grid)
    summary = pd.read_csv(run_dirs["out_staged"] / "rung1_model_summary.csv")
    comparisons = pd.read_csv(run_dirs["out_staged"] / "rung1_comparisons.csv")
    for (scheme, gene_set), scored in prepared.items():
        rows = summary.loc[(summary["scheme"] == scheme) & (summary["gene_set"] == gene_set)]
        assert set(rows["n_pairs"]) == {len(scored.scores)}
        contrasts = comparisons.loc[
            (comparisons["scheme"] == scheme) & (comparisons["gene_set"] == gene_set)
        ]
        assert set(contrasts["n_pairs"]) == {len(scored.scores)}
