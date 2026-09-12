"""Task 10b of rung 1: gather the rounds and the redraws into the run's tables and figures.

Design.md sections 6 to 9. The 157 rounds (``scripts/heldout_fit.py``) and the 8 redraw blocks
(``scripts/heldout_redraws.py``) each write their own slice of the run into the cache. This step
is what turns those slices into the run's result:

* **The gathered tables.** Every per-pair score, every round's chosen settings, each model's mean
  score with its interval and its fraction of the ceiling, and every comparison with its
  interval, p-value, minimum detectable effect, Holm adjustment and two-way design effect.
* **The control tables.** The five known-answer controls of design section 8, run here at the
  design's own sizes, seeds and ceiling values -- the same controls ``tests/test_rung1_controls.py``
  asserts on, recorded as the evidence behind those assertions rather than only as a green test.
* **The figures**, one per design section 8 bullet, each drawn from a table this step wrote and
  each shown beside its control (project rule 4).
* **The leakage records and the parameter sidecar** -- what each model version saw before the run,
  and the commit, arguments, seeds and input checksums the run was made from (project rule 1).

Two properties are load-bearing and are enforced rather than assumed.

**One scored population.** A comparison is the mean paired difference over the pairs EVERY model
scored, and a model's reported mean score is taken over that same population; the redraw blocks
took their draws over it too. The population is rebuilt here the way ``heldout_redraws.py`` builds
it -- pivot per (scheme, gene set), drop any pair a model left NaN -- so the estimate, the
interval and the mean score all describe the same pairs.

**No fabricated zero.** A contrast between two models that scored every pair identically is one
fit under two names; reported, it is an exact zero with no spread, which reads exactly like a
clean null. ``paired_differences`` raises instead. Likewise the leakage records refuse to report
"no exposure" when the crosswalk merely failed to match, and the run refuses to gather a partial
set of rounds or redraw blocks, since a comparison taken over some of them is silently a
different comparison.

    uv run python scripts/heldout_combine.py \\
        --grid docs/tasks/rung1-held-out-prediction/rung1_grid.json \\
        --cache /scratch/alpine/$USER/rung1_cache \\
        --drug-metadata path/to/drug_metadata.parquet \\
        --out-dir docs/tasks/rung1-held-out-prediction
"""

# pandas and scipy ship no PEP-561 type stubs in this environment; under strict mode that turns
# every call site into a cascade of reportUnknown* noise about *their* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmharness.heldout import COMPARISONS, Scheme  # noqa: E402
from fmharness.heldout.answers import scoreable  # noqa: E402
from fmharness.heldout.comparisons import (  # noqa: E402
    N_BLOCKS,
    N_DRAWS,
    RedrawSummary,
    design_effect,
    holm,
    mean_score_ci,
    summarize_redraws,
    width_ratio,
)

# The controls' machinery and every number it uses come from fmharness.heldout.controls, which
# tests/test_rung1_controls.py asserts on and this step publishes (ruling 38). Both callers run
# THE SAME code: a second copy that merely agreed today would let the published evidence table
# keep the old behaviour the next time the test harness changed, with nothing failing.
from fmharness.heldout.controls import (  # noqa: E402
    ALPHA,
    DETECTION_RATE,
    FIT_GRID,
    FIT_NULL_GRID_SEED,
    FIT_NULL_STAND_IN_SEED,
    FIT_SEEDS,
    NULL_UNIT_SEED,
    R_ALL,
    R_RESPONDING,
    REDRAW_SEED,
    REPETITIONS,
    SCORE_GRID,
    SCORE_POOL_SEED,
    SCORE_RESPONDING_SEED,
    SCORE_UNRELATED_SEEDS,
    SE_MULTIPLE,
    SPLIT_AMPLITUDE,
    SPLIT_GRID,
    SPLIT_SEEDS,
    STRENGTH,
    contrast_summary,
    fit_control,
    fit_null_control,
    grid_scores,
    null_repetition_pvalues,
    planted_reliability_pool,
    split_control_run,
    synthetic_answers,
)
from fmharness.heldout.descriptions import NMF_SEED, RANDOM_SEEDS  # noqa: E402
from fmharness.heldout.figures import (  # noqa: E402
    fig_build,
    fig_fit,
    fig_null,
    fig_score,
    fig_split,
)
from fmharness.heldout.grid import Grid, load_drug_metadata, load_grid  # noqa: E402
from fmharness.heldout.leakage import leakage_profiles, leakage_records  # noqa: E402
from fmharness.heldout.records import is_done, sha256_file, write_record  # noqa: E402
from fmharness.heldout.scoring import GENE_SETS, fraction_of_ceiling, score_pairs  # noqa: E402

#: The schemes, in the order every table lays them out.
SCHEMES: tuple[Scheme, ...] = ("lolo", "lodo")

#: The three Stack versions whose per-line embeddings the run read (design.md section 4).
STACK_VERSIONS: tuple[str, ...] = ("base", "cytokine", "drug")

#: The seed the model summary's intervals are redrawn with. Distinct from every other seed the
#: run uses -- the stand-ins' 101-106, NMF's 0, the redraw blocks' 7000-7007 -- so no two stages
#: share a random stream. Recorded in the parameter sidecar with the rest.
SUMMARY_SEED = 7100

#: ``rung1_pair_scores.csv.gz``'s columns, in order: the rounds' own contract, gathered.
PAIR_SCORE_COLUMNS: tuple[str, ...] = (
    "scheme",
    "round",
    "model",
    "line",
    "drug",
    "gene_set",
    "r",
    "n_genes",
)

#: ``rung1_settings.csv``'s columns: each round's settings file, with the round it came from.
SETTINGS_COLUMNS: tuple[str, ...] = (
    "scheme",
    "round",
    "model",
    "lambda",
    "k",
    "loss_min",
    "lambda_at_edge",
)

#: ``rung1_model_summary.csv``'s columns. ``scheme`` is a key: a model fitted under both schemes
#: predicts different things in each, and pooling the two would average two different questions.
MODEL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "scheme",
    "model",
    "gene_set",
    "n_pairs",
    "mean_r",
    "ci_lo",
    "ci_hi",
    "sd",
    "mde",
    "n_draws",
    "n_dropped",
    "sqrt_sb",
    "fraction_of_ceiling",
)

#: ``rung1_comparisons.csv``'s columns. ``p_holm`` is filled only where design section 7 adjusts
#: (the declared comparisons, responding genes, within each test) and empty everywhere else.
COMPARISON_COLUMNS: tuple[str, ...] = (
    "comparison",
    "scheme",
    "model_a",
    "model_b",
    "hypothesis",
    "holm_family",
    "gene_set",
    "n_pairs",
    "estimate",
    "ci_lo",
    "ci_hi",
    "p",
    "p_holm",
    "sd",
    "mde",
    "n_draws",
    "n_dropped",
    "design_effect",
    "width_ratio",
)

BUILD_CONTROL_COLUMNS: tuple[str, ...] = (
    "check",
    "subject",
    "identity_share",
    "null_mean",
    "null_p99",
    "n_lines",
    "n_shuffles",
    "above_null_p99",
    "n_shared_tensors",
    "n_identical_tensors",
    "n_different_tensors",
    "encoders_identical",
)

SPLIT_CONTROL_COLUMNS: tuple[str, ...] = (
    "scheme",
    "split",
    "n_lines",
    "n_drugs",
    "n_genes",
    "amplitude",
    "estimate",
    "ci_lo",
    "ci_hi",
    "p",
    "sd",
    "mde",
    "detected",
    "within_mde",
)

FIT_CONTROL_COLUMNS: tuple[str, ...] = (
    "scheme",
    "planted",
    "contrast",
    "estimate",
    "ci_lo",
    "ci_hi",
    "p",
    "sd",
    "mde",
    "ratio_to_mde",
    "lambda_at_top_share",
    "n_lines",
    "n_drugs",
    "n_genes",
    "width",
    "strength",
    "reliability",
)

SCORE_CONTROL_COLUMNS: tuple[str, ...] = (
    "reliability",
    "gene_set",
    "prediction",
    "planted",
    "mean_r",
    "se",
    "n_pairs",
    "within_3_se",
)

NULL_CONTROL_COLUMNS: tuple[str, ...] = (
    "check",
    "target_rate",
    "repetitions",
    "alpha",
    "detections",
    "rate",
    "ci_lo",
    "ci_hi",
    "inside_interval",
)

#: The figures, one per design section 8 bullet, in the order the design lists the steps.
FIGURES: tuple[str, ...] = (
    "01_build.png",
    "02_split.png",
    "03_fit.png",
    "04_score.png",
    "05_null.png",
)

LEAKAGE_JSON = "rung1_leakage_profiles.json"
PARAMS_JSON = "rung1_run.params.json"

#: READING THESE TABLES BACK: pandas' default CSV float parser is not correctly rounded and can
#: shift the last bit of a double, so a mean recomputed from ``rung1_pair_scores.csv.gz`` with a
#: plain ``read_csv`` differs from this run's own in the 16th digit. ``to_csv``'s default output
#: does round-trip exactly -- the fix belongs on the reader, not the writer (measured: 5,000 of
#: 5,000 floats exact with ``float_precision="round_trip"``, 3,445 of 5,000 without). Task 12's
#: battery must read this table, and ``rung1_settings.csv``, with ``float_precision="round_trip"``.

#: The keys ``rung1_weights_check.json`` must carry. The build control's weights row is H2's
#: premise -- that the drug fine-tune is a different model from the base checkpoint -- so an
#: absent key is refused rather than defaulted into a published claim.
WEIGHTS_CHECK_KEYS: tuple[str, ...] = (
    "n_shared",
    "n_identical",
    "n_different",
    "identical_all",
)


def _load_script(name: str, path: Path) -> ModuleType:
    """Import a sibling ``scripts/`` module by path, reusing one already imported.

    ``scripts`` is not a package, so the redraw runner -- whose scored-pair population, contrast
    list and base seed this step must use rather than re-derive -- is loaded the way the project's
    tests load it.
    """
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REDRAWS = _load_script("heldout_redraws", REPO / "scripts" / "heldout_redraws.py")

#: The redraws' base seed (ruling 33), recorded in the parameter sidecar with every other seed.
REDRAW_BASE_SEED: int = int(_REDRAWS.REDRAW_BASE_SEED)


@dataclass(frozen=True)
class Contrast:
    """One reported difference: ``model_a`` minus ``model_b``, on ``scheme``.

    ``hypothesis`` is the design's label for a declared comparison and empty for a
    description-versus-its-stand-in contrast, which section 7 reports unadjusted beside them.
    """

    id: str
    scheme: str
    model_a: str
    model_b: str
    hypothesis: str


@dataclass(frozen=True)
class Scored:
    """One scheme and gene set's scored pairs: one model per column, one row per pair kept."""

    scores: pd.DataFrame
    line_index: np.ndarray
    drug_index: np.ndarray


# ==============================================================================================
# Gathering what the rounds and the blocks wrote


def gather_pair_scores(cache: Path, grid: Grid) -> pd.DataFrame:
    """Every round's per-pair scores, both schemes, in round order.

    Delegates to the redraw runner's own loader, so this step gathers exactly the rounds the
    redraws were taken over and refuses, naming them, until all 157 are there.
    """
    return cast(pd.DataFrame, _REDRAWS.load_pair_scores(cache, grid))


def gather_settings(cache: Path, grid: Grid) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Every round's chosen settings, and the candidate component counts they were chosen from.

    Returns the settings table with its round and scheme attached, and the realized
    ``component_ks`` recorded beside the rounds (ruling 36). Raises when a round is missing, and
    when two rounds tuned over different candidate sets -- the chosen k in the table means
    nothing without the one set it was chosen from.
    """
    frames: list[pd.DataFrame] = []
    component_ks: dict[str, list[int]] = {}
    seen_from: str | None = None
    for scheme, n_rounds in zip(SCHEMES, (len(grid.lines), len(grid.drugs)), strict=True):
        for index in range(n_rounds):
            path = cache / f"settings_{scheme}_{index:03d}.csv"
            if not is_done(path):
                raise SystemExit(
                    f"combine refuses: {path} is missing or invalid (run scripts/heldout_fit.py "
                    f"--scheme {scheme} --round {index})"
                )
            table = pd.read_csv(path, float_precision="round_trip")
            table.insert(0, "round", index)
            table.insert(0, "scheme", scheme)
            frames.append(table)

            record = json.loads(Path(f"{path}.done.json").read_text())
            if "component_ks" not in record:
                raise ValueError(
                    f"{path}.done.json records no component_ks; the chosen k in the settings "
                    "table means nothing without the candidate set it was chosen from, and a "
                    "sidecar defaulting to {} would satisfy ruling 36 vacuously"
                )
            realized = {
                str(model): [int(k) for k in ks]
                for model, ks in dict(record["component_ks"]).items()
            }
            if seen_from is None:
                component_ks, seen_from = realized, f"{scheme} round {index}"
            elif realized != component_ks:
                raise ValueError(
                    f"{scheme} round {index} tuned over the candidate component counts {realized}, "
                    f"but {seen_from} tuned over {component_ks}; the chosen k is only "
                    "interpretable against one candidate set"
                )
    settings = pd.concat(frames, ignore_index=True)
    if tuple(settings.columns) != SETTINGS_COLUMNS:
        raise ValueError(f"settings columns {tuple(settings.columns)} are not {SETTINGS_COLUMNS}")
    return settings, component_ks


def gather_redraws(cache: Path) -> pd.DataFrame:
    """Every redraw block, concatenated in block order.

    Refuses until all ``N_BLOCKS`` are there: an interval, a p-value or an MDE taken over some of
    the blocks is a different statistic from the one the design declares, and nothing downstream
    could tell.
    """
    paths = [Path(_REDRAWS.block_path(cache, block)) for block in range(N_BLOCKS)]
    missing = [str(path) for path in paths if not is_done(path)]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise SystemExit(
            f"combine refuses: {len(missing)} of {N_BLOCKS} redraw blocks are missing or invalid "
            f"(run scripts/heldout_redraws.py for each of: {shown})"
        )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def read_ceiling(out_dir: Path) -> dict[str, float]:
    """Each gene set's ceiling √SB from ``rung1_ceiling.csv`` (design.md section 6).

    Raises rather than defaulting: a fraction of the ceiling computed against a guessed
    denominator is a number with no meaning, and a missing ceiling is a stage that did not run.
    """
    path = out_dir / "rung1_ceiling.csv"
    if not path.exists():
        raise SystemExit(f"combine refuses: {path} is missing (run scripts/heldout_grid.py)")
    table = pd.read_csv(path)
    ceiling = {
        str(gene_set): float(value)
        for gene_set, value in zip(table["gene_set"], table["sqrt_sb"], strict=True)
    }
    missing = [gene_set for gene_set in GENE_SETS if gene_set not in ceiling]
    if missing:
        raise ValueError(f"{path} has no ceiling for gene set(s) {missing}")
    return ceiling


def scored_population(pair_scores: pd.DataFrame, grid: Grid) -> dict[tuple[str, str], Scored]:
    """The pairs every model scored, per (scheme, gene set), with each pair's grid position.

    Built exactly as ``heldout_redraws.py`` builds it -- pivot to one model per column, then drop
    any pair a model left NaN -- so the estimates computed here are taken over the same pairs the
    redraws in the cache were taken over. ``tests/test_heldout_pipeline.py`` pins that agreement.
    """
    line_position = {line: i for i, line in enumerate(grid.lines)}
    drug_position = {drug: j for j, drug in enumerate(grid.drugs)}
    prepared: dict[tuple[str, str], Scored] = {}
    for scheme in SCHEMES:
        for gene_set in GENE_SETS:
            chosen = (pair_scores["scheme"] == scheme) & (pair_scores["gene_set"] == gene_set)
            wide = pair_scores.loc[chosen].pivot(
                index=["line", "drug"], columns="model", values="r"
            )
            wide = wide.dropna(axis=0, how="any")
            lines = np.array([line_position[str(line)] for line, _ in wide.index], dtype=np.int64)
            drugs = np.array([drug_position[str(drug)] for _, drug in wide.index], dtype=np.int64)
            prepared[(scheme, gene_set)] = Scored(wide, lines, drugs)
    return prepared


def paired_differences(
    scored: pd.DataFrame, model_a: str, model_b: str, comparison: str
) -> np.ndarray:
    """``model_a``'s per-pair score minus ``model_b``'s, over the pairs both scored.

    Raises when the two models scored every pair identically. Two models that agree to the last
    bit on thousands of pairs are one fit reached by two names -- a kernel reused, a stand-in
    dispatched to its description's matrices -- and their contrast is an exact zero with no
    spread, which in the reported table is indistinguishable from a correct null. Task 10a's
    review found exactly that defect; it must fail loudly here rather than be published.
    """
    for model in (model_a, model_b):
        if model not in scored.columns:
            raise ValueError(f"{comparison}: no scored pairs for model {model!r}")
    differences = (scored[model_a] - scored[model_b]).to_numpy(dtype=np.float64)
    if differences.size and not np.any(differences):
        raise ValueError(
            f"{comparison}: {model_a} and {model_b} scored all {differences.size} pairs "
            "identically, so their contrast is an exact zero with no spread; two models "
            "collapsed onto one fit, not a null result"
        )
    return differences


def contrasts() -> list[Contrast]:
    """Every difference the run reports: the declared comparisons, then the stand-in contrasts.

    The declared ones carry their hypothesis and are Holm-adjusted within their test; the
    description-versus-its-stand-in contrasts are reported unadjusted (design section 7).
    """
    declared = [
        Contrast(
            comparison.id,
            comparison.scheme,
            comparison.model_a,
            comparison.model_b,
            comparison.hypothesis,
        )
        for comparison in COMPARISONS
    ]
    stand_ins = [
        Contrast(str(c.id), str(c.scheme), str(c.model_a), str(c.model_b), "")
        for c in _REDRAWS.stand_in_contrasts()
    ]
    return declared + stand_ins


# ==============================================================================================
# The result tables


def model_summary_table(
    population: Mapping[tuple[str, str], Scored], grid: Grid, ceiling: Mapping[str, float]
) -> pd.DataFrame:
    """Each model's mean score over the pairs every model scored, with its interval and its
    fraction of the ceiling (design section 6).

    The interval comes from redrawing the held-out units -- the 50 lines when a line was hidden,
    the 107 drugs when a drug was hidden -- exactly as a comparison's does.
    """
    rows: list[dict[str, object]] = []
    for scheme in SCHEMES:
        n_units = len(grid.lines) if scheme == "lolo" else len(grid.drugs)
        for gene_set in GENE_SETS:
            scored = population[(scheme, gene_set)]
            units = scored.line_index if scheme == "lolo" else scored.drug_index
            sqrt_sb = ceiling[gene_set]
            for model in scored.scores.columns:
                values = scored.scores[model].to_numpy(dtype=np.float64)
                summary = mean_score_ci(values, units, n_units, N_DRAWS, SUMMARY_SEED)
                rows.append(
                    {
                        "scheme": scheme,
                        "model": str(model),
                        "gene_set": gene_set,
                        "n_pairs": int(values.size),
                        "mean_r": summary["estimate"],
                        "ci_lo": summary["ci_lo"],
                        "ci_hi": summary["ci_hi"],
                        "sd": summary["sd"],
                        "mde": summary["mde"],
                        "n_draws": summary["n_draws"],
                        "n_dropped": summary["n_dropped"],
                        "sqrt_sb": sqrt_sb,
                        "fraction_of_ceiling": fraction_of_ceiling(summary["estimate"], sqrt_sb),
                    }
                )
    return pd.DataFrame(rows, columns=list(MODEL_SUMMARY_COLUMNS))


def _draws_for(
    redraws: pd.DataFrame, comparison: str, gene_set: str
) -> tuple[np.ndarray, np.ndarray]:
    """One contrast's one-way and two-way redraws, in draw order, checked to be the full set."""
    chosen = (redraws["comparison"] == comparison) & (redraws["gene_set"] == gene_set)
    rows = redraws.loc[chosen].sort_values("draw")
    if len(rows) != N_DRAWS:
        raise ValueError(
            f"{comparison} on {gene_set} genes has {len(rows)} redraws, not {N_DRAWS}; a "
            "comparison read off part of its draws is a different comparison"
        )
    if rows["draw"].tolist() != list(range(N_DRAWS)):
        raise ValueError(f"{comparison} on {gene_set} genes is missing or repeating draws")
    return (
        rows["estimate"].to_numpy(dtype=np.float64),
        rows["estimate_two_way"].to_numpy(dtype=np.float64),
    )


def comparison_table(
    population: Mapping[tuple[str, str], Scored], redraws: pd.DataFrame, grid: Grid
) -> pd.DataFrame:
    """Every comparison and stand-in contrast, on both gene sets (design section 7).

    The estimate is the mean paired difference over the scored pairs; the interval is the 2.5th
    and 97.5th percentiles of the 2,000 redraws, the p-value twice the share of redraws past
    zero, and the MDE 2.8 of their standard deviations. Holm's adjustment is applied WITHIN each
    test -- the 6 leave-one-line-out comparisons are one family, the 5 leave-one-drug-out
    comparisons another -- on responding genes; the all-genes rows and every stand-in contrast
    are reported unadjusted, and their ``p_holm`` is left empty rather than filled with the raw
    value. Each row also carries the two-way design effect and the interval-width ratio.
    """
    expected = {(contrast.id, gene_set) for contrast in contrasts() for gene_set in GENE_SETS}
    found = set(zip(redraws["comparison"], redraws["gene_set"], strict=True))
    if not expected <= found:
        raise ValueError(
            f"the redraw blocks are missing {len(expected - found)} of the "
            f"{len(expected)} (contrast, gene set) pairs the run reports: "
            f"{sorted(expected - found)[:5]}"
        )

    rows: list[dict[str, object]] = []
    for contrast in contrasts():
        for gene_set in GENE_SETS:
            scored = population[(contrast.scheme, gene_set)]
            differences = paired_differences(
                scored.scores, contrast.model_a, contrast.model_b, contrast.id
            )
            one_way, two_way = _draws_for(redraws, contrast.id, gene_set)
            summary = summarize_redraws(float(differences.mean()), one_way)
            rows.append(
                {
                    "comparison": contrast.id,
                    "scheme": contrast.scheme,
                    "model_a": contrast.model_a,
                    "model_b": contrast.model_b,
                    "hypothesis": contrast.hypothesis,
                    "holm_family": "",
                    "gene_set": gene_set,
                    "n_pairs": int(differences.size),
                    "estimate": summary["estimate"],
                    "ci_lo": summary["ci_lo"],
                    "ci_hi": summary["ci_hi"],
                    "p": summary["p"],
                    "p_holm": float("nan"),
                    "sd": summary["sd"],
                    "mde": summary["mde"],
                    "n_draws": summary["n_draws"],
                    "n_dropped": summary["n_dropped"],
                    "design_effect": design_effect(two_way, one_way),
                    "width_ratio": width_ratio(two_way, one_way),
                }
            )

    table = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    declared = sorted(comparison.id for comparison in COMPARISONS)
    for scheme in SCHEMES:
        family = (
            table["comparison"].isin(declared)
            & (table["scheme"] == scheme)
            & (table["gene_set"] == "responding")
        )
        if not family.any():
            continue
        table.loc[family, "p_holm"] = holm(table.loc[family, "p"].to_numpy(dtype=np.float64))
        table.loc[family, "holm_family"] = scheme
    return table


# ==============================================================================================
# The controls of design section 8, at the design's own sizes, seeds and ceiling values


def _control_row(
    scheme: str, planted: bool, contrast: str, summary: RedrawSummary
) -> dict[str, object]:
    mde = summary["mde"]
    return {
        "scheme": scheme,
        "planted": planted,
        "contrast": contrast,
        "estimate": summary["estimate"],
        "ci_lo": summary["ci_lo"],
        "ci_hi": summary["ci_hi"],
        "p": summary["p"],
        "sd": summary["sd"],
        "mde": mde,
        "ratio_to_mde": summary["estimate"] / mde if mde > 0 else float("nan"),
        "lambda_at_top_share": float("nan"),
        "n_lines": FIT_GRID["n_lines"],
        "n_drugs": FIT_GRID["n_drugs"],
        "n_genes": FIT_GRID["n_genes"],
        "width": FIT_GRID["width"],
        "strength": STRENGTH if planted else 0.0,
        "reliability": R_RESPONDING,
    }


def control_fit() -> pd.DataFrame:
    """The fit control: a known line- or drug-specific response, planted and then looked for.

    Rulings 26, 27 and 28, and the recorded departure of 2026-09-11. On a synthetic grid of the
    screen's size at the design's responding-gene reliability, with a response of strength 0.3
    planted in a width-20 description: the oracle that knows the true change gains at least twice
    its own MDE (the ratio is reported), the matching description's gain is detected and does not
    exceed the oracle's by more than 3 standard errors of their difference, and the random
    stand-in is held to the scheme's own requirement -- with a line hidden its gain must stay
    within its MDE, and with a drug hidden, where a random 20-column basis already spans part of
    line space, the description must beat it and the stand-in's own gain is reported rather than
    required to vanish. The rows with ``planted`` false are the negative control: nothing planted,
    so no model may gain beyond its MDE, and ridge must sit at the top of its penalty grid.
    """
    rows: list[dict[str, object]] = []
    for scheme in SCHEMES:
        control = fit_control(scheme, FIT_SEEDS[scheme])
        for name, summary in (
            ("oracle", control.oracle),
            ("description", control.description),
            ("description_minus_oracle", control.description_minus_oracle),
            ("random", control.random),
            ("description_minus_random", control.description_minus_random),
        ):
            rows.append(_control_row(scheme, True, name, summary))

    for scheme in SCHEMES:
        null = fit_null_control(scheme)
        for name, summary in null.gains.items():
            row = _control_row(scheme, False, name, summary)
            share = null.lambda_at_top.get(name)
            if share is not None:
                row["lambda_at_top_share"] = share
            rows.append(row)
    return pd.DataFrame(rows, columns=list(FIT_CONTROL_COLUMNS))


def control_split() -> pd.DataFrame:
    """The split control: a signature planted in one unit's own answers, and two fits for it.

    A random gene signature is added to each line's (or drug's) OWN answers and nowhere else. A
    deliberately broken split that leaves the hidden unit in training recovers it -- detected, and
    above its MDE -- while the shipped split scores it at zero within its MDE. Run on a small grid
    because the leaky fit is a dense solve over every (line, drug) pair (task 9's note).
    """
    rows: list[dict[str, object]] = []
    for scheme in SCHEMES:
        run = split_control_run(scheme)
        for split, prediction in (("leaky", run.leaky), ("shipped", run.shipped)):
            scores = grid_scores(prediction, run.signature_answers)
            summary = contrast_summary(scores, np.zeros_like(scores), scheme)
            rows.append(
                {
                    "scheme": scheme,
                    "split": split,
                    "n_lines": SPLIT_GRID["n_lines"],
                    "n_drugs": SPLIT_GRID["n_drugs"],
                    "n_genes": SPLIT_GRID["n_genes"],
                    "amplitude": SPLIT_AMPLITUDE,
                    "estimate": summary["estimate"],
                    "ci_lo": summary["ci_lo"],
                    "ci_hi": summary["ci_hi"],
                    "p": summary["p"],
                    "sd": summary["sd"],
                    "mde": summary["mde"],
                    "detected": bool(summary["estimate"] > summary["mde"] and summary["p"] < ALPHA),
                    "within_mde": bool(abs(summary["estimate"]) <= summary["mde"]),
                }
            )
    return pd.DataFrame(rows, columns=list(SPLIT_CONTROL_COLUMNS))


def control_score() -> pd.DataFrame:
    """The score control: synthetic answers of known reliability, through the real scoring code.

    A measurement is the true change plus noise, with reliability R = var(truth) / var(measured).
    Across genes the true change then correlates with a measurement at √R, and two independent
    measurements correlate at R -- which is why design section 6's ceiling is √SB and not SB. Both
    values are planted at the design's two reliabilities and recovered here by ``score_pairs``
    itself. An unrelated prediction scores zero.
    """
    n_lines, n_drugs, n_genes = SCORE_GRID["n_lines"], SCORE_GRID["n_drugs"], SCORE_GRID["n_genes"]
    lines, drugs = np.divmod(np.arange(n_lines * n_drugs), n_drugs)
    rows: list[dict[str, object]] = []

    for reliability in (R_RESPONDING, R_ALL):
        truth, measurement, second = planted_reliability_pool(
            reliability, n_lines * n_drugs, n_genes, seed=SCORE_POOL_SEED
        )
        responding = (
            np.random.default_rng(SCORE_RESPONDING_SEED).random((n_lines, n_drugs, n_genes)) < 0.5
        )
        answers = synthetic_answers(measurement.reshape(n_lines, n_drugs, n_genes), responding)
        masks = scoreable(answers, ())
        for name, prediction, planted in (
            ("truth", truth, float(np.sqrt(reliability))),
            ("second_measurement", second, reliability),
        ):
            frame = score_pairs(prediction, lines, drugs, answers, masks)
            rows.extend(_score_control_rows(frame, reliability, name, planted))

    unrelated_pool, responding_seed, prediction_seed = SCORE_UNRELATED_SEEDS
    _, measurement, _ = planted_reliability_pool(
        R_RESPONDING, n_lines * n_drugs, n_genes, seed=unrelated_pool
    )
    responding = np.random.default_rng(responding_seed).random((n_lines, n_drugs, n_genes)) < 0.5
    answers = synthetic_answers(measurement.reshape(n_lines, n_drugs, n_genes), responding)
    unrelated = np.random.default_rng(prediction_seed).standard_normal((n_lines * n_drugs, n_genes))
    frame = score_pairs(unrelated, lines, drugs, answers, scoreable(answers, ()))
    rows.extend(_score_control_rows(frame, R_RESPONDING, "unrelated", 0.0))

    return pd.DataFrame(rows, columns=list(SCORE_CONTROL_COLUMNS))


def _score_control_rows(
    frame: pd.DataFrame, reliability: float, prediction: str, planted: float
) -> list[dict[str, object]]:
    """One scored prediction's row per gene set, with the Monte Carlo error of its own mean."""
    rows: list[dict[str, object]] = []
    for gene_set in GENE_SETS:
        scores = frame.loc[frame["gene_set"] == gene_set, "r"].to_numpy(dtype=np.float64)
        mean = float(scores.mean())
        se = float(scores.std(ddof=1) / np.sqrt(scores.size))
        rows.append(
            {
                "reliability": reliability,
                "gene_set": gene_set,
                "prediction": prediction,
                "planted": planted,
                "mean_r": mean,
                "se": se,
                "n_pairs": int(scores.size),
                "within_3_se": bool(abs(mean - planted) <= SE_MULTIPLE * se),
            }
        )
    return rows


def control_null() -> pd.DataFrame:
    """The null control: what the redraws detect, and what they invent.

    Over 200 synthetic comparisons with the screen's own unit structure: shifted by exactly the
    MDE its own redraws estimate, a comparison is detected about 80% of the time -- which is what
    an MDE means -- and with nothing planted, at most 5%. Both rates are read against a binomial
    99% interval fixed before the run (invariant 9).
    """
    repetitions = null_repetition_pvalues()

    rows: list[dict[str, object]] = []
    detections = int(np.count_nonzero(repetitions.p_planted < ALPHA))
    low, high = stats.binom.interval(0.99, REPETITIONS, DETECTION_RATE)
    rows.append(
        {
            "check": "detection_at_mde",
            "target_rate": DETECTION_RATE,
            "repetitions": REPETITIONS,
            "alpha": ALPHA,
            "detections": detections,
            "rate": detections / REPETITIONS,
            "ci_lo": float(low),
            "ci_hi": float(high),
            "inside_interval": bool(float(low) <= detections <= float(high)),
        }
    )
    false_positives = int(np.count_nonzero(repetitions.p_null < ALPHA))
    fp_low, fp_high = stats.binom.interval(0.99, REPETITIONS, ALPHA)
    rows.append(
        {
            "check": "false_positive_rate",
            "target_rate": ALPHA,
            "repetitions": REPETITIONS,
            "alpha": ALPHA,
            "detections": false_positives,
            "rate": false_positives / REPETITIONS,
            "ci_lo": float(fp_low),
            "ci_hi": float(fp_high),
            # One-sided: a rate below the interval is a conservative test, not a defect.
            "inside_interval": bool(false_positives <= float(fp_high)),
        }
    )
    return pd.DataFrame(rows, columns=list(NULL_CONTROL_COLUMNS))


def build_inputs(out_dir: Path) -> dict[str, Path]:
    """The tables the earlier stages left in ``--out-dir`` that the build control reads."""
    files = {
        "identity_match": out_dir / "rung1_identity_match.csv",
        "cells": out_dir / "rung1_cells.csv",
        "weights_check": out_dir / "rung1_weights_check.json",
    }
    for version in STACK_VERSIONS:
        files[f"identity_match_stack_{version}"] = (
            out_dir / f"rung1_identity_match_stack_{version}.csv"
        )
    for description in ("expression", "pca", "nmf"):
        files[f"identity_grid_{description}"] = out_dir / f"rung1_identity_grid_{description}.csv"
    for version in STACK_VERSIONS:
        files[f"identity_grid_stack_{version}"] = (
            out_dir / f"rung1_identity_grid_stack_{version}.csv"
        )
    return files


def read_build_tables(
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """The build stage's own tables: the identity matches, their grids, the cells, the weights.

    Refuses, naming what is missing, rather than writing a build control with silent gaps: a
    description whose identity row never arrived would otherwise simply not appear in the table.
    """
    files = build_inputs(out_dir)
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise SystemExit(
            f"combine refuses: {len(missing)} build-stage table(s) are missing from {out_dir} "
            f"(run the descriptions and embed stages): {', '.join(missing[:5])}"
        )
    matches = pd.concat(
        [pd.read_csv(files["identity_match"])]
        + [pd.read_csv(files[f"identity_match_stack_{v}"]) for v in STACK_VERSIONS],
        ignore_index=True,
    )
    grids = {
        description: pd.read_csv(files[f"identity_grid_{description}"])
        for description in ("expression", "pca", "nmf")
    }
    for version in STACK_VERSIONS:
        grids[f"stack_{version}"] = pd.read_csv(files[f"identity_grid_stack_{version}"])
    cells = pd.read_csv(files["cells"])
    weights = cast(dict[str, Any], json.loads(files["weights_check"].read_text()))
    absent = [key for key in WEIGHTS_CHECK_KEYS if key not in weights]
    if absent:
        raise SystemExit(
            f"combine refuses: {files['weights_check']} has no {absent}. The build control's "
            "weights row is H2's premise -- that the drug fine-tune is a different model from "
            "the base checkpoint -- and defaulting the missing keys would publish "
            "'encoders differ, 0 tensors differ' from an empty or renamed file"
        )
    return matches, grids, cells, weights


def control_build(matches: pd.DataFrame, weights: Mapping[str, Any]) -> pd.DataFrame:
    """The build control: does a description identify the line it was built from?

    Each description is built from half of a line's untreated cells and matched against every
    line's other half; the share of lines that match themselves is read against the same
    statistic with the cells' line labels shuffled. The last row is the drug fine-tune's weights
    check: if its encoder were identical to the base checkpoint, the Stack drug and Stack base
    descriptions would be the same numbers and H2 would compare a model with itself.
    """
    rows: list[dict[str, object]] = []
    for row in matches.itertuples():
        share = float(cast(Any, row).identity_share)
        p99 = float(cast(Any, row).null_p99)
        rows.append(
            {
                "check": "half_vs_half_identity",
                "subject": str(cast(Any, row).description),
                "identity_share": share,
                "null_mean": float(cast(Any, row).null_mean),
                "null_p99": p99,
                "n_lines": int(cast(Any, row).n_lines),
                "n_shuffles": int(cast(Any, row).n_shuffles),
                "above_null_p99": bool(share > p99),
                "n_shared_tensors": float("nan"),
                "n_identical_tensors": float("nan"),
                "n_different_tensors": float("nan"),
                "encoders_identical": None,
            }
        )
    rows.append(
        {
            "check": "drug_finetune_weights_differ",
            "subject": "stack_drug",
            "identity_share": float("nan"),
            "null_mean": float("nan"),
            "null_p99": float("nan"),
            "n_lines": None,
            "n_shuffles": None,
            "above_null_p99": None,
            # Indexed, never defaulted: read_build_tables has already refused a file missing any
            # of these, and a direct KeyError here is still louder than a manufactured number.
            "n_shared_tensors": int(weights["n_shared"]),
            "n_identical_tensors": int(weights["n_identical"]),
            "n_different_tensors": int(weights["n_different"]),
            "encoders_identical": bool(weights["identical_all"]),
        }
    )
    return pd.DataFrame(rows, columns=list(BUILD_CONTROL_COLUMNS))


# ==============================================================================================
# Writing: every output with its completion record, and the run's parameter sidecar


def output_paths(out_dir: Path) -> list[Path]:
    """Every file the combine writes, in the order it writes them."""
    names = [
        "rung1_pair_scores.csv.gz",
        "rung1_model_summary.csv",
        "rung1_comparisons.csv",
        "rung1_settings.csv",
        *(f"rung1_control_{name}.csv" for name in ("build", "split", "fit", "score", "null")),
        LEAKAGE_JSON,
        PARAMS_JSON,
    ]
    return [out_dir / name for name in names] + [out_dir / "figures" / name for name in FIGURES]


def write_table(path: Path, frame: pd.DataFrame, columns: Sequence[str]) -> Path:
    """Write one table and its completion record, checking it carries the contract's columns.

    A gzipped table is written with a zero timestamp, so the same table twice is the same bytes:
    gzip's default header carries the time of writing, which would make every rerun differ from
    the last and break the staged-equals-one-process comparison (invariant 8).
    """
    if tuple(frame.columns) != tuple(columns):
        raise ValueError(f"{path.name} columns {tuple(frame.columns)} are not {tuple(columns)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".csv.gz"):
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    else:
        frame.to_csv(path, index=False)
    write_record(path, {"rows": len(frame)})
    return path


def _git_sha() -> str:
    """The commit this run was made at.

    Raises rather than recording ``"unknown"``: project rule 1 makes the producing commit one of
    the three things that cannot be recovered afterwards, so a provenance record without it is
    not a provenance record -- it only looks like one.
    """
    try:
        finished = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(
            f"combine refuses: git could not name the commit this run was made at ({error}); "
            "the parameter sidecar cannot record the producing commit"
        ) from error
    sha = finished.stdout.strip()
    if not sha:
        raise SystemExit("combine refuses: git rev-parse HEAD returned nothing")
    return sha


def input_files(cache: Path, out_dir: Path, grid: Grid) -> dict[str, Path]:
    """Every file the run read, keyed ``<root>/<name>``.

    Keyed by the path relative to its root rather than absolutely, so two runs of the same cache
    into different directories pin the same names -- the sidecar records what was read, not where
    the reader stood. The drug table's own checksum is inside ``rung1_grid.json``'s
    ``source_sha256``, and that file is pinned here.
    """
    files: dict[str, Path] = {}

    def add(root_name: str, root: Path, path: Path) -> None:
        """Pin one input, refusing if it is not there.

        A missing file must not simply be left out: this is the record the audit pins bytes
        with, and an input that silently stops being listed reads as "not an input to this run"
        rather than as the gap it is.
        """
        if not path.exists():
            raise SystemExit(
                f"combine refuses: {path} is missing, so the parameter sidecar cannot pin it. "
                "Every input the run read has to appear in the record with its sha256"
            )
        files[f"{root_name}/{path.relative_to(root).as_posix()}"] = path

    for name in ("answers.npz", "descriptions.npz", "tanimoto.npz"):
        add("cache", cache, cache / name)
    for version in STACK_VERSIONS:
        add("cache", cache, cache / f"embedding_{version}.parquet")
    for scheme, n_rounds in zip(SCHEMES, (len(grid.lines), len(grid.drugs)), strict=True):
        for index in range(n_rounds):
            add("cache", cache, cache / f"scores_{scheme}_{index:03d}.parquet")
            add("cache", cache, cache / f"settings_{scheme}_{index:03d}.csv")
    for block in range(N_BLOCKS):
        add("cache", cache, Path(_REDRAWS.block_path(cache, block)))

    add("out_dir", out_dir, out_dir / "rung1_ceiling.csv")
    for path in build_inputs(out_dir).values():
        add("out_dir", out_dir, path)

    # The grid is written by its own stage into --out-dir and copied into the cache by some
    # runs, so it legitimately lives in either root -- but it must be in one of them.
    grid_roots = [("cache", cache), ("out_dir", out_dir)]
    found = [(name, root) for name, root in grid_roots if (root / "rung1_grid.json").exists()]
    if not found:
        raise SystemExit(
            f"combine refuses: rung1_grid.json is in neither {cache} nor {out_dir}, so the "
            "sidecar cannot pin the grid the run was restricted to"
        )
    for name, root in found:
        add(name, root, root / "rung1_grid.json")
    return files


def params_record(
    cache: Path,
    out_dir: Path,
    grid: Grid,
    ceiling: Mapping[str, float],
    component_ks: Mapping[str, Sequence[int]],
    args: Mapping[str, str],
) -> dict[str, Any]:
    """The run's provenance record (project rule 1): the commit, the arguments, every seed, the
    realized component counts, and the sha256 of every input file."""
    return {
        "result": "rung1_run",
        "git_sha": _git_sha(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "args": dict(args),
        "n_lines": len(grid.lines),
        "n_drugs": len(grid.drugs),
        "excluded_pairs": [list(pair) for pair in grid.excluded_pairs],
        "n_draws": N_DRAWS,
        "n_blocks": N_BLOCKS,
        "draws_per_block": N_DRAWS // N_BLOCKS,
        "ceiling": dict(ceiling),
        "component_ks": {model: list(ks) for model, ks in component_ks.items()},
        "seeds": {
            "redraw_base_seed": REDRAW_BASE_SEED,
            "model_summary": SUMMARY_SEED,
            "random_stand_ins": dict(RANDOM_SEEDS),
            "nmf": NMF_SEED,
            "controls": {
                "redraw": REDRAW_SEED,
                "fit": dict(FIT_SEEDS),
                "fit_null_grid": FIT_NULL_GRID_SEED,
                "fit_null_stand_in": FIT_NULL_STAND_IN_SEED,
                "split": {scheme: list(seeds) for scheme, seeds in SPLIT_SEEDS.items()},
                "score_pool": SCORE_POOL_SEED,
                "score_responding": SCORE_RESPONDING_SEED,
                "score_unrelated": list(SCORE_UNRELATED_SEEDS),
                "null_units": NULL_UNIT_SEED,
            },
        },
        "inputs": {
            name: sha256_file(path)
            for name, path in sorted(input_files(cache, out_dir, grid).items())
        },
    }


def combine(
    out_dir: Path,
    cache: Path,
    *,
    grid: Grid,
    drug_metadata: pd.DataFrame,
    args: Mapping[str, str] | None = None,
) -> None:
    """Gather the run and write its tables, controls, figures, leakage records and provenance.

    Skips entirely when every output is already done and valid, so a rerun after a failed figure
    step costs nothing and a completed combine is never silently rewritten.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir)
    if all(is_done(path) for path in paths):
        print(f"every combine output in {out_dir} is already done, skipping")
        return

    ceiling = read_ceiling(out_dir)
    pair_scores = gather_pair_scores(cache, grid)
    settings, component_ks = gather_settings(cache, grid)
    redraws = gather_redraws(cache)
    matches, identity_grids, cells, weights = read_build_tables(out_dir)

    population = scored_population(pair_scores, grid)
    summary = model_summary_table(population, grid, ceiling)
    comparisons = comparison_table(population, redraws, grid)

    write_table(out_dir / "rung1_pair_scores.csv.gz", pair_scores, PAIR_SCORE_COLUMNS)
    write_table(out_dir / "rung1_model_summary.csv", summary, MODEL_SUMMARY_COLUMNS)
    write_table(out_dir / "rung1_comparisons.csv", comparisons, COMPARISON_COLUMNS)
    write_table(out_dir / "rung1_settings.csv", settings, SETTINGS_COLUMNS)

    build = control_build(matches, weights)
    split = control_split()
    fit = control_fit()
    score = control_score()
    null = control_null()
    write_table(out_dir / "rung1_control_build.csv", build, BUILD_CONTROL_COLUMNS)
    write_table(out_dir / "rung1_control_split.csv", split, SPLIT_CONTROL_COLUMNS)
    write_table(out_dir / "rung1_control_fit.csv", fit, FIT_CONTROL_COLUMNS)
    write_table(out_dir / "rung1_control_score.csv", score, SCORE_CONTROL_COLUMNS)
    write_table(out_dir / "rung1_control_null.csv", null, NULL_CONTROL_COLUMNS)

    leakage_path = out_dir / LEAKAGE_JSON
    leakage_path.write_text(
        json.dumps(leakage_records(leakage_profiles(grid, drug_metadata)), indent=2, sort_keys=True)
        + "\n"
    )
    write_record(leakage_path)

    figures = out_dir / "figures"
    written = [
        fig_build(cells, matches, identity_grids, weights, figures / FIGURES[0]),
        fig_split(pair_scores, split, grid.lines, grid.drugs, figures / FIGURES[1]),
        fig_fit(settings, summary, fit, figures / FIGURES[2]),
        fig_score(summary, pd.read_csv(out_dir / "rung1_ceiling.csv"), score, figures / FIGURES[3]),
        fig_null(comparisons, redraws, figures / FIGURES[4]),
    ]
    for path in written:
        write_record(Path(path))

    params_path = out_dir / PARAMS_JSON
    recorded = args if args is not None else {"cache": str(cache), "out_dir": str(out_dir)}
    params_path.write_text(
        json.dumps(
            params_record(cache, out_dir, grid, ceiling, component_ks, recorded),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_record(params_path)

    print(f"wrote {len(paths)} files into {out_dir}")
    print(
        f"  {len(pair_scores)} pair scores, {len(summary)} model rows, "
        f"{len(comparisons)} comparison rows"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--drug-metadata", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    combine(
        args.out_dir,
        args.cache,
        grid=load_grid(args.grid),
        drug_metadata=load_drug_metadata(args.drug_metadata),
        args={key: str(value) for key, value in vars(args).items()},
    )


if __name__ == "__main__":
    main()
