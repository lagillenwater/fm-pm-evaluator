"""Task 10a of rung 1: one block of the redraws every comparison is read with.

Design.md section 7. A comparison is the mean, over the pairs both models scored, of one model's
score minus the other's. Its uncertainty comes from redrawing the held-out units with repeats --
the 50 lines when a line is hidden, the 107 drugs when a drug is hidden -- 2,000 times, and
recomputing the mean each time. The 2,000 redraws run as ``N_BLOCKS`` blocks of
``DRAWS_PER_BLOCK``, one job each; block ``b`` is seeded ``redraw_block_seed(REDRAW_BASE_SEED,
b)``, so every draw comes from its block's seed alone and the blocks reproduce the same draws
whatever order or process they run in (invariant 4).

What is redrawn: every comparison declared in ``fmharness.heldout.COMPARISONS`` (design section
7's table, 6 leave-one-line-out and 5 leave-one-drug-out), plus -- for both schemes -- each
description against its own random stand-in, which design section 7 also reports, unadjusted.
Each is redrawn on both gene sets, and also with lines and drugs redrawn together
(``two_way_estimates``), whose spread against the one-way spread is the design effect 10b's
combine reads off. Only the redraws are computed here: the intervals, p-values, MDEs and Holm
adjustment are computed once, in the combine, over the concatenated blocks.

The pairs a comparison is taken over are those **every** model scored on that scheme and gene
set: a pair with a NaN score (fewer than 50 usable genes, or no variance on one side) is dropped
for every model, so all models share one scored population.

Writes ``redraws_{b}.parquet`` into ``--cache`` -- ``comparison, gene_set, draw, estimate,
estimate_two_way``, with ``draw`` numbered across blocks so the blocks concatenate into
0..1,999 -- and its ``<name>.done.json``; a block already done skips. Reads every round's
``scores_{scheme}_{i:03d}.parquet`` and refuses to run until all of them are there.

    uv run python scripts/heldout_redraws.py --block 0 \\
        --grid docs/tasks/rung1-held-out-prediction/rung1_grid.json \\
        --cache /scratch/alpine/$USER/rung1_cache
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmharness.heldout import COMPARISONS, MODELS, Scheme  # noqa: E402
from fmharness.heldout.comparisons import (  # noqa: E402
    DRAWS_PER_BLOCK,
    N_BLOCKS,
    redraw_block_seed,
    redraw_estimates,
    two_way_estimates,
)
from fmharness.heldout.grid import Grid, load_grid  # noqa: E402
from fmharness.heldout.records import is_done, write_record  # noqa: E402
from fmharness.heldout.scoring import GENE_SETS  # noqa: E402

#: The base seed of the redraws (ruling 33): block ``b`` draws with ``REDRAW_BASE_SEED + b``.
#: Distinct from every other seed the run uses -- the stand-ins' 101-106
#: (``descriptions.RANDOM_SEEDS``), NMF's 0 -- so no two stages share a random stream. 10b's
#: params sidecar records it with the run.
REDRAW_BASE_SEED = 7000

#: The schemes a comparison can be taken under, in the order blocks write them.
SCHEMES: tuple[Scheme, ...] = ("lolo", "lodo")

#: ``redraws_{b}.parquet``'s columns, in order (the data contract).
REDRAW_COLUMNS: tuple[str, ...] = (
    "comparison",
    "gene_set",
    "draw",
    "estimate",
    "estimate_two_way",
)


@dataclass(frozen=True)
class Contrast:
    """One redrawn difference: ``model_a`` minus ``model_b``, on ``scheme``.

    The declared comparisons of ``COMPARISONS`` carry a hypothesis; the stand-in contrasts do
    not (design section 7 reports them unadjusted, beside the hypothesis tests), so a contrast
    holds only what a redraw needs. ``id`` is the comparison id the tables are keyed by, built
    the same way ``COMPARISONS``' ids are.
    """

    id: str
    scheme: Scheme
    model_a: str
    model_b: str


@dataclass(frozen=True)
class ScoredPairs:
    """One scheme and gene set's scored pairs, shared by every contrast taken over them.

    ``scores`` has one row per (line, drug) pair and one column per model, holding only the
    pairs every model scored; ``line_index`` and ``drug_index`` give each row's position in the
    grid's line and drug order -- the unit a redraw draws.
    """

    scores: pd.DataFrame
    line_index: np.ndarray
    drug_index: np.ndarray


def stand_in_contrasts() -> tuple[Contrast, ...]:
    """Each description against its own random stand-in, for both schemes: the contrast that
    says whether a description gains from what it describes, or from the method every model
    shares. Design section 7 reports these unadjusted; they are not in ``COMPARISONS``, whose
    rows are the hypothesis tests."""
    return tuple(
        Contrast(f"{scheme}_{spec.id}_vs_random_{spec.id}", scheme, spec.id, f"random_{spec.id}")
        for scheme in SCHEMES
        for spec in MODELS.values()
        if spec.kind == "ridge"
        and scheme in spec.schemes
        and f"random_{spec.id}" in MODELS
        and scheme in MODELS[f"random_{spec.id}"].schemes
    )


def all_contrasts() -> tuple[Contrast, ...]:
    """Every difference a block redraws: the declared comparisons, then the stand-in contrasts."""
    declared = tuple(
        Contrast(comparison.id, comparison.scheme, comparison.model_a, comparison.model_b)
        for comparison in COMPARISONS
    )
    return declared + stand_in_contrasts()


def load_pair_scores(cache: Path, grid: Grid) -> pd.DataFrame:
    """Every round's per-pair scores, both schemes, concatenated in round order.

    Refuses (``SystemExit``) unless every one of the grid's 50 + 107 rounds has a valid
    completion record: a comparison taken over some of the rounds would be a different
    comparison, silently.
    """
    paths = [
        cache / f"scores_{scheme}_{index:03d}.parquet"
        for scheme, n_rounds in zip(SCHEMES, (len(grid.lines), len(grid.drugs)), strict=True)
        for index in range(n_rounds)
    ]
    missing = [str(path) for path in paths if not is_done(path)]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise SystemExit(
            f"redraws refuse: {len(missing)} of {len(paths)} rounds are missing or invalid "
            f"(run scripts/heldout_fit.py for each of: {shown})"
        )
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _scored_pairs(pair_scores: pd.DataFrame, grid: Grid) -> dict[tuple[str, str], ScoredPairs]:
    """The scored pairs of each (scheme, gene set), as one model-per-column table.

    Pairs come in (line, drug) order, and only pairs every model scored are kept: dropping a
    pair with a NaN score for every model is what keeps the models on one scored population.
    """
    line_position = {line: i for i, line in enumerate(grid.lines)}
    drug_position = {drug: j for j, drug in enumerate(grid.drugs)}
    prepared: dict[tuple[str, str], ScoredPairs] = {}
    for scheme in SCHEMES:
        for gene_set in GENE_SETS:
            chosen = (pair_scores["scheme"] == scheme) & (pair_scores["gene_set"] == gene_set)
            wide = pair_scores.loc[chosen].pivot(
                index=["line", "drug"], columns="model", values="r"
            )
            wide = wide.dropna(axis=0, how="any")
            lines = np.array([line_position[str(line)] for line, _ in wide.index], dtype=np.int64)
            drugs = np.array([drug_position[str(drug)] for _, drug in wide.index], dtype=np.int64)
            prepared[(scheme, gene_set)] = ScoredPairs(wide, lines, drugs)
    return prepared


def run_redraw_block(
    pair_scores: pd.DataFrame,
    block: int,
    draws_per_block: int = DRAWS_PER_BLOCK,
    seed: int = REDRAW_BASE_SEED,
    *,
    grid: Grid,
) -> pd.DataFrame:
    """One block of redraws of every contrast, on both gene sets.

    ``pair_scores`` is every round's per-pair scores (``load_pair_scores``); ``seed`` is the base
    seed, from which this block's seed is ``redraw_block_seed(seed, block)``. ``grid`` is needed
    for the number of held-out units: a line (or drug) that no pair was scored on is still one of
    the 50 (or 107) units a redraw draws from, and only the grid says how many there are.

    Returns one row per (contrast, gene set, draw) with the columns of ``REDRAW_COLUMNS``:
    ``estimate`` redraws the hidden unit alone -- lines when a line is hidden, drugs when a drug
    is hidden -- and ``estimate_two_way`` redraws lines and drugs together. ``draw`` numbers the
    draws across blocks (``block * draws_per_block`` onward), so concatenated blocks run 0..N-1.
    """
    if not 0 <= block < N_BLOCKS:
        raise ValueError(f"block {block} is outside 0..{N_BLOCKS - 1}")
    if draws_per_block < 1:
        raise ValueError(f"draws_per_block must be at least 1; got {draws_per_block}")

    block_seed = redraw_block_seed(seed, block)
    draws = np.arange(block * draws_per_block, (block + 1) * draws_per_block, dtype=np.int64)
    n_lines, n_drugs = len(grid.lines), len(grid.drugs)
    prepared = _scored_pairs(pair_scores, grid)

    frames: list[pd.DataFrame] = []
    for contrast in all_contrasts():
        for gene_set in GENE_SETS:
            scored = prepared[(contrast.scheme, gene_set)]
            absent = [
                model
                for model in (contrast.model_a, contrast.model_b)
                if model not in scored.scores.columns
            ]
            if absent:
                raise ValueError(
                    f"{contrast.id} on {gene_set} genes: no scored pairs for {absent}; "
                    "every model of a contrast must have been fitted in every round"
                )
            differences = (
                scored.scores[contrast.model_a] - scored.scores[contrast.model_b]
            ).to_numpy(dtype=np.float64)
            unit = scored.line_index if contrast.scheme == "lolo" else scored.drug_index
            n_units = n_lines if contrast.scheme == "lolo" else n_drugs
            frames.append(
                pd.DataFrame(
                    {
                        "comparison": np.full(draws_per_block, contrast.id, dtype=object),
                        "gene_set": np.full(draws_per_block, gene_set, dtype=object),
                        "draw": draws,
                        "estimate": redraw_estimates(
                            differences, unit, n_units, draws_per_block, block_seed
                        ),
                        "estimate_two_way": two_way_estimates(
                            differences,
                            scored.line_index,
                            scored.drug_index,
                            n_lines,
                            n_drugs,
                            draws_per_block,
                            block_seed,
                        ),
                    }
                )
            )

    table = pd.concat(frames, ignore_index=True)
    if tuple(table.columns) != REDRAW_COLUMNS:
        raise ValueError(f"redraw columns {tuple(table.columns)} are not {REDRAW_COLUMNS}")
    return table


def block_path(cache: Path, block: int) -> Path:
    """The block's output path."""
    return cache / f"redraws_{block}.parquet"


def write_block(cache: Path, block: int, draws: pd.DataFrame, block_seed: int) -> Path:
    """Write one block's redraws and its completion record.

    ``block_seed`` is the seed the draws actually came from, passed in rather than recomputed
    from this module's base seed: a block drawn from another base would otherwise be recorded
    with provenance it does not have.
    """
    cache.mkdir(parents=True, exist_ok=True)
    path = block_path(cache, block)
    draws.to_parquet(path, index=False)
    write_record(path, {"block": block, "rows": len(draws), "block_seed": block_seed})
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()

    block = int(args.block)
    if not 0 <= block < N_BLOCKS:
        raise SystemExit(f"--block {block} is outside 0..{N_BLOCKS - 1}")

    args.cache.mkdir(parents=True, exist_ok=True)
    path = block_path(args.cache, block)
    if is_done(path):
        print(f"{path} already done, skipping")
        return

    grid = load_grid(args.grid)
    pair_scores = load_pair_scores(args.cache, grid)
    block_seed = redraw_block_seed(REDRAW_BASE_SEED, block)
    draws = run_redraw_block(pair_scores, block, seed=REDRAW_BASE_SEED, grid=grid)
    write_block(args.cache, block, draws, block_seed)
    print(f"wrote {path} ({len(draws)} rows, seed {block_seed})")


if __name__ == "__main__":
    main()
