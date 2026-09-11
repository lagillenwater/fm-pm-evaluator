"""Task 10a of rung 1: one round of the run -- one line, or one drug, hidden.

Design.md sections 3, 5 and 6. A **leave-one-line-out** round hides line ``i``: every model
declared for that scheme learns from the other 49 lines and predicts all 107 drugs for the
hidden line. A **leave-one-drug-out** round hides drug ``i``: models learn from the other 106
drugs and predict that drug in all 50 lines. One round is one job of task 11's 157-task array
(0-49 the lines, 50-156 the drugs), so a round is self-contained: it reads the cache, fits,
scores, writes its two files with their completion records, and a round already done skips.

Which models run, and what each one is, is declared once in ``fmharness.heldout.MODELS`` --
this script reads that declaration rather than repeating the list. Every setting (the ridge
penalty, the number of PCA or NMF components, the number of nearest lines) is chosen inside the
round by the closed-form leave-one-out tuning in ``fmharness.heldout.models``, from training
data only. Each description's **random stand-in** is drawn through the one helper
``stand_in`` below, seeded from ``descriptions.RANDOM_SEEDS`` and no other source, so every
stage of the run uses the same stand-in for a description.

Scoring is ``scoring.score_pairs`` on both gene sets, over the pairs ``answers.scoreable``
keeps -- at least 50 qualifying genes, and never the leaked pairs of design section 7, which
are removed for every model. It is called once per model per round, on that round's pairs
alone (at most 107), never on the whole 5,350-pair grid at once.

Writes into ``--cache``:

* ``scores_{scheme}_{i:03d}.parquet`` -- ``scheme, round, model, line, drug, gene_set, r, n_genes``
* ``settings_{scheme}_{i:03d}.csv`` -- ``model, lambda, k, loss_min, lambda_at_edge``

each with its ``<name>.done.json`` completion record. A model that chooses no penalty (the drug
average) or no component count writes an empty ``lambda`` or ``k``; ``loss_min`` is the tuning
loss at the chosen setting, empty for a model that tunes nothing.

Reads from ``--cache``: ``answers.npz``, ``descriptions.npz``, ``tanimoto.npz`` and
``embedding_{version}.parquet`` for the three Stack versions; the grid comes from ``--grid``.

    uv run python scripts/heldout_fit.py --scheme lolo --round 0 \\
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmharness.heldout import MODELS, ModelSpec, Scheme  # noqa: E402
from fmharness.heldout.answers import Answers, scoreable  # noqa: E402
from fmharness.heldout.descriptions import (  # noqa: E402
    RANDOM_SEEDS,
    linear_kernel,
    random_stand_in,
)
from fmharness.heldout.grid import Grid, load_grid  # noqa: E402
from fmharness.heldout.models import (  # noqa: E402
    Fit,
    drug_average,
    nearest_lines_lolo,
    ridge_lodo,
    ridge_lodo_k,
    ridge_lolo,
    ridge_lolo_k,
    similarity_from_description,
)
from fmharness.heldout.records import is_done, write_record  # noqa: E402
from fmharness.heldout.scoring import MIN_GENES, SCORE_COLUMNS, score_pairs  # noqa: E402

#: The candidate component counts a PCA or NMF ridge chooses between (design.md section 5).
#: PCA takes the first k of its ordered components; NMF has a separate ``nmf_{k}`` matrix per k.
COMPONENT_KS: tuple[int, ...] = (2, 5, 10, 15, 20)

#: The three Stack versions, each with its own per-line mean embedding in the cache.
STACK_VERSIONS: tuple[str, ...] = ("base", "cytokine", "drug")

#: The design's grid (section 2). At this size the candidate component counts the round actually
#: tunes over must be ``COMPONENT_KS`` exactly -- a cache missing an ``nmf_{k}`` array, or holding
#: a narrower PCA, would otherwise quietly run a model with fewer candidates than the design
#: declares. A smaller grid (a test fixture) legitimately has fewer lines than components.
FULL_GRID_LINES, FULL_GRID_DRUGS = 50, 107

#: ``scores_{scheme}_{i:03d}.parquet``'s columns, in order (the data contract).
SCORE_FILE_COLUMNS: tuple[str, ...] = ("scheme", "round", "model", *SCORE_COLUMNS)

#: ``settings_{scheme}_{i:03d}.csv``'s columns, in order (the data contract).
SETTINGS_COLUMNS: tuple[str, ...] = ("model", "lambda", "k", "loss_min", "lambda_at_edge")


def stand_in(description: str, n_lines: int, width: int) -> np.ndarray:
    """The random stand-in for ``description``: ``width`` standard normal columns per line.

    The one helper every stage draws a stand-in through. Its seed comes from
    ``descriptions.RANDOM_SEEDS`` and nowhere else, so the stand-in a description is compared
    against is the same array in every stage -- the arrays ``descriptions.npz`` stores for
    expression, PCA and NMF are exactly these, and the Stack versions' stand-ins (whose width
    is only known once an embedding is loaded) are drawn here from the same declaration.
    """
    if description not in RANDOM_SEEDS:
        raise KeyError(
            f"no declared stand-in seed for description {description!r}; "
            f"RANDOM_SEEDS holds {sorted(RANDOM_SEEDS)}"
        )
    return random_stand_in(n_lines, width, RANDOM_SEEDS[description])


@dataclass(frozen=True)
class RoundInputs:
    """Everything a round reads, loaded once: the same object serves every round in a process.

    ``delta0`` is the answer with untested entries set to 0 (untested genes count as zero change
    in training) and ``tested`` marks the entries the screen measured -- the pair of arrays every
    model in ``fmharness.heldout.models`` takes. ``masks`` is ``answers.scoreable``'s bool
    ``[L, D]`` array per gene set, so every model is scored on the same pairs (invariant 2).
    ``kernels`` and ``components`` are both keyed by **model id**, never by description: a model
    with one kernel is in ``kernels``, a model whose component count is tuned is in
    ``components`` with one matrix per candidate count. A random stand-in therefore has its own
    entry built from its own stand-in matrix, and cannot be fitted on the description it stands
    in for. ``similarity`` is the line-by-line correlation nearest lines ranks by; ``tanimoto``
    is the drugs' chemical similarity in grid order.
    """

    grid: Grid
    answers: Answers
    masks: dict[str, np.ndarray]
    delta0: np.ndarray
    tested: np.ndarray
    kernels: dict[str, np.ndarray]
    components: dict[str, dict[int, np.ndarray]]
    similarity: np.ndarray
    tanimoto: np.ndarray


def models_for(scheme: Scheme) -> list[ModelSpec]:
    """Every model declared for ``scheme``, in declaration order."""
    return [spec for spec in MODELS.values() if scheme in spec.schemes]


def _load_answers(cache: Path, grid: Grid) -> Answers:
    """``answers.npz`` as ``Answers``; raises unless its lines and drugs are the grid's, in grid
    order (every later index is a position in that order)."""
    with np.load(cache / "answers.npz") as npz:
        lines = tuple(str(v) for v in npz["lines"])
        drugs = tuple(str(v) for v in npz["drugs"])
        genes = tuple(str(v) for v in npz["genes"])
        delta = npz["delta"]
        responding = npz["responding"]
    if lines != grid.lines or drugs != grid.drugs:
        raise ValueError(
            f"answers.npz holds {len(lines)} lines and {len(drugs)} drugs that are not the "
            f"grid's {len(grid.lines)} lines and {len(grid.drugs)} drugs in grid order"
        )
    return Answers(lines=lines, drugs=drugs, genes=genes, delta=delta, responding=responding)


def _stack_embedding(cache: Path, version: str, grid: Grid) -> np.ndarray:
    """One Stack version's per-line mean embedding, reindexed to grid line order; raises naming
    any grid line the embedding is missing."""
    path = cache / f"embedding_{version}.parquet"
    frame = pd.read_parquet(path)
    present = {str(v) for v in frame.index}
    missing = [line for line in grid.lines if line not in present]
    if missing:
        raise ValueError(f"{path} is missing grid lines: {missing}")
    frame.index = pd.Index([str(v) for v in frame.index])
    return frame.loc[list(grid.lines)].to_numpy(dtype=np.float64)


def _descriptions(
    cache: Path, grid: Grid
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """Each description over the grid's lines, and the candidate component matrices of the two
    descriptions whose count is chosen inside the round.

    The description returned for ``pca`` and ``nmf`` is their widest candidate: a stand-in of
    "the same width per description" (design section 4) is that width, which is how
    ``descriptions.npz``'s own stand-ins were drawn. PCA's candidates are the first k of its
    ordered components, for every declared k the scores are wide enough for; NMF's are the
    ``nmf_{k}`` matrices the cache holds.
    """
    n_lines = len(grid.lines)
    with np.load(cache / "descriptions.npz") as npz:
        stored = {key: npz[key] for key in npz.files}
    lines = tuple(str(v) for v in stored["lines"])
    if lines != grid.lines:
        raise ValueError("descriptions.npz's lines are not the grid's lines in grid order")

    pca = stored["pca"]
    pca_ks = {k: pca[:, :k] for k in COMPONENT_KS if k <= pca.shape[1]}
    if not pca_ks:
        raise ValueError(
            f"the PCA description is {pca.shape[1]} columns wide, narrower than the smallest "
            f"candidate component count {min(COMPONENT_KS)}"
        )
    nmf_ks = {int(key[len("nmf_") :]): stored[key] for key in stored if key.startswith("nmf_")}
    if not nmf_ks:
        raise ValueError("descriptions.npz holds no nmf_{k} arrays")
    components = {"pca": pca_ks, "nmf": nmf_ks}

    if (len(grid.lines), len(grid.drugs)) == (FULL_GRID_LINES, FULL_GRID_DRUGS):
        narrowed = {
            name: tuple(sorted(family))
            for name, family in components.items()
            if tuple(sorted(family)) != COMPONENT_KS
        }
        if narrowed:
            raise ValueError(
                f"on the design's {FULL_GRID_LINES} x {FULL_GRID_DRUGS} grid every tuned "
                f"description must offer the candidate component counts {COMPONENT_KS}; got "
                f"{narrowed} (a cache written with fewer nmf_{{k}} arrays, or a narrower PCA, "
                "would silently run a model the design does not declare)"
            )

    descriptions: dict[str, np.ndarray] = {
        "expression": stored["expression"],
        "pca": pca_ks[max(pca_ks)],
        "nmf": nmf_ks[max(nmf_ks)],
    }
    for version in STACK_VERSIONS:
        descriptions[f"stack_{version}"] = _stack_embedding(cache, version, grid)

    misshapen = {
        name: matrix.shape
        for name, matrix in descriptions.items()
        if matrix.ndim != 2 or matrix.shape[0] != n_lines or not np.isfinite(matrix).all()
    }
    if misshapen:
        raise ValueError(f"descriptions are not finite [{n_lines}, width] arrays: {misshapen}")
    return descriptions, components


def _tanimoto(cache: Path, grid: Grid) -> np.ndarray:
    """The drugs' chemical similarity, checked to be the grid's drugs in grid order."""
    with np.load(cache / "tanimoto.npz") as npz:
        drugs = tuple(str(v) for v in npz["drugs"])
        similarity = npz["similarity"]
    if drugs != grid.drugs:
        raise ValueError("tanimoto.npz's drugs are not the grid's drugs in grid order")
    if similarity.shape != (len(drugs), len(drugs)):
        raise ValueError(
            f"tanimoto.npz's similarity must be [{len(drugs)}, {len(drugs)}]; "
            f"got shape {similarity.shape}"
        )
    return similarity


def load_inputs(grid: Grid, cache: Path) -> RoundInputs:
    """Read the cache once: answers, scoreable masks, every model's own matrices, the
    nearest-lines similarity and the drugs' chemical similarity.

    Descriptions hold no drug response (invariant 3), so each kernel is built over all 50 lines
    and reveals nothing about a hidden answer; the same matrices serve every round.

    Each model gets its entry under its **own id**. A description whose component count is tuned
    (PCA, NMF) gets its candidate matrices; **its random stand-in gets the same candidate counts**
    (ruling 35), as the first k columns of the stand-in the one helper draws -- design section 5
    puts a stand-in through the same fit as the description, and for these two that fit includes
    choosing k. The stand-in's columns are exchangeable draws from one distribution, so its first
    k are a k-wide draw from it; no second seed source is introduced.
    """
    answers = _load_answers(cache, grid)
    masks = scoreable(answers, grid.excluded_pairs, MIN_GENES)
    descriptions, candidates = _descriptions(cache, grid)
    n_lines = len(grid.lines)

    kernels: dict[str, np.ndarray] = {}
    components: dict[str, dict[int, np.ndarray]] = {}
    for spec in MODELS.values():
        if spec.description is None or spec.kind == "nearest_lines":
            continue
        described = descriptions[spec.description]
        family = candidates.get(spec.description)
        if spec.kind != "random":
            if family is None:
                kernels[spec.id] = linear_kernel(described)
            else:
                components[spec.id] = family
            continue
        matrix = stand_in(spec.description, n_lines, described.shape[1])
        if family is None:
            kernels[spec.id] = linear_kernel(matrix)
            continue
        widest = max(family)
        if widest > matrix.shape[1]:
            raise ValueError(
                f"the stand-in for {spec.description} is {matrix.shape[1]} columns wide, "
                f"narrower than its widest candidate component count {widest}"
            )
        components[spec.id] = {k: matrix[:, :k] for k in sorted(family)}

    return RoundInputs(
        grid=grid,
        answers=answers,
        masks=masks,
        delta0=np.nan_to_num(answers.delta, nan=0.0),
        tested=np.isfinite(answers.delta),
        kernels=kernels,
        components=components,
        similarity=similarity_from_description(descriptions["expression"]),
        tanimoto=_tanimoto(cache, grid),
    )


def realized_component_ks(inputs: RoundInputs) -> dict[str, list[int]]:
    """The candidate component counts each tuned model actually chose between, by model id.

    Recorded beside the round's settings so task 12's battery can check the run tuned over the
    design's declared counts, and not over a set a narrowed cache quietly handed it.
    """
    return {model: sorted(family) for model, family in sorted(inputs.components.items())}


def _reference_fit(prediction: np.ndarray) -> Fit:
    """A model that tunes nothing (the drug average) as a ``Fit``: no penalty, no component
    count, and no tuning loss to report."""
    return Fit(prediction=prediction, lam=None, k=None, loss_min=float("nan"), at_edge=False)


def _kernel(spec: ModelSpec, inputs: RoundInputs) -> np.ndarray:
    """The kernel built for this model id. Raises rather than letting a model with no matrices of
    its own fall through to another model's."""
    kernel = inputs.kernels.get(spec.id)
    if kernel is None:
        raise KeyError(f"no kernel or candidate components were built for model {spec.id!r}")
    return kernel


def _fit_lolo(spec: ModelSpec, index: int, inputs: RoundInputs) -> Fit:
    """One model's fit with line ``index`` hidden.

    Both ridge branches look their matrices up by **model id**, so a random stand-in is fitted on
    its own stand-in -- over the same candidate component counts as the description it stands in
    for, when that description tunes one (ruling 35) -- and never on the description itself.
    """
    if spec.kind == "reference":
        n_lines = len(inputs.grid.lines)
        train = np.flatnonzero(np.arange(n_lines) != index)
        return _reference_fit(drug_average(inputs.delta0, train))
    if spec.kind == "nearest_lines":
        return nearest_lines_lolo(inputs.similarity, inputs.delta0, inputs.tested, index)
    family = inputs.components.get(spec.id)
    if family is not None:
        return ridge_lolo_k(family, inputs.delta0, inputs.tested, index)
    return ridge_lolo(_kernel(spec, inputs), inputs.delta0, inputs.tested, index)


def _fit_lodo(spec: ModelSpec, index: int, inputs: RoundInputs) -> Fit:
    """One model's fit with drug ``index`` hidden. Chemistry only is the same ridge regression
    with an all-zero line kernel: chemically similar drugs share effects, whatever the line.

    As in ``_fit_lolo``, both ridge branches look up by model id, so a stand-in is fitted on its
    own matrices.
    """
    n_lines = len(inputs.grid.lines)
    if spec.kind == "reference":
        no_line_term = np.zeros((n_lines, n_lines))
        return ridge_lodo(no_line_term, inputs.tanimoto, inputs.delta0, inputs.tested, index)
    family = inputs.components.get(spec.id)
    if family is not None:
        return ridge_lodo_k(family, inputs.tanimoto, inputs.delta0, inputs.tested, index)
    return ridge_lodo(_kernel(spec, inputs), inputs.tanimoto, inputs.delta0, inputs.tested, index)


def _round_pairs(scheme: Scheme, index: int, inputs: RoundInputs) -> tuple[np.ndarray, np.ndarray]:
    """The (line, drug) indices a round predicts, in prediction row order: the hidden line's
    drugs, or the hidden drug's lines."""
    n_lines, n_drugs = len(inputs.grid.lines), len(inputs.grid.drugs)
    if scheme == "lolo":
        return np.full(n_drugs, index, dtype=np.int64), np.arange(n_drugs, dtype=np.int64)
    return np.arange(n_lines, dtype=np.int64), np.full(n_lines, index, dtype=np.int64)


def _settings_row(spec: ModelSpec, fit: Fit) -> dict[str, object]:
    return {
        "model": spec.id,
        "lambda": float("nan") if fit.lam is None else float(fit.lam),
        "k": float("nan") if fit.k is None else float(fit.k),
        "loss_min": float(fit.loss_min),
        "lambda_at_edge": bool(fit.at_edge),
    }


def run_round(scheme: Scheme, index: int, inputs: RoundInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score every model declared for ``scheme`` with unit ``index`` hidden.

    Returns ``(scores, settings)``: one row per (model, scored pair, gene set) with the columns
    of ``SCORE_FILE_COLUMNS``, and one row per model with the columns of ``SETTINGS_COLUMNS``,
    models in declaration order. ``score_pairs`` is called once per model, on this round's pairs
    alone -- the hidden line's drugs, or the hidden drug's lines -- so no call ever holds more
    than 107 pairs' predictions and answers.
    """
    n_units = len(inputs.grid.lines) if scheme == "lolo" else len(inputs.grid.drugs)
    if not 0 <= index < n_units:
        raise ValueError(f"round {index} is outside 0..{n_units - 1} for scheme {scheme}")

    line_idx, drug_idx = _round_pairs(scheme, index, inputs)
    score_frames: list[pd.DataFrame] = []
    settings_rows: list[dict[str, object]] = []
    for spec in models_for(scheme):
        fit = _fit_lolo(spec, index, inputs) if scheme == "lolo" else _fit_lodo(spec, index, inputs)
        scores = score_pairs(fit.prediction, line_idx, drug_idx, inputs.answers, inputs.masks)
        scores.insert(0, "model", spec.id)
        scores.insert(0, "round", index)
        scores.insert(0, "scheme", scheme)
        score_frames.append(scores)
        settings_rows.append(_settings_row(spec, fit))

    scores = pd.concat(score_frames, ignore_index=True)
    settings = pd.DataFrame(settings_rows, columns=list(SETTINGS_COLUMNS))
    if tuple(scores.columns) != SCORE_FILE_COLUMNS:
        raise ValueError(f"scores columns {tuple(scores.columns)} are not {SCORE_FILE_COLUMNS}")
    return scores, settings


def round_paths(cache: Path, scheme: Scheme, index: int) -> tuple[Path, Path]:
    """The round's two output paths: its per-pair scores and its chosen settings."""
    return (
        cache / f"scores_{scheme}_{index:03d}.parquet",
        cache / f"settings_{scheme}_{index:03d}.csv",
    )


def write_round(
    cache: Path,
    scheme: Scheme,
    index: int,
    scores: pd.DataFrame,
    settings: pd.DataFrame,
    component_ks: Mapping[str, Sequence[int]],
) -> tuple[Path, Path]:
    """Write the round's two files and their completion records.

    ``component_ks`` (from ``realized_component_ks``) is recorded beside the settings, so the
    candidate component counts the round actually tuned over are pinned with its bytes -- the
    chosen k in the settings table is only interpretable against the set it was chosen from.
    """
    cache.mkdir(parents=True, exist_ok=True)
    scores_path, settings_path = round_paths(cache, scheme, index)
    record = {"scheme": scheme, "round": index}
    scores.to_parquet(scores_path, index=False)
    write_record(scores_path, {**record, "rows": len(scores)})
    settings.to_csv(settings_path, index=False)
    write_record(
        settings_path,
        {
            **record,
            "models": len(settings),
            "component_ks": {model: list(ks) for model, ks in component_ks.items()},
        },
    )
    return scores_path, settings_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scheme", required=True, choices=("lolo", "lodo"))
    ap.add_argument("--round", dest="round_index", type=int, required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()

    scheme = cast(Scheme, args.scheme)
    index = int(args.round_index)
    grid = load_grid(args.grid)
    n_units = len(grid.lines) if scheme == "lolo" else len(grid.drugs)
    if not 0 <= index < n_units:
        raise SystemExit(f"--round {index} is outside 0..{n_units - 1} for --scheme {scheme}")

    args.cache.mkdir(parents=True, exist_ok=True)
    scores_path, settings_path = round_paths(args.cache, scheme, index)
    if is_done(scores_path) and is_done(settings_path):
        print(f"{scores_path} and {settings_path} already done, skipping")
        return

    inputs = load_inputs(grid, args.cache)
    scores, settings = run_round(scheme, index, inputs)
    write_round(args.cache, scheme, index, scores, settings, realized_component_ks(inputs))
    print(f"wrote {scores_path} ({len(scores)} rows)")
    print(f"wrote {settings_path} ({len(settings)} models)")


if __name__ == "__main__":
    main()
