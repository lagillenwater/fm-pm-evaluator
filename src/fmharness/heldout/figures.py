"""Rung 1's figures: one per design.md section 8 bullet, each drawn from a table the run wrote.

Rung 1 asks whether a model can predict the expression change of a cell line, or a drug, it has
never seen. These five figures are how a reader checks each step of that measurement rather than
taking it on trust: how the lines were described (build), how a line or a drug was hidden
(split), what the models chose and recovered (fit), what they scored against the ceiling (score),
and what the head-to-head comparisons are read against (null).

Two rules from the task design hold for every function here, exactly as they do for rung 0's
``fmharness.figures``.

**A figure is drawn from a committed table.** Each function takes already-written tables and an
output path, and reads nothing else -- no cache, no answer array, no statistic recomputed from
the screen. Every number a panel draws came from a table that ships beside the figure, so a
reader can recompute it.

**A figure that has a control shows it beside the real data.** Real data alone says what the run
produced; real data beside a planted answer says whether the machinery read it correctly. Each
figure pairs its real panel with the matching control table's panel, and names which is which.

Every function returns the path it wrote and tolerates a thin or empty table without raising:
the figure step runs at the end of a long cluster job, and a crash there throws the run away.

Colour is used the same way throughout: one fixed hue per gene set (never cycled), real data in
blue against its control in orange, and a single-hue sequential ramp for the identity grids, both
panels of a pair sharing one scale so they can be read against each other.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# The non-interactive backend is chosen BEFORE pyplot is imported, so a figure renders
# identically on a headless cluster node and on a laptop.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

#: Real data, its control, and one fixed hue per gene set -- assigned once, never cycled.
_REAL_COLOR = "tab:blue"
_CONTROL_COLOR = "tab:orange"
_GENE_SET_COLOR = {"responding": "tab:red", "all": "tab:blue"}
_SEQUENTIAL = "Blues"

#: The two schemes, in the order every panel lays them out.
_SCHEMES: tuple[str, ...] = ("lolo", "lodo")
_SCHEME_LABEL = {"lolo": "leave one line out", "lodo": "leave one drug out"}


# --------------------------------------------------------------------------------------------
# Reading a table. Every column any figure draws arrives through one of these accessors, as
# plain numpy, which keeps the drawing code vectorised and pandas' untyped surface out of the
# type checker's way. Mirrors rung 0's fmharness.figures, whose house style these follow.
# --------------------------------------------------------------------------------------------


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """One column as floats, aligned row-for-row; all-``nan`` when the column is absent."""
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    coerced = cast(Any, pd.to_numeric(cast(Any, frame)[column], errors="coerce"))
    return np.asarray(coerced.to_numpy(dtype=float), dtype=float)


def _labels(frame: pd.DataFrame, column: str) -> np.ndarray:
    """One column as strings, aligned row-for-row; empty strings when the column is absent."""
    if column not in frame.columns:
        return np.array([""] * len(frame), dtype=str)
    raw = cast(Any, frame)[column].tolist()
    return np.array([str(value) for value in raw], dtype=str)


def _flags(frame: pd.DataFrame, column: str) -> np.ndarray:
    """One column as booleans, aligned row-for-row; all false when the column is absent."""
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    filled = cast(Any, frame)[column].fillna(False)
    return np.asarray(filled.to_numpy(dtype=bool), dtype=bool)


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    """The finite values of one column, or an empty array when the column is absent."""
    values = _numeric(frame, column)
    return values[np.isfinite(values)]


def _ordered_unique(values: np.ndarray) -> list[str]:
    """Distinct labels in the order they first appear, so panels keep the table's own order."""
    return [str(value) for value in dict.fromkeys(values.tolist())]


def _note_empty(ax: Axes, message: str) -> None:
    """Say plainly that a panel had nothing to draw, rather than showing bare axes."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8, transform=ax.transAxes)


def _legend(ax: Axes, *, fontsize: int = 7, ncol: int = 1) -> None:
    """Draw a legend only when something is labelled, so an empty panel stays quiet."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=fontsize, ncol=ncol)


def _legend_from(ax: Axes, handles: Sequence[Any], *, fontsize: int = 7) -> None:
    """Draw a legend from explicit proxy artists, or nothing when there are none.

    Where one call draws bars of several colours, the legend has to be built from real ``Patch``
    and ``Line2D`` artists. The obvious shortcut -- a second ``ax.bar([], [])`` carrying only a
    label -- makes a container with no patch in it, and matplotlib then falls back to a default
    swatch: every entry comes out the same colour and the legend stops matching the marks it
    explains (seen on the fit and split panels before this was fixed).
    """
    if handles:
        ax.legend(handles=list(handles), frameon=False, fontsize=fontsize)


def _split_control_handles() -> list[Any]:
    """The split control panel's legend: the broken split, the shipped split, and the MDE mark."""
    return [
        Patch(facecolor=_CONTROL_COLOR, label="broken split (hidden unit kept in)"),
        Patch(facecolor=_REAL_COLOR, label="shipped split"),
        Line2D([], [], color="k", lw=1.2, linestyle="--", label="minimum detectable effect"),
    ]


def _finish(fig: Figure, out: Path) -> Path:
    """Write the figure and return its path, which is what every public function returns."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _error_bars(low: np.ndarray, high: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Interval ends as matplotlib's (below, above) offsets from the estimate, never negative."""
    below = np.clip(np.nan_to_num(centre - low, nan=0.0), 0.0, None)
    above = np.clip(np.nan_to_num(high - centre, nan=0.0), 0.0, None)
    return np.vstack([below, above])


# --------------------------------------------------------------------------------------------
# build -- the line descriptions the models are given
# --------------------------------------------------------------------------------------------


def _identity_matrix(grid: pd.DataFrame, source: str) -> tuple[np.ndarray, list[str]]:
    """One source's correlation grid as a square array, with its line order.

    The table is long (``line_a, line_b, r, source``); the lines keep the order they first
    appear in, so the diagonal is the match a description should make with its own other half.
    """
    chosen = cast(Any, grid).loc[_labels(grid, "source") == source]
    if len(chosen) == 0:
        return np.empty((0, 0)), []
    lines = _ordered_unique(_labels(chosen, "line_a"))
    position = {line: i for i, line in enumerate(lines)}
    matrix = np.full((len(lines), len(lines)), np.nan)
    rows = np.array([position.get(line, -1) for line in _labels(chosen, "line_a")])
    columns = np.array([position.get(line, -1) for line in _labels(chosen, "line_b")])
    values = _numeric(chosen, "r")
    keep = (rows >= 0) & (columns >= 0)
    matrix[rows[keep], columns[keep]] = values[keep]
    return matrix, lines


def fig_build(
    cells: pd.DataFrame,
    identity_match: pd.DataFrame,
    identity_grids: Mapping[str, pd.DataFrame],
    weights_check: Mapping[str, object],
    out: Path,
) -> Path:
    """What every model is told about a cell line, and whether that description identifies it.

    Panel (a) is the material the descriptions are built from: how many untreated (DMSO) cells
    each line contributed, and over how many plates. A line described from few cells, or from one
    plate, carries more of that plate's batch than of the line.

    Panel (b) is the build control as a number. A description built from half a line's cells
    should match that line's OTHER half better than it matches any other line; the bar is the
    share of lines where it does, and the marker beside it is the 99th percentile of the same
    statistic with the cells' line labels shuffled. A bar that does not clear its marker is a
    description that does not identify the line it came from. The drug fine-tune's weights check
    is printed here too: if its encoder were byte-identical to the base checkpoint, the two Stack
    versions would be one model under two names and H2 would be untestable.

    The lower two rows are the same control as a picture, one column per description: the real
    half-versus-half correlation grid above, and the shuffled grid below it on the SAME colour
    scale. The real grid should have a bright diagonal -- each line matching itself -- and the
    shuffled grid should have none.
    """
    descriptions = list(identity_grids)
    n_columns = max(1, len(descriptions))
    fig = plt.figure(figsize=(max(11.0, 2.6 * n_columns), 11.0), layout="constrained")
    grid = fig.add_gridspec(3, n_columns)

    ax_cells = fig.add_subplot(grid[0, : max(1, n_columns // 2)])
    lines = _labels(cells, "line")
    selected = _numeric(cells, "n_selected")
    seen = _numeric(cells, "n_dmso_seen")
    if len(cells) and np.isfinite(selected).any():
        order = _ordered_unique(lines)
        position = {line: i for i, line in enumerate(order)}
        x = np.array([position.get(line, -1) for line in lines], dtype=float)
        per_line_selected = np.array(
            [np.nansum(selected[lines == line]) for line in order], dtype=float
        )
        per_line_seen = np.array([np.nansum(seen[lines == line]) for line in order], dtype=float)
        ax_cells.bar(
            np.arange(len(order)),
            per_line_seen,
            color=_CONTROL_COLOR,
            alpha=0.45,
            label="DMSO cells seen",
        )
        ax_cells.bar(
            np.arange(len(order)),
            per_line_selected,
            color=_REAL_COLOR,
            alpha=0.9,
            label="cells selected for the description",
        )
        drawn = x >= 0
        ax_cells.scatter(
            x[drawn],
            selected[drawn],
            s=10,
            color="k",
            zorder=3,
            label="one plate's contribution",
        )
        ax_cells.set_xticks(np.arange(len(order)))
        if len(order) <= 20:
            ax_cells.set_xticklabels(order, rotation=90, fontsize=6)
        else:
            ax_cells.set_xticklabels([])
    else:
        _note_empty(ax_cells, "no cells table")
    ax_cells.set_xlabel("cell line")
    ax_cells.set_ylabel("cells (count)")
    ax_cells.set_title("(a) cells per line and plate", fontsize=9)
    _legend(ax_cells, fontsize=7)

    ax_match = fig.add_subplot(grid[0, max(1, n_columns // 2) :])
    subjects = _labels(identity_match, "description")
    share = _numeric(identity_match, "identity_share")
    null_p99 = _numeric(identity_match, "null_p99")
    null_mean = _numeric(identity_match, "null_mean")
    if len(identity_match):
        positions = np.arange(len(subjects), dtype=float)
        ax_match.bar(
            positions, share, color=_REAL_COLOR, alpha=0.9, label="real halves: identity match"
        )
        ax_match.scatter(
            positions,
            null_p99,
            marker="_",
            s=200,
            color=_CONTROL_COLOR,
            linewidths=2.0,
            zorder=3,
            label="shuffled labels: 99th percentile",
        )
        ax_match.scatter(
            positions, null_mean, marker="x", s=30, color="k", zorder=3, label="shuffled mean"
        )
        ax_match.set_xticks(positions)
        ax_match.set_xticklabels([str(name) for name in subjects], rotation=30, fontsize=7)
        ax_match.set_ylim(0.0, 1.05)
    else:
        _note_empty(ax_match, "no identity-match table")
    identical = bool(weights_check.get("identical_all", False))
    n_different = weights_check.get("n_different", "?")
    ax_match.set_xlabel("line description")
    ax_match.set_ylabel("share of lines matching their own other half")
    ax_match.set_title(
        "(b) build control: each description against its shuffled null\n"
        f"drug fine-tune encoder differs from the base: {not identical} "
        f"({n_different} tensors differ)",
        fontsize=9,
    )
    _legend(ax_match, fontsize=7)

    for column, description in enumerate(descriptions):
        real, order = _identity_matrix(identity_grids[description], "real")
        shuffled, _ = _identity_matrix(identity_grids[description], "shuffled")
        both = [m for m in (real, shuffled) if m.size]
        if both:
            stacked = np.concatenate([m[np.isfinite(m)].ravel() for m in both])
            low = float(np.min(stacked)) if stacked.size else 0.0
            high = float(np.max(stacked)) if stacked.size else 1.0
        else:
            low, high = 0.0, 1.0
        for row, (matrix, source) in enumerate(((real, "real"), (shuffled, "shuffled"))):
            ax = fig.add_subplot(grid[1 + row, column])
            if matrix.size:
                ax.imshow(matrix, cmap=_SEQUENTIAL, vmin=low, vmax=high, aspect="auto")
            else:
                _note_empty(ax, "no grid")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("second half (lines)" if row else "")
            ax.set_ylabel("first half (lines)" if column == 0 else "")
            ax.set_title(
                f"{description} -- {source}" + (f" ({len(order)} lines)" if row == 0 else ""),
                fontsize=8,
            )

    fig.suptitle(
        "build: the line descriptions, and whether a description identifies its own line",
        fontsize=11,
    )
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------
# split -- hiding a line or a drug
# --------------------------------------------------------------------------------------------


def fig_split(
    pair_scores: pd.DataFrame,
    split_control: pd.DataFrame,
    lines: Sequence[str],
    drugs: Sequence[str],
    out: Path,
) -> Path:
    """Which pairs the rounds actually scored, and whether the split holds.

    Panel (a) is the whole grid of rounds: one cell per (line, drug) pair, shaded by how many of
    the two schemes scored it. Every pair should be scored under both -- once when its line was
    hidden, once when its drug was hidden -- so a pair that is not is either short of the
    50-gene minimum or one of the leaked pairs removed for every model. The panel is drawn from
    the run's own pair-score table, so the gaps a reader sees are the gaps the comparisons were
    taken over.

    Panel (b) is the split control. A random gene signature is planted in one unit's OWN answers
    and nowhere else, and two fits predict that unit: a deliberately broken split that leaves the
    hidden unit in training, and the shipped split. The broken fit recovers the signature -- that
    is what a leak looks like -- and the shipped fit must sit at zero within its own minimum
    detectable effect. The bars are those correlations with their intervals; the dashed marks are
    the MDE, the smallest effect this control could have seen.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), layout="constrained", width_ratios=[3, 2])
    ax_grid, ax_control = axes[0], axes[1]

    line_position = {line: i for i, line in enumerate(lines)}
    drug_position = {drug: j for j, drug in enumerate(drugs)}
    covered = np.zeros((max(1, len(lines)), max(1, len(drugs))), dtype=float)
    if len(pair_scores):
        scored_lines = _labels(pair_scores, "line")
        scored_drugs = _labels(pair_scores, "drug")
        schemes = _labels(pair_scores, "scheme")
        for scheme in _SCHEMES:
            chosen = schemes == scheme
            rows = np.array([line_position.get(line, -1) for line in scored_lines[chosen]])
            columns = np.array([drug_position.get(drug, -1) for drug in scored_drugs[chosen]])
            keep = (rows >= 0) & (columns >= 0)
            seen = np.zeros(covered.shape, dtype=bool)
            seen[rows[keep], columns[keep]] = True
            covered += seen
    image = ax_grid.imshow(covered, cmap=_SEQUENTIAL, vmin=0.0, vmax=2.0, aspect="auto")
    bar = fig.colorbar(image, ax=ax_grid, ticks=[0, 1, 2], fraction=0.046)
    bar.ax.set_yticklabels(["scored in neither", "one scheme", "both schemes"], fontsize=7)
    ax_grid.set_xlabel(f"drug ({len(drugs)} in the grid)")
    ax_grid.set_ylabel(f"cell line ({len(lines)} in the grid)")
    ax_grid.set_title(
        f"(a) the {len(lines)} x {len(drugs)} grid of rounds: which pairs each scheme scored",
        fontsize=9,
    )

    if len(split_control):
        schemes = _labels(split_control, "scheme")
        splits = _labels(split_control, "split")
        estimate = _numeric(split_control, "estimate")
        mde = _numeric(split_control, "mde")
        yerr = _error_bars(
            _numeric(split_control, "ci_lo"), _numeric(split_control, "ci_hi"), estimate
        )
        labels: list[str] = []
        positions: list[float] = []
        colors: list[str] = []
        for index in range(len(split_control)):
            positions.append(float(index))
            labels.append(f"{_SCHEME_LABEL.get(schemes[index], schemes[index])}\n{splits[index]}")
            colors.append(_CONTROL_COLOR if splits[index] == "leaky" else _REAL_COLOR)
        ax_control.bar(positions, estimate, color=colors, alpha=0.9)
        ax_control.errorbar(
            positions, estimate, yerr=yerr, fmt="none", ecolor="k", capsize=4, lw=1.2
        )
        for position, value in zip(positions, mde, strict=True):
            if np.isfinite(value):
                ax_control.plot(
                    [position - 0.4, position + 0.4],
                    [value, value],
                    color="k",
                    lw=1.2,
                    linestyle="--",
                )
        ax_control.set_xticks(positions)
        ax_control.set_xticklabels(labels, fontsize=7)
        ax_control.axhline(0.0, color="k", lw=0.8)
    else:
        _note_empty(ax_control, "no split control table")
    ax_control.set_ylabel("correlation with the hidden unit's planted signature")
    ax_control.set_title(
        "(b) split control: the planted signal under the broken\nand the shipped split", fontsize=9
    )
    _legend_from(ax_control, _split_control_handles() if len(split_control) else [], fontsize=7)

    fig.suptitle(
        "split: what each round scored, and whether the hidden unit stayed hidden", fontsize=11
    )
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------
# fit -- what the models chose, and what they recovered
# --------------------------------------------------------------------------------------------


def fig_fit(
    settings: pd.DataFrame, model_summary: pd.DataFrame, fit_control: pd.DataFrame, out: Path
) -> Path:
    """The settings each round chose, and the gain a planted answer produced beside the real one.

    Panel (a) is every round's chosen ridge penalty, one column per model. The penalty is chosen
    inside the round from training data alone, so its spread across rounds is a property of the
    data rather than a setting anyone picked. A model pinned at the top of the grid in every
    round is one whose description bought it nothing: the tuning is shrinking it to the round's
    reference. Panel (b) is the same for the number of PCA or NMF components, which is chosen the
    same way -- for the descriptions and, per ruling 35, for their random stand-ins too.

    Panel (c) is the fit control beside the real scores. On the left, a synthetic grid of the
    screen's own size with a line- or drug-specific response planted in it: the oracle that knows
    the true change, the matching description, and a random stand-in of the same width, each as a
    gain over the round's reference with its interval. The description must find the plant and
    the stand-in must not. On the right, each model's mean score on the real data. The two panels
    answer different questions -- can this machinery find a known answer, and what did it find in
    the screen -- and they are drawn together because neither is worth much alone.
    """
    fig = plt.figure(figsize=(15.0, 9.0), layout="constrained")
    grid = fig.add_gridspec(2, 2)

    ax_lambda = fig.add_subplot(grid[0, 0])
    models = _labels(settings, "model")
    lambdas = _numeric(settings, "lambda")
    order = _ordered_unique(models)
    if order and np.isfinite(lambdas).any():
        rng = np.random.default_rng(0)
        for index, model in enumerate(order):
            chosen = (models == model) & np.isfinite(lambdas)
            if not chosen.any():
                continue
            jitter = rng.uniform(-0.22, 0.22, size=int(chosen.sum()))
            ax_lambda.scatter(
                index + jitter,
                lambdas[chosen],
                s=9,
                color=_REAL_COLOR,
                alpha=0.55,
                edgecolors="none",
            )
        ax_lambda.set_yscale("log")
        ax_lambda.set_xticks(np.arange(len(order)))
        ax_lambda.set_xticklabels(order, rotation=90, fontsize=6)
    else:
        _note_empty(ax_lambda, "no chosen penalties")
    ax_lambda.set_xlabel("model")
    ax_lambda.set_ylabel("ridge penalty chosen in the round")
    ax_lambda.set_title("(a) the penalty each round chose, per model", fontsize=9)

    ax_k = fig.add_subplot(grid[0, 1])
    ks = _numeric(settings, "k")
    with_k = [model for model in order if np.isfinite(ks[models == model]).any()]
    if with_k:
        for index, model in enumerate(with_k):
            chosen = (models == model) & np.isfinite(ks)
            values, counts = np.unique(ks[chosen], return_counts=True)
            ax_k.scatter(
                np.full(values.size, index),
                values,
                s=12 + 3 * counts,
                color=_REAL_COLOR,
                alpha=0.7,
                edgecolors="none",
            )
        ax_k.set_xticks(np.arange(len(with_k)))
        ax_k.set_xticklabels(with_k, rotation=90, fontsize=6)
    else:
        _note_empty(ax_k, "no model chose a component count")
    ax_k.set_xlabel("model")
    ax_k.set_ylabel("components (or neighbours) chosen")
    ax_k.set_title("(b) the component count each round chose (marker size = rounds)", fontsize=9)

    ax_control = fig.add_subplot(grid[1, 0])
    planted = cast(Any, fit_control).loc[_flags(fit_control, "planted")]
    wanted = ("oracle", "description", "random")
    rows = planted.loc[np.isin(_labels(planted, "contrast"), wanted)]
    if len(rows):
        contrasts = _labels(rows, "contrast")
        schemes = _labels(rows, "scheme")
        estimate = _numeric(rows, "estimate")
        yerr = _error_bars(_numeric(rows, "ci_lo"), _numeric(rows, "ci_hi"), estimate)
        positions = np.arange(len(rows), dtype=float)
        colors = [
            _REAL_COLOR if contrast == "description" else _CONTROL_COLOR for contrast in contrasts
        ]
        ax_control.bar(positions, estimate, color=colors, alpha=0.9)
        ax_control.errorbar(
            positions, estimate, yerr=yerr, fmt="none", ecolor="k", capsize=4, lw=1.2
        )
        ax_control.set_xticks(positions)
        ax_control.set_xticklabels(
            [
                f"{_SCHEME_LABEL.get(scheme, scheme)}\n{contrast}"
                for scheme, contrast in zip(schemes, contrasts, strict=True)
            ],
            fontsize=6,
        )
        ax_control.axhline(0.0, color="k", lw=0.8)
    else:
        _note_empty(ax_control, "no fit control table")
    ax_control.set_ylabel("gain over the scheme's reference (mean r)")
    ax_control.set_title(
        "(c) fit control: the planted gain found, by the oracle, the\nmatching description, and a"
        " random stand-in",
        fontsize=9,
    )

    ax_real = fig.add_subplot(grid[1, 1])
    responding = cast(Any, model_summary).loc[_labels(model_summary, "gene_set") == "responding"]
    if len(responding):
        names = _labels(responding, "model")
        schemes = _labels(responding, "scheme")
        mean_r = _numeric(responding, "mean_r")
        yerr = _error_bars(_numeric(responding, "ci_lo"), _numeric(responding, "ci_hi"), mean_r)
        positions = np.arange(len(responding), dtype=float)
        colors = [_REAL_COLOR if scheme == "lolo" else _CONTROL_COLOR for scheme in schemes]
        ax_real.bar(positions, mean_r, color=colors, alpha=0.9)
        ax_real.errorbar(positions, mean_r, yerr=yerr, fmt="none", ecolor="k", capsize=3, lw=1.0)
        ax_real.set_xticks(positions)
        ax_real.set_xticklabels(
            [f"{scheme}: {name}" for scheme, name in zip(schemes, names, strict=True)],
            rotation=90,
            fontsize=6,
        )
        ax_real.axhline(0.0, color="k", lw=0.8)
    else:
        _note_empty(ax_real, "no model summary")
    ax_real.set_ylabel("mean score on responding genes (r)")
    ax_real.set_title("(d) each model's real score, both schemes", fontsize=9)
    _legend_from(
        ax_real,
        [
            Patch(facecolor=_REAL_COLOR, label=_SCHEME_LABEL["lolo"]),
            Patch(facecolor=_CONTROL_COLOR, label=_SCHEME_LABEL["lodo"]),
        ]
        if len(responding)
        else [],
        fontsize=7,
    )

    fig.suptitle(
        "fit: the settings chosen per round, and the planted gain beside the real", fontsize=11
    )
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------
# score -- the scores against the ceiling
# --------------------------------------------------------------------------------------------


def fig_score(
    model_summary: pd.DataFrame, ceiling: pd.DataFrame, score_control: pd.DataFrame, out: Path
) -> Path:
    """Every model's score against the best score anything could reach, and the test of that bar.

    One panel per scheme: each model's mean score over the pairs every model scored, with its
    confidence interval, on both gene sets. Two horizontal lines per gene set say what the score
    is read against -- the ceiling √SB, the score a perfect prediction would reach against a
    measurement that carries noise once, and rung 0's own split-half r, the agreement of two such
    measurements. A model sitting near the ceiling has found nearly everything reproducible in
    the answer; a model near zero has found nothing.

    The last panel is the score control, and it is what makes the ceiling more than an assertion.
    Synthetic answers are built at a KNOWN reliability R, and the true change scores √R against
    them while a second noisy measurement of the same truth scores R. The bars are what the real
    scoring code returned; the marks are the planted values it had to reproduce. An unrelated
    prediction scores zero.
    """
    schemes = [scheme for scheme in _SCHEMES if scheme in set(_labels(model_summary, "scheme"))]
    n_panels = max(1, len(schemes)) + 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.0 * n_panels, 5.6), layout="constrained", squeeze=False
    )

    ceiling_by_set = {
        str(row): (value, rung0)
        for row, value, rung0 in zip(
            _labels(ceiling, "gene_set"),
            _numeric(ceiling, "sqrt_sb"),
            _numeric(ceiling, "split_half_r"),
            strict=True,
        )
    }

    for position, scheme in enumerate(schemes or ["lolo"]):
        ax = axes[0][position]
        rows = cast(Any, model_summary).loc[_labels(model_summary, "scheme") == scheme]
        models = _ordered_unique(_labels(rows, "model"))
        if not models:
            _note_empty(ax, f"no models scored under {scheme}")
        width = 0.38
        headroom: list[float] = []
        for offset, gene_set in zip((-0.19, 0.19), ("responding", "all"), strict=True):
            chosen = rows.loc[_labels(rows, "gene_set") == gene_set]
            if not len(chosen):
                continue
            names = _labels(chosen, "model")
            x = np.array([models.index(name) for name in names], dtype=float) + offset
            mean_r = _numeric(chosen, "mean_r")
            yerr = _error_bars(_numeric(chosen, "ci_lo"), _numeric(chosen, "ci_hi"), mean_r)
            ax.bar(
                x,
                mean_r,
                width=width,
                color=_GENE_SET_COLOR[gene_set],
                alpha=0.85,
                label=f"{gene_set} genes",
            )
            ax.errorbar(x, mean_r, yerr=yerr, fmt="none", ecolor="k", capsize=3, lw=1.0)
            # The reference lines stay legend entries -- naming them on the lines themselves put
            # the text over the bars, since two of the four sit BELOW the bar tops. The legend is
            # kept clear of them by the headroom set after this loop instead.
            headroom.extend(_numeric(chosen, "ci_hi").tolist())
            sqrt_sb, rung0_r = ceiling_by_set.get(gene_set, (np.nan, np.nan))
            for value, style, line_width, text in (
                (sqrt_sb, "--", 1.4, f"{gene_set}: ceiling sqrt(SB) = {sqrt_sb:.4f}"),
                (rung0_r, ":", 1.0, f"{gene_set}: rung 0 split-half r = {rung0_r:.4f}"),
            ):
                if not np.isfinite(value):
                    continue
                ax.axhline(
                    value,
                    color=_GENE_SET_COLOR[gene_set],
                    lw=line_width,
                    linestyle=style,
                    label=text,
                )
                headroom.append(float(value))
        # Room above everything drawn, so the legend has empty axes to sit in rather than
        # covering the ceiling lines it names.
        drawn_values = [value for value in headroom if np.isfinite(value)]
        if drawn_values:
            ax.set_ylim(top=max(drawn_values) * 1.45)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=90, fontsize=6)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("model")
        ax.set_ylabel("mean score over the scored pairs (r)")
        ax.set_title(f"({'ab'[position]}) {_SCHEME_LABEL.get(scheme, scheme)}", fontsize=9)
        _legend(ax, fontsize=6)

    ax_control = axes[0][-1]
    if len(score_control):
        predictions = _labels(score_control, "prediction")
        gene_sets = _labels(score_control, "gene_set")
        reliability = _numeric(score_control, "reliability")
        mean_r = _numeric(score_control, "mean_r")
        planted = _numeric(score_control, "planted")
        positions = np.arange(len(score_control), dtype=float)
        colors = [_GENE_SET_COLOR.get(gene_set, _REAL_COLOR) for gene_set in gene_sets]
        ax_control.bar(positions, mean_r, color=colors, alpha=0.85, label="scored by the real code")
        ax_control.scatter(
            positions,
            planted,
            marker="_",
            s=160,
            color="k",
            linewidths=1.8,
            zorder=3,
            label="planted value it had to reproduce",
        )
        ax_control.set_xticks(positions)
        # One rotated line each. Ten three-line horizontal labels overprinted one another in this
        # panel's width, which the rendered figure showed and no assertion about the file could.
        ax_control.set_xticklabels(
            [
                f"{prediction}, {gene_set} genes, R = {r:.4f}"
                for r, prediction, gene_set in zip(reliability, predictions, gene_sets, strict=True)
            ],
            rotation=90,
            fontsize=6,
        )
        ax_control.axhline(0.0, color="k", lw=0.8)
    else:
        _note_empty(ax_control, "no score control table")
    ax_control.set_ylabel("mean score (r)")
    ax_control.set_title(
        "score control: the true change scores sqrt(R),\na second measurement scores R", fontsize=9
    )
    _legend(ax_control, fontsize=6)

    fig.suptitle(
        "score: every model against the ceiling, and the ceiling against a known R", fontsize=11
    )
    return _finish(fig, out)


# --------------------------------------------------------------------------------------------
# null -- what a comparison is read against
# --------------------------------------------------------------------------------------------


#: The hypotheses whose redraw distributions the null figure draws, one panel each (design §8).
_NULL_PANEL_HYPOTHESES: tuple[str, ...] = ("H1(a)", "H2")


def null_panel_comparisons(comparisons: pd.DataFrame) -> list[str]:
    """One comparison per hypothesis in ``_NULL_PANEL_HYPOTHESES``, chosen SEPARATELY.

    Design section 8 asks for "redraws for H1(a) and H2". Taking the first two responding-gene
    comparisons overall does not do that: H1(a) names one comparison under each scheme (Stack
    base against the drug average when a line is hidden, against chemistry only when a drug is
    hidden), so both panels filled with H1(a) and H2 -- one of the rung's two hypotheses -- never
    got a panel at all. Each hypothesis now takes its own first match.
    """
    hypotheses = _labels(comparisons, "hypothesis")
    responding = _labels(comparisons, "gene_set") == "responding"
    names = _labels(comparisons, "comparison")
    drawn: list[str] = []
    for wanted in _NULL_PANEL_HYPOTHESES:
        for name in _ordered_unique(names[(hypotheses == wanted) & responding]):
            if name not in drawn:
                drawn.append(name)
                break
    return drawn


def fig_null(comparisons: pd.DataFrame, redraws: pd.DataFrame, out: Path) -> Path:
    """Every head-to-head comparison with what it could and could not have detected.

    The left panel is the run's answer to its own hypotheses: one row per comparison, the mean
    paired difference between two models with the middle 95% of its redraws, on both gene sets.
    An interval crossing zero is a comparison that did not separate the two models. Beside each
    estimate is its minimum detectable effect -- 2.8 redraw standard deviations, the smallest
    true difference this design would call significant 80% of the time. Reading the two together
    is what separates "no difference" from "not enough data to tell", which is why the MDE is
    drawn rather than tabulated alone.

    The right panels are the redraw distributions behind two of those rows: H1(a), Stack's
    embedding against the reference that knows nothing about the line, and H2, the drug
    fine-tune against the cytokine-tuned release. The vertical line is the estimate and the
    shaded span its interval; the distribution is what the comparison would have looked like had
    the held-out units come out differently.
    """
    fig = plt.figure(figsize=(16.0, 9.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[3, 2])

    ax = fig.add_subplot(grid[:, 0])
    ids = _ordered_unique(_labels(comparisons, "comparison"))
    if ids:
        for offset, gene_set in zip((0.18, -0.18), ("responding", "all"), strict=True):
            chosen = cast(Any, comparisons).loc[_labels(comparisons, "gene_set") == gene_set]
            if not len(chosen):
                continue
            names = _labels(chosen, "comparison")
            y = np.array([ids.index(name) for name in names], dtype=float) + offset
            estimate = _numeric(chosen, "estimate")
            xerr = _error_bars(_numeric(chosen, "ci_lo"), _numeric(chosen, "ci_hi"), estimate)
            ax.errorbar(
                estimate,
                y,
                xerr=xerr,
                fmt="o",
                ms=4,
                lw=1.2,
                capsize=3,
                color=_GENE_SET_COLOR[gene_set],
                label=f"{gene_set} genes (estimate and 95% interval)",
            )
            mde = _numeric(chosen, "mde")
            ax.scatter(
                mde,
                y,
                marker="|",
                s=90,
                color=_GENE_SET_COLOR[gene_set],
                alpha=0.65,
                label=f"{gene_set} genes: minimum detectable effect",
            )
            ax.scatter(-mde, y, marker="|", s=90, color=_GENE_SET_COLOR[gene_set], alpha=0.65)
        ax.set_yticks(np.arange(len(ids)))
        ax.set_yticklabels(ids, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0.0, color="k", lw=1.0)
    else:
        _note_empty(ax, "no comparisons table")
    ax.set_xlabel("mean paired difference in score (model A minus model B)")
    ax.set_title("(a) every comparison with its interval and MDE, both gene sets", fontsize=9)
    _legend(ax, fontsize=6)

    drawn = null_panel_comparisons(comparisons)

    for position, name in enumerate(drawn[:2]):
        ax_draws = fig.add_subplot(grid[position, 1])
        chosen = cast(Any, redraws).loc[
            (_labels(redraws, "comparison") == name)
            & (_labels(redraws, "gene_set") == "responding")
        ]
        values = _finite(chosen, "estimate")
        two_way = _finite(chosen, "estimate_two_way")
        if values.size:
            edges = np.linspace(
                float(np.min(np.concatenate([values, two_way]))),
                float(np.max(np.concatenate([values, two_way]))),
                41,
            )
            ax_draws.hist(
                values,
                bins=edges.tolist(),
                density=True,
                color=_REAL_COLOR,
                alpha=0.6,
                label="redrawing the hidden units",
            )
            if two_way.size:
                ax_draws.hist(
                    two_way,
                    bins=edges.tolist(),
                    density=True,
                    color=_CONTROL_COLOR,
                    alpha=0.45,
                    label="redrawing lines and drugs together",
                )
            row = cast(Any, comparisons).loc[
                (_labels(comparisons, "comparison") == name)
                & (_labels(comparisons, "gene_set") == "responding")
            ]
            estimate = float(_numeric(row, "estimate")[0]) if len(row) else float("nan")
            if np.isfinite(estimate):
                ax_draws.axvline(estimate, color="k", lw=1.4, label=f"estimate = {estimate:.4f}")
                ax_draws.axvspan(
                    float(_numeric(row, "ci_lo")[0]),
                    float(_numeric(row, "ci_hi")[0]),
                    color="k",
                    alpha=0.08,
                )
            ax_draws.axvline(0.0, color="k", lw=0.8, linestyle="--")
        else:
            _note_empty(ax_draws, f"no redraws for {name}")
        ax_draws.set_xlabel("mean paired difference, one value per redraw")
        ax_draws.set_ylabel("density")
        ax_draws.set_title(f"({'bc'[position]}) {name}, responding genes", fontsize=9)
        _legend(ax_draws, fontsize=6)

    if not drawn:
        _note_empty(fig.add_subplot(grid[0, 1]), "no H1(a) or H2 comparison to draw")

    fig.suptitle("null: what each comparison could detect, and the redraws behind it", fontsize=11)
    return _finish(fig, out)


__all__ = ["fig_build", "fig_fit", "fig_null", "fig_score", "fig_split"]
