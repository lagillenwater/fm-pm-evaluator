"""Executable verification of rung 1's held-out-prediction claims (PROCESS section 3).

Every check here recomputes a reported number from the run's own written artifacts -- no cluster
access, no project code, no trust in any write-up -- and prints the claim beside the recomputed
value and a verdict. Trust comes from re-derivation the reader performs, not from narrative the
writer asserts: a reviewer who runs this has re-derived the evidence rather than read about it,
and a number recomputed at read time cannot drift the way a number transcribed across documents
can.

    uv run python scripts/verify_rung1.py
    uv run python scripts/verify_rung1.py --task-dir path/to/a/run --cache path/to/its/cache

The default task directory is ``docs/tasks/rung1-held-out-prediction``, where the run's tables
sit UNCOMMITTED between the run and promotion (PROCESS, "What reaches GitHub, and when").
``--cache`` points at the scratch cache the rounds and redraw blocks were written into; the
intervals, p-values and minimum detectable effects are read off those blocks, so without them
those checks SKIP rather than pass. The notebook
``docs/tasks/rung1-held-out-prediction/verify.ipynb`` recomputes the same claims in
self-contained cells -- standard library and pandas, nothing imported from this project -- and
runs this script only as its final cross-check; ``tests/test_verify_rung1.py`` runs this battery
in continuous integration, so the branch's green does not depend on anyone opening the notebook.

**A check that cannot run SKIPs, naming what was absent, and never passes.** That is the rule
this battery is built around: rung 1's two reviews both caught a number manufactured from
silence (a dictionary default read as a measurement; a figure panel labelled with a hypothesis
it could not show), and a verification battery reporting PASS on an input it never read is the
same defect with a green tick on it.

What is NOT checkable from the task folder alone, stated rather than hidden:

* **The redraws.** 2,000 draws per contrast live in the run's scratch cache as
  ``redraws_{b}.parquet``, not in the task folder. Pass ``--cache`` and every interval, p-value,
  MDE and design effect is recomputed from them; without it those checks skip by name.
* **The inputs the run read.** The answers, descriptions and embeddings are cluster-side and far
  too large to commit. What is checked is the parameter sidecar's sha256 of each one, for every
  input present in this tree, plus rung 0's promoted per-pair table, which is committed.
* **The design's own grid.** A fixture run is smaller than 50 lines x 107 drugs on purpose, so
  the claims that are only true of the design's grid (its 5,350 pairs, its candidate component
  counts, its two leaked pairs) are checked on a design-size grid and skipped, naming the
  shape, on a smaller one.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TASK = "rung1-held-out-prediction"
DEFAULT_TASK_DIR = REPO / "docs" / "tasks" / TASK

#: Rung 0's promoted tables: the ceiling is read from the per-pair one, restricted to this
#: grid's pairs (design.md section 6), and the grid itself is its complete 5 uM sub-table.
RUNG0_PER_PAIR = REPO / "results" / "rung0-assay-reliability" / "rung0_per_pair_r.csv"
RUNG0_DOSE_STRATA = REPO / "results" / "rung0-assay-reliability" / "rung0_dose_strata.csv"

#: Which local file each ``source_sha256`` key in the grid record names. The drug table is a
#: Hugging Face parquet pulled on the cluster, so it has no local path and its entry is
#: reported unresolved rather than silently passed.
GRID_SOURCES: dict[str, Path] = {"per_pair": RUNG0_PER_PAIR, "dose_strata": RUNG0_DOSE_STRATA}

#: Design section 2: one dose, and a complete grid.
DOSE_UM = 5.0
DESIGN_LINES, DESIGN_DRUGS = 50, 107
DESIGN_PAIRS = DESIGN_LINES * DESIGN_DRUGS

#: Design section 6's ceilings. The pair counts behind them (4,593 responding, 5,350 all) are
#: not transcribed here: the ceiling check recomputes them from rung 0's promoted table, which
#: is a stronger statement than agreeing with a number copied out of the design.
DESIGN_CEILING = {"responding": 0.8575, "all": 0.3876}

SCHEMES = ("lolo", "lodo")
GENE_SETS = ("responding", "all")

#: Design section 7's statistics: 2,000 redraws in 8 blocks of 250, an MDE of 2.8 redraw
#: standard deviations, and the base seed ruling 33 fixed.
N_DRAWS, N_BLOCKS, DRAWS_PER_BLOCK = 2000, 8, 250
MDE_FACTOR = 2.8
REDRAW_BASE_SEED = 7000

#: Design section 5: the component counts a PCA or NMF ridge -- and, under ruling 35, its random
#: stand-in -- chooses between, and the four models that choose one.
COMPONENT_KS = [2, 5, 10, 15, 20]
TUNED_MODELS = ("nmf", "pca", "random_nmf", "random_pca")

#: Design section 5 again: nearest lines chooses how many training lines to average, from its own
#: candidates. The settings table's ``k`` column carries BOTH meanings -- a component count for
#: the four models above, a neighbour count here -- so the two are checked against their own
#: candidate sets and never against each other's.
NEAREST_LINES = "nearest_lines"
NEAREST_LINES_KS = [3, 5, 10, 20]

#: Design section 7's leakage: the sci-Plex fine-tune saw A549 with five compounds; two of them
#: are in the grid, and both pairs are removed for EVERY model. Compared on stripped drug names,
#: because the screen's own names carry trailing whitespace.
SCIPLEX_LINE = "ACH-000681"
SCIPLEX_PAIRS = {(SCIPLEX_LINE, "Temsirolimus"), (SCIPLEX_LINE, "Trametinib")}

#: Ruling 40: a leakage record exists for the three Stack versions only. Every other model is
#: fitted in-run from training data alone and has no exposure of its own.
STACK_VERSIONS = ("stack_base", "stack_cytokine", "stack_drug")

#: Design section 7's table, as ids: 6 leave-one-line-out comparisons and 5 leave-one-drug-out.
#: Holm's adjustment applies within each of those two families, on responding genes.
DECLARED_COMPARISONS: dict[str, str] = {
    "lolo_stack_base_vs_drug_average": "lolo",
    "lolo_stack_base_vs_expression": "lolo",
    "lolo_stack_base_vs_pca": "lolo",
    "lolo_stack_base_vs_nmf": "lolo",
    "lolo_stack_base_vs_nearest_lines": "lolo",
    "lolo_stack_drug_vs_stack_cytokine": "lolo",
    "lodo_stack_base_vs_chemistry_only": "lodo",
    "lodo_stack_base_vs_expression": "lodo",
    "lodo_stack_base_vs_pca": "lodo",
    "lodo_stack_base_vs_nmf": "lodo",
    "lodo_stack_drug_vs_stack_cytokine": "lodo",
}

#: Design section 7 also reports, unadjusted, each ridge description against its own random
#: stand-in: the control behind H1(b), since a description that beats its stand-in gained from
#: what it describes rather than from the method every model shares. The run builds these ids in
#: ``scripts/heldout_redraws.py``; they are enumerated here so that a run reporting none of them
#: cannot pass a claim whose text says it covers them.
STAND_IN_DESCRIPTIONS = ("expression", "pca", "nmf", "stack_base", "stack_cytokine", "stack_drug")

STAND_IN_CONTRASTS: dict[str, str] = {
    f"{scheme}_{description}_vs_random_{description}": scheme
    for scheme in SCHEMES
    for description in STAND_IN_DESCRIPTIONS
}

#: Every difference the run reports: 11 declared comparisons and 12 stand-in contrasts.
ALL_CONTRASTS: dict[str, str] = {**DECLARED_COMPARISONS, **STAND_IN_CONTRASTS}

#: The family whose numbers a promotion rests on: everything read off the redraw blocks.
REDRAW_GROUP = "redraws"

#: One figure per design section 8 bullet, and the tables each was drawn from. A figure whose
#: source table was never written is a stage that did not run; a figure missing while its tables
#: exist is a broken figure step, and the battery has to tell those apart.
FIGURES: dict[str, tuple[str, ...]] = {
    "01_build.png": ("rung1_cells.csv", "rung1_identity_match.csv", "rung1_weights_check.json"),
    "02_split.png": ("rung1_pair_scores.csv.gz", "rung1_control_split.csv"),
    "03_fit.png": ("rung1_settings.csv", "rung1_model_summary.csv", "rung1_control_fit.csv"),
    "04_score.png": ("rung1_model_summary.csv", "rung1_ceiling.csv", "rung1_control_score.csv"),
    "05_null.png": ("rung1_comparisons.csv",),
}

GRID_JSON = "rung1_grid.json"
CEILING_CSV = "rung1_ceiling.csv"
PAIR_SCORES = "rung1_pair_scores.csv.gz"
MODEL_SUMMARY = "rung1_model_summary.csv"
COMPARISONS = "rung1_comparisons.csv"
SETTINGS = "rung1_settings.csv"
LEAKAGE_JSON = "rung1_leakage_profiles.json"
PARAMS_JSON = "rung1_run.params.json"

#: A recomputation agrees with the run's own arithmetic to within floating-point association.
#: The tables are written at full precision and read back with ``float_precision="round_trip"``
#: (task 10b measured the default reader shifting the last bit), so this is not a rounding
#: allowance: it is the width of the associativity difference between two summations.
TOLERANCE = 1e-9


@dataclass
class Check:
    """One claim, the value recomputed from the artifacts, and whether they agree.

    ``skipped`` marks a claim that cannot be checked in this working tree: the redraw blocks
    left on scratch, a fixture grid smaller than the design's, a run that has not happened. A
    skipped check reports SKIP and does not fail the battery, so absence is stated rather than
    either hidden or dressed up as a failure -- and never counted as a pass.

    ``group`` names the family a check belongs to, so ``exit_status`` can refuse a clean exit
    when a family that carries the promoted numbers was skipped rather than run.
    """

    name: str
    claim: str
    computed: str
    ok: bool
    skipped: bool = False
    group: str = ""


def skipped(name: str, why: str, group: str = "") -> Check:
    return Check(name, "not checkable in this working tree", why, True, skipped=True, group=group)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_items(items: Sequence[str]) -> str:
    """The grid record's hash of a name list: sha256 of the sorted names, newline-joined."""
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()


def _close(claim: float, actual: float, tolerance: float = TOLERANCE) -> bool:
    if not (np.isfinite(claim) and np.isfinite(actual)):
        return bool(np.isnan(claim) and np.isnan(actual))
    return abs(claim - actual) <= tolerance


def _f(value: Any) -> float:
    """One table cell as a float. A cell read back from a CSV is untyped until something asks it
    for a number, and pandas ships no stubs here, so the ask happens in one annotated place."""
    return float(value)


def _i(value: Any) -> int:
    """One table cell as an integer, for the same reason as ``_f``."""
    return int(value)


def read_table(path: Path) -> pd.DataFrame:
    """Read a written table so that every value is the one the run wrote.

    ``keep_default_na=False`` keeps the cell line whose DepMap identifier is the literal string
    ``NA`` (design.md, inclusion rules) from becoming a missing value, and
    ``float_precision="round_trip"`` keeps pandas' default CSV float parser from shifting the
    last bit of a double -- task 10b measured that, and without it a recomputed mean differs
    from the run's own in the 16th digit for no reason but the reader.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text())
    return loaded if isinstance(loaded, dict) else None


def grid_record(task_dir: Path, cache: Path | None = None) -> dict[str, Any] | None:
    """The run's grid record, from the task folder or (as some runs write it) the cache."""
    for root in (task_dir, cache):
        if root is None:
            continue
        record = _read_json(root / GRID_JSON)
        if record is not None:
            return record
    return None


def is_design_grid(record: dict[str, Any]) -> bool:
    """True unless this grid is plainly a smaller fixture than the design's.

    A grid at least as large as the design's in either dimension is checked strictly against
    50 x 107: a run that lost a drug must FAIL here, not skip. Only a grid smaller in BOTH
    dimensions -- a test fixture -- is exempt.
    """
    lines, drugs = len(record.get("lines", [])), len(record.get("drugs", []))
    return not (lines < DESIGN_LINES and drugs < DESIGN_DRUGS)


def grid_shape(record: dict[str, Any]) -> str:
    return f"{len(record.get('lines', []))} lines x {len(record.get('drugs', []))} drugs"


def grid_from_rung0_table(path: Path, dose: float = DOSE_UM) -> dict[str, Any]:
    """Rung 1's grid, re-derived here from rung 0's promoted per-pair table.

    Design section 2's rule, written out: at ``dose``, the lines are rung 0's distinct
    ``patient`` values and the drugs are those measured in every one of them. Recomputed rather
    than read back, so the committed grid is checked against the table it claims to come from.
    """
    table = read_table(path)
    at_dose = table[pd.to_numeric(table["dose"], errors="coerce") == dose]
    lines = sorted({str(value) for value in at_dose["patient"]})
    per_drug = at_dose.groupby("drug")["patient"].nunique()
    drugs = sorted(
        str(name)
        for name, count in per_drug.items()
        if int(count) == len(lines)  # pyright: ignore[reportArgumentType]
    )
    return {
        "dose": dose,
        "lines": lines,
        "drugs": drugs,
        "metadata_name": {drug: drug for drug in drugs},
        "excluded_pairs": [],
        "sha256_lines": sha256_items(lines),
        "sha256_drugs": sha256_items(drugs),
        "source_sha256": {},
    }


# ------------------------------------------------------------------------------------------
# the grid: what was scored, what it hashes to, and where it came from
# ------------------------------------------------------------------------------------------


def check_grid(task_dir: Path, cache: Path | None = None) -> list[Check]:
    """The grid record against design section 2: one dose, a complete grid, and hashes that
    recompute from the names they pin."""
    record = grid_record(task_dir, cache)
    names = [
        "grid: the dose is 5.0 uM",
        f"grid: {DESIGN_DRUGS} drugs x {DESIGN_LINES} lines = {DESIGN_PAIRS:,} pairs",
        "grid: sha256_lines recomputes from the grid's lines",
        "grid: sha256_drugs recomputes from the grid's drugs",
        "grid: every source_sha256 recomputes from the file it names",
    ]
    if record is None:
        return [skipped(name, f"{task_dir / GRID_JSON} does not exist") for name in names]

    checks: list[Check] = []
    dose = float(record.get("dose", float("nan")))
    checks.append(
        Check(
            names[0],
            f"{GRID_JSON} records dose={record.get('dose')}",
            f"design.md section 2 fixes the dose at {DOSE_UM} uM",
            _close(dose, DOSE_UM),
        )
    )

    lines, drugs = list(record.get("lines", [])), list(record.get("drugs", []))
    if is_design_grid(record):
        checks.append(
            Check(
                names[1],
                f"{GRID_JSON} lists {len(drugs)} drugs and {len(lines)} lines",
                f"{len(lines)} x {len(drugs)} = {len(lines) * len(drugs):,} pairs",
                (len(lines), len(drugs)) == (DESIGN_LINES, DESIGN_DRUGS),
            )
        )
    else:
        checks.append(
            skipped(
                names[1],
                f"this run's grid is {grid_shape(record)}, smaller than the design's "
                f"{DESIGN_LINES} x {DESIGN_DRUGS} in both dimensions (a fixture)",
            )
        )

    design_grid = is_design_grid(record)
    for name, key, items in (
        (names[2], "sha256_lines", lines),
        (names[3], "sha256_drugs", drugs),
    ):
        recorded = str(record.get(key, ""))
        if not recorded:
            # A missing field is a defect on the design's grid, not an excuse to skip: the
            # restriction record's data contract names it, and skipping here would key the
            # check on its own input being absent.
            checks.append(
                Check(
                    name,
                    f"{GRID_JSON} records no {key}",
                    f"the restriction record's contract requires {key}, and this is the "
                    f"design's {DESIGN_LINES} x {DESIGN_DRUGS} grid",
                    False,
                )
                if design_grid
                else skipped(
                    name,
                    f"{GRID_JSON} records no {key}; this grid is {grid_shape(record)}, a "
                    "fixture rather than the design's",
                )
            )
            continue
        recomputed = sha256_items([str(item) for item in items])
        checks.append(
            Check(
                name,
                f"{key} {recorded}",
                f"sha256 of the {len(items)} sorted names: {recomputed}",
                recorded == recomputed,
            )
        )

    checks.append(_check_grid_sources(record, names[4], design_grid))
    return checks


def _check_grid_sources(record: dict[str, Any], name: str, design_grid: bool) -> Check:
    """Each ``source_sha256`` entry against the file it names, where that file is in this tree."""
    sources = dict(record.get("source_sha256", {}))
    if not sources:
        if design_grid:
            return Check(
                name,
                f"{GRID_JSON} records no source_sha256",
                "the restriction record's contract requires it: without it nothing pins which "
                "bytes of rung 0's tables the grid was built from",
                False,
            )
        return skipped(
            name,
            f"{GRID_JSON} records no source_sha256; this grid is {grid_shape(record)}, a "
            "fixture rather than the design's",
        )
    matched, moved, unresolved = [], [], []
    for key, digest in sources.items():
        path = GRID_SOURCES.get(str(key))
        if path is None or not path.exists():
            unresolved.append(str(key))
        elif sha256_of(path) == str(digest):
            matched.append(str(key))
        else:
            moved.append(str(key))
    if not matched and not moved:
        return skipped(name, f"none of {sorted(sources)} is in this tree (cluster-side inputs)")
    detail = f"{len(matched)} of {len(sources)} recompute"
    if moved:
        detail += f"; CHANGED {sorted(moved)}"
    if unresolved:
        detail += f"; not in this tree {sorted(unresolved)}"
    return Check(name, f"{len(sources)} source checksums in {GRID_JSON}", detail, not moved)


def check_grid_provenance(task_dir: Path, cache: Path | None = None) -> list[Check]:
    """The grid's lines and drugs against rung 0's promoted table: the complete 5 uM sub-table,
    re-derived here rather than taken on trust (design section 2 builds it from that table, never
    from a hand-typed list)."""
    name = "grid: the lines and drugs are rung 0's complete 5 uM grid"
    record = grid_record(task_dir, cache)
    if record is None:
        return [skipped(name, f"{task_dir / GRID_JSON} does not exist")]
    if not RUNG0_PER_PAIR.exists():
        return [skipped(name, f"{RUNG0_PER_PAIR} is not in this checkout")]

    derived = grid_from_rung0_table(RUNG0_PER_PAIR)
    lines = [str(value) for value in record.get("lines", [])]
    drugs = [str(value) for value in record.get("drugs", [])]
    shared = set(lines) & set(derived["lines"])
    if not shared:
        return [
            skipped(
                name,
                f"none of this run's {len(lines)} lines is in rung 0's 5 uM table, so this grid "
                "is a fixture rather than the screen's",
            )
        ]
    agree = lines == derived["lines"] and drugs == derived["drugs"]
    return [
        Check(
            name,
            f"{GRID_JSON} lists {len(lines)} lines and {len(drugs)} drugs",
            f"rung 0's complete 5 uM grid gives {len(derived['lines'])} lines and "
            f"{len(derived['drugs'])} drugs; identical: {agree}",
            agree,
        )
    ]


# ------------------------------------------------------------------------------------------
# the ceiling: the one number rung 1 inherits rather than measures
# ------------------------------------------------------------------------------------------


def check_ceiling(task_dir: Path, repo: Path = REPO, cache: Path | None = None) -> list[Check]:
    """``rung1_ceiling.csv``: its own arithmetic, the design's declared values, and a
    recomputation from rung 0's promoted per-pair table restricted to this grid."""
    names = [
        "ceiling: sb is 2r/(1+r) of the split-half mean, and sqrt_sb its square root",
        "ceiling: the declared √SB of design section 6",
        "ceiling: recomputes from rung 0's promoted per-pair table",
    ]
    path = task_dir / CEILING_CSV
    if not path.exists():
        return [skipped(name, f"{path} does not exist") for name in names]
    table = read_table(path)
    rows = {str(row["gene_set"]): row for _, row in table.iterrows()}

    checks: list[Check] = []
    bad_arithmetic: list[str] = []
    for gene_set, row in rows.items():
        split_half = _f(row["split_half_r"])
        sb, sqrt_sb = _f(row["sb"]), _f(row["sqrt_sb"])
        expected_sb = 2.0 * split_half / (1.0 + split_half)
        if not (_close(sb, expected_sb, 1e-4) and _close(sqrt_sb, float(np.sqrt(sb)), 1e-4)):
            bad_arithmetic.append(gene_set)
    checks.append(
        Check(
            names[0],
            f"{len(rows)} gene sets, each with split_half_r, sb and sqrt_sb",
            "every row recomputes" if not bad_arithmetic else f"disagreeing: {bad_arithmetic}",
            not bad_arithmetic and bool(rows),
        )
    )

    declared = {
        gene_set: (_f(rows[gene_set]["sqrt_sb"]) if gene_set in rows else float("nan"))
        for gene_set in GENE_SETS
    }
    checks.append(
        Check(
            names[1],
            ", ".join(f"{gene_set} {value:.4f}" for gene_set, value in declared.items()),
            ", ".join(f"{gene_set} {value:.4f}" for gene_set, value in DESIGN_CEILING.items()),
            all(_close(declared[g], DESIGN_CEILING[g], 5e-5) for g in GENE_SETS),
        )
    )

    checks.append(_check_ceiling_against_rung0(task_dir, rows, names[2], repo, cache))
    return checks


def _check_ceiling_against_rung0(
    task_dir: Path,
    rows: dict[str, Any],
    name: str,
    repo: Path,
    cache: Path | None,
) -> Check:
    """The ceiling recomputed from rung 0's promoted per-pair table, restricted to this grid.

    For each gene set: average rung 0's per-pair split-half correlation over the grid's pairs it
    scored, lift it with Spearman-Brown, take the square root. That is design section 6's
    construction, done here from the committed table.
    """
    per_pair = repo / RUNG0_PER_PAIR.relative_to(REPO)
    record = grid_record(task_dir, cache)
    if not per_pair.exists():
        return skipped(name, f"{per_pair} is not in this checkout")
    if record is None:
        return skipped(name, f"{task_dir / GRID_JSON} does not exist, so there is no grid to use")

    table = read_table(per_pair)
    at_dose = cast(Any, table[pd.to_numeric(table["dose"], errors="coerce") == DOSE_UM])
    lines = sorted({str(value) for value in record.get("lines", [])})
    drugs = sorted({str(value) for value in record.get("drugs", [])})
    in_grid = at_dose[at_dose["patient"].isin(lines) & at_dose["drug"].isin(drugs)]
    if in_grid.empty:
        return skipped(
            name,
            "none of this run's pairs is in rung 0's 5 uM table, so this grid is a fixture "
            "rather than the screen's",
        )

    disagreeing: list[str] = []
    detail: list[str] = []
    for gene_set, column in (("responding", "r_responder"), ("all", "r")):
        scored = cast(Any, pd.to_numeric(in_grid[column], errors="coerce")).dropna()
        split_half = _f(scored.mean())
        sb = 2.0 * split_half / (1.0 + split_half)
        sqrt_sb = float(np.sqrt(sb))
        detail.append(f"{gene_set} {len(scored):,} pairs, r {split_half:.4f}, √SB {sqrt_sb:.4f}")
        row = rows.get(gene_set)
        if row is None:
            disagreeing.append(f"{gene_set} (no row)")
            continue
        agrees = (
            _close(float(row["split_half_r"]), split_half, 5e-5)
            and _close(float(row["sqrt_sb"]), sqrt_sb, 5e-5)
            and int(row["pairs_scored_rung0"]) == len(scored)
        )
        if not agrees:
            disagreeing.append(gene_set)
    claim = "; ".join(
        f"{gene_set} {int(rows[gene_set]['pairs_scored_rung0']):,} pairs, "
        f"r {float(rows[gene_set]['split_half_r']):.4f}, "
        f"√SB {float(rows[gene_set]['sqrt_sb']):.4f}"
        for gene_set in rows
    )
    return Check(name, claim, "; ".join(detail), not disagreeing)


# ------------------------------------------------------------------------------------------
# the scored population: the pairs every model scored, rebuilt from the written scores
# ------------------------------------------------------------------------------------------


def load_pair_scores(task_dir: Path) -> pd.DataFrame | None:
    path = task_dir / PAIR_SCORES
    return read_table(path) if path.exists() else None


def scored_population(pair_scores: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """One model per column, only the pairs every model scored -- the population every estimate,
    every mean score and every redraw in the run was taken over."""
    prepared: dict[tuple[str, str], pd.DataFrame] = {}
    for scheme in SCHEMES:
        for gene_set in GENE_SETS:
            chosen = (pair_scores["scheme"] == scheme) & (pair_scores["gene_set"] == gene_set)
            rows = pair_scores.loc[chosen]
            if rows.empty:
                continue
            wide = rows.pivot(index=["line", "drug"], columns="model", values="r")
            prepared[(scheme, gene_set)] = wide.dropna(axis=0, how="any")
    return prepared


def load_redraws(cache: Path | None) -> tuple[pd.DataFrame | None, str]:
    """Every redraw block, concatenated -- or ``None`` and the reason there are none.

    All ``N_BLOCKS`` or nothing: a statistic read off part of its draws is a different
    statistic, and reporting it as checked would be worse than reporting it unchecked.
    """
    if cache is None:
        return None, "no --cache was given, so the redraw blocks were not read"
    paths = [cache / f"redraws_{block}.parquet" for block in range(N_BLOCKS)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        return None, f"{len(missing)} of {N_BLOCKS} redraw blocks are absent from {cache}"
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True), ""


def _draws(redraws: pd.DataFrame, comparison: str, gene_set: str, column: str) -> np.ndarray:
    chosen = (redraws["comparison"] == comparison) & (redraws["gene_set"] == gene_set)
    values = redraws.loc[chosen].sort_values("draw")[column].to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _percentiles(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _two_sided_p(draws: np.ndarray) -> float:
    """Design section 7's p-value: twice the smaller share of draws on either side of zero, each
    share counted with one added draw so p is never exactly 0."""
    n = draws.size
    at_or_below = (1 + int(np.count_nonzero(draws <= 0))) / (1 + n)
    at_or_above = (1 + int(np.count_nonzero(draws >= 0))) / (1 + n)
    return float(min(1.0, 2.0 * min(at_or_below, at_or_above)))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm's step-down adjusted p-values, in the input order: the i-th smallest of m is
    multiplied by m - i, a running maximum keeps them monotone, each capped at 1."""
    values = np.asarray(p, dtype=np.float64)
    m = values.size
    order = np.argsort(values, kind="stable")
    stepped = np.maximum.accumulate((m - np.arange(m)) * values[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, stepped)
    return adjusted


# ------------------------------------------------------------------------------------------
# each model's score, and its fraction of the ceiling
# ------------------------------------------------------------------------------------------


def check_model_summary(task_dir: Path) -> list[Check]:
    """``rung1_model_summary.csv`` against the per-pair scores it summarises.

    Keyed by (scheme, model, gene set), per ruling 39: a model fitted with a line hidden and the
    same model fitted with a drug hidden predict different things, and pooling them would
    average two questions.
    """
    names = ["model summary: one row per scheme, model and gene set"]
    summary_path, scores_path = task_dir / MODEL_SUMMARY, task_dir / PAIR_SCORES
    if not summary_path.exists() or not scores_path.exists():
        missing = summary_path if not summary_path.exists() else scores_path
        base = [skipped(names[0], f"{missing} does not exist")]
        return base + [
            skipped(f"model summary: {scheme}/{gene_set} {what}", f"{missing} does not exist")
            for scheme in SCHEMES
            for gene_set in GENE_SETS
            for what in (
                "mean score recomputes from the pair scores",
                "fraction of the ceiling is the mean over √SB",
                "the interval contains the mean",
            )
        ]

    summary = read_table(summary_path)
    pair_scores = load_pair_scores(task_dir)
    assert pair_scores is not None
    population = scored_population(pair_scores)
    ceiling = _ceiling_values(task_dir)

    expected = {
        (scheme, str(model), gene_set)
        for (scheme, gene_set), wide in population.items()
        for model in wide.columns
    }
    reported = {
        (str(row["scheme"]), str(row["model"]), str(row["gene_set"]))
        for _, row in summary.iterrows()
    }
    checks = [
        Check(
            names[0],
            f"{len(summary)} rows in {MODEL_SUMMARY}",
            f"{len(expected)} (scheme, model, gene set) combinations in the pair scores; "
            f"missing {sorted(expected - reported)[:3]}, extra {sorted(reported - expected)[:3]}",
            expected == reported and len(summary) == len(expected),
        )
    ]

    for (scheme, gene_set), wide in sorted(population.items()):
        rows = summary.loc[(summary["scheme"] == scheme) & (summary["gene_set"] == gene_set)]
        indexed = rows.set_index("model")
        worst_mean, worst_fraction = 0.0, 0.0
        outside: list[str] = []
        disagreeing: list[str] = []
        for model in wide.columns:
            values = wide[model].to_numpy(dtype=np.float64)
            mean_r = float(values.mean())
            if str(model) not in indexed.index:
                disagreeing.append(str(model))
                continue
            row = indexed.loc[str(model)]
            worst_mean = max(worst_mean, abs(float(row["mean_r"]) - mean_r))
            if int(row["n_pairs"]) != len(values):
                disagreeing.append(f"{model} (n_pairs)")
            sqrt_sb = float(row["sqrt_sb"])
            if gene_set in ceiling and not _close(sqrt_sb, ceiling[gene_set], 5e-5):
                disagreeing.append(f"{model} (sqrt_sb)")
            worst_fraction = max(
                worst_fraction, abs(float(row["fraction_of_ceiling"]) - mean_r / sqrt_sb)
            )
            if not float(row["ci_lo"]) <= mean_r <= float(row["ci_hi"]):
                outside.append(str(model))
        key = f"model summary: {scheme}/{gene_set}"
        checks.extend(
            [
                Check(
                    f"{key} mean score recomputes from the pair scores",
                    f"{len(rows)} models' mean_r over {len(wide)} pairs",
                    f"largest difference {worst_mean:.3e}"
                    + (f"; disagreeing {disagreeing}" if disagreeing else ""),
                    worst_mean <= TOLERANCE and not disagreeing,
                ),
                # Without rung1_ceiling.csv the only denominator available is the summary's own
                # sqrt_sb, so the arithmetic would check the table against itself. That is
                # self-consistency, not verification, and it fails rather than passing quietly.
                Check(
                    f"{key} fraction of the ceiling is the mean over √SB",
                    f"{len(rows)} fractions"
                    + (f" against √SB {ceiling[gene_set]:.4f}" if gene_set in ceiling else ""),
                    f"largest difference {worst_fraction:.3e}"
                    if gene_set in ceiling
                    else f"{task_dir / CEILING_CSV} is absent, so the reported fractions could "
                    "only be read against the denominator the same table carries",
                    gene_set in ceiling and worst_fraction <= TOLERANCE,
                ),
                Check(
                    f"{key} the interval contains the mean",
                    f"{len(rows)} intervals from the redrawn held-out units",
                    "every mean inside its interval"
                    if not outside
                    else f"outside their interval: {outside}",
                    not outside,
                ),
            ]
        )
    return checks


def _ceiling_values(task_dir: Path) -> dict[str, float]:
    path = task_dir / CEILING_CSV
    if not path.exists():
        return {}
    table = read_table(path)
    return {str(row["gene_set"]): _f(row["sqrt_sb"]) for _, row in table.iterrows()}


# ------------------------------------------------------------------------------------------
# the comparisons: estimate, interval, p-value, MDE, design effect, Holm
# ------------------------------------------------------------------------------------------


def check_comparisons(task_dir: Path, cache: Path | None = None) -> list[Check]:
    """``rung1_comparisons.csv`` against the per-pair scores and the redraw blocks."""
    names = [
        "comparisons: every declared comparison and stand-in contrast is on both gene sets",
    ]
    redraw_names = [
        "comparisons: confidence intervals are the redraws' 2.5/97.5 percentiles",
        "comparisons: p-values recompute from the redraws",
        "comparisons: minimum detectable effects are 2.8 redraw standard deviations",
        "comparisons: the two-way design effect and its width ratio recompute",
        "redraws: every contrast holds all 2,000 draws, once each",
    ]
    path, scores_path = task_dir / COMPARISONS, task_dir / PAIR_SCORES
    if not path.exists() or not scores_path.exists():
        missing = path if not path.exists() else scores_path
        return (
            [skipped(names[0], f"{missing} does not exist")]
            + [
                skipped(
                    f"comparisons: {gene_set} estimates recompute as the mean paired difference",
                    f"{missing} does not exist",
                )
                for gene_set in GENE_SETS
            ]
            + [
                skipped(name, f"{missing} does not exist", group=REDRAW_GROUP)
                for name in redraw_names
            ]
            + [
                skipped(
                    f"comparisons: Holm within each test ({scheme})", f"{missing} does not exist"
                )
                for scheme in SCHEMES
            ]
            + [
                skipped(
                    "comparisons: p_holm is empty where the design reports unadjusted",
                    f"{missing} does not exist",
                )
            ]
        )

    table = read_table(path)
    pair_scores = load_pair_scores(task_dir)
    assert pair_scores is not None
    population = scored_population(pair_scores)

    reported = set(table["comparison"].astype(str))
    missing_declared = sorted(set(DECLARED_COMPARISONS) - reported)
    missing_stand_ins = sorted(set(STAND_IN_CONTRASTS) - reported)
    per_contrast = table.groupby("comparison").size()
    checks = [
        Check(
            names[0],
            f"{len(reported)} contrasts x {len(GENE_SETS)} gene sets = {len(table)} rows",
            f"design section 7 declares {len(DECLARED_COMPARISONS)} comparisons and "
            f"{len(STAND_IN_CONTRASTS)} description-versus-stand-in contrasts, "
            f"{len(ALL_CONTRASTS)} in all; missing comparisons {missing_declared}; missing "
            f"stand-in contrasts {missing_stand_ins}; rows per contrast "
            f"{sorted(set(per_contrast.tolist()))}",
            not missing_declared
            and not missing_stand_ins
            and set(per_contrast.tolist()) == {len(GENE_SETS)},
        )
    ]
    checks.extend(_check_estimates(table, population))
    checks.extend(_check_redraw_statistics(table, cache, redraw_names))
    checks.extend(_check_holm(table))
    return checks


def _check_estimates(
    table: pd.DataFrame, population: dict[tuple[str, str], pd.DataFrame]
) -> list[Check]:
    """Every estimate as the mean paired difference over the pairs both models scored."""
    checks: list[Check] = []
    for gene_set in GENE_SETS:
        rows = table.loc[table["gene_set"] == gene_set]
        name = f"comparisons: {gene_set} estimates recompute as the mean paired difference"
        if rows.empty:
            checks.append(skipped(name, f"no {gene_set}-gene rows in {COMPARISONS}"))
            continue
        worst, disagreeing = 0.0, []
        for _, row in rows.iterrows():
            wide = population.get((str(row["scheme"]), gene_set))
            if wide is None or not {str(row["model_a"]), str(row["model_b"])} <= set(wide.columns):
                disagreeing.append(f"{row['comparison']} (no scored pairs)")
                continue
            difference = wide[str(row["model_a"])].to_numpy(dtype=np.float64) - wide[
                str(row["model_b"])
            ].to_numpy(dtype=np.float64)
            worst = max(worst, abs(float(row["estimate"]) - float(difference.mean())))
            if int(row["n_pairs"]) != difference.size:
                disagreeing.append(f"{row['comparison']} (n_pairs)")
        checks.append(
            Check(
                name,
                f"{len(rows)} estimates on {gene_set} genes",
                f"largest difference {worst:.3e}"
                + (f"; disagreeing {disagreeing}" if disagreeing else ""),
                worst <= TOLERANCE and not disagreeing,
            )
        )
    return checks


def _check_redraw_statistics(
    table: pd.DataFrame, cache: Path | None, names: Sequence[str]
) -> list[Check]:
    """The interval, p-value, MDE and design effect of every row, from the redraw blocks."""
    redraws, why = load_redraws(cache)
    if redraws is None:
        return [skipped(name, why, group=REDRAW_GROUP) for name in names]

    worst = {"ci": 0.0, "p": 0.0, "mde": 0.0, "effect": 0.0}
    absent: list[str] = []
    incomplete: list[str] = []
    for _, row in table.iterrows():
        comparison, gene_set = str(row["comparison"]), str(row["gene_set"])
        one_way = _draws(redraws, comparison, gene_set, "estimate")
        two_way = _draws(redraws, comparison, gene_set, "estimate_two_way")
        if one_way.size == 0:
            absent.append(f"{comparison}/{gene_set}")
            continue
        if _i(row["n_draws"]) != one_way.size:
            incomplete.append(f"{comparison}/{gene_set}")
        low, high = _percentiles(one_way)
        worst["ci"] = max(worst["ci"], abs(_f(row["ci_lo"]) - low), abs(_f(row["ci_hi"]) - high))
        worst["p"] = max(worst["p"], abs(_f(row["p"]) - _two_sided_p(one_way)))
        sd = float(one_way.std(ddof=1))
        worst["mde"] = max(worst["mde"], abs(_f(row["mde"]) - MDE_FACTOR * sd))
        effect = float(np.var(two_way, ddof=1) / np.var(one_way, ddof=1))
        worst["effect"] = max(
            worst["effect"],
            abs(_f(row["design_effect"]) - effect),
            abs(_f(row["width_ratio"]) - float(np.sqrt(effect))),
        )

    trouble = (f"; contrasts with no draws {absent}" if absent else "") + (
        f"; wrong n_draws {incomplete}" if incomplete else ""
    )
    clean = not absent and not incomplete
    checks = [
        Check(
            names[0],
            f"{len(table)} intervals in {COMPARISONS}",
            f"largest difference from the redraws' percentiles {worst['ci']:.3e}{trouble}",
            worst["ci"] <= TOLERANCE and clean,
            group=REDRAW_GROUP,
        ),
        Check(
            names[1],
            f"{len(table)} two-sided p-values",
            f"largest difference {worst['p']:.3e}{trouble}",
            worst["p"] <= TOLERANCE and clean,
            group=REDRAW_GROUP,
        ),
        Check(
            names[2],
            f"{len(table)} minimum detectable effects",
            f"largest difference from {MDE_FACTOR} x the redraw sd {worst['mde']:.3e}{trouble}",
            worst["mde"] <= TOLERANCE and clean,
            group=REDRAW_GROUP,
        ),
        Check(
            names[3],
            f"{len(table)} design effects and width ratios",
            f"largest difference {worst['effect']:.3e}{trouble}",
            worst["effect"] <= TOLERANCE and clean,
            group=REDRAW_GROUP,
        ),
    ]
    checks.append(_check_redraw_completeness(redraws, names[4]))
    return checks


def _check_redraw_completeness(redraws: pd.DataFrame, name: str) -> Check:
    """Every (contrast, gene set) redrawn exactly ``N_DRAWS`` times, numbered 0..N-1 across the
    blocks: a comparison read off part of its draws is a different comparison."""
    counts = redraws.groupby(["comparison", "gene_set"])["draw"].agg(["size", "nunique", "max"])
    wrong = counts[
        (counts["size"] != N_DRAWS)
        | (counts["nunique"] != N_DRAWS)
        | (counts["max"] != N_DRAWS - 1)
    ]
    return Check(
        name,
        f"{len(counts)} (contrast, gene set) pairs, {N_BLOCKS} blocks of {DRAWS_PER_BLOCK}",
        f"{len(counts) - len(wrong)} of {len(counts)} hold draws 0..{N_DRAWS - 1} once each"
        + (f"; wrong: {list(cast(Any, wrong).index)[:3]}" if len(wrong) else ""),
        len(wrong) == 0 and len(counts) > 0,
        group=REDRAW_GROUP,
    )


def _check_holm(table: pd.DataFrame) -> list[Check]:
    """Holm's adjustment recomputed within each test, and left off everything design section 7
    reports unadjusted (the all-genes rows and every stand-in contrast)."""
    checks: list[Check] = []
    adjusted = table.loc[table["p_holm"].notna()]
    for scheme in SCHEMES:
        name = f"comparisons: Holm within each test ({scheme})"
        family = adjusted.loc[adjusted["scheme"] == scheme].sort_values("comparison")
        if family.empty:
            checks.append(skipped(name, f"no adjusted {scheme} rows in {COMPARISONS}"))
            continue
        recomputed = holm(family["p"].to_numpy(dtype=np.float64))
        worst = float(np.max(np.abs(recomputed - family["p_holm"].to_numpy(dtype=np.float64))))
        expected = sum(1 for s in DECLARED_COMPARISONS.values() if s == scheme)
        checks.append(
            Check(
                name,
                f"{len(family)} adjusted p-values ({scheme}), design section 7 declares {expected}",
                f"largest difference from Holm over the family {worst:.3e}",
                worst <= TOLERANCE and len(family) == expected,
            )
        )

    unadjusted = table.loc[table["p_holm"].isna()]
    wrong = [
        str(row["comparison"])
        for _, row in adjusted.iterrows()
        if str(row["gene_set"]) != "responding"
        or str(row["comparison"]) not in DECLARED_COMPARISONS
    ]
    checks.append(
        Check(
            "comparisons: p_holm is empty where the design reports unadjusted",
            f"{len(adjusted)} rows carry p_holm, {len(unadjusted)} do not",
            "every adjusted row is a declared comparison on responding genes"
            if not wrong
            else f"adjusted but should not be: {wrong}",
            not wrong and not adjusted.empty,
        )
    )
    return checks


# ------------------------------------------------------------------------------------------
# leakage: the pairs that came out, and what each model version saw
# ------------------------------------------------------------------------------------------


def check_leakage(task_dir: Path, cache: Path | None = None) -> list[Check]:
    """The removed pairs, and the leakage records design section 7 asks for."""
    names = [
        "leakage: the excluded pairs are scored for no model",
        "leakage: the design's two sci-Plex pairs are the grid's excluded pairs",
        "leakage: one record per Stack version, saying what it saw",
    ]
    record = grid_record(task_dir, cache)
    scores_path = task_dir / PAIR_SCORES
    checks: list[Check] = []

    if record is None:
        checks.append(skipped(names[0], f"{task_dir / GRID_JSON} does not exist"))
        checks.append(skipped(names[1], f"{task_dir / GRID_JSON} does not exist"))
    else:
        excluded = [(str(pair[0]), str(pair[1])) for pair in record.get("excluded_pairs", [])]
        if not scores_path.exists():
            checks.append(skipped(names[0], f"{scores_path} does not exist"))
        elif not excluded:
            checks.append(
                skipped(names[0], f"{GRID_JSON} removed no pairs, so there is nothing to look for")
            )
        else:
            scores = read_table(scores_path)
            present = {
                (str(line), str(drug))
                for line, drug in zip(scores["line"], scores["drug"], strict=True)
            }
            found = [pair for pair in excluded if pair in present]
            checks.append(
                Check(
                    names[0],
                    f"{len(excluded)} pairs removed for every model: {excluded}",
                    f"{len(found)} of them appear in {PAIR_SCORES}"
                    + (f": {found}" if found else "")
                    + f" (over {len(present):,} scored pairs)",
                    not found,
                )
            )

        lines = {str(value) for value in record.get("lines", [])}
        if SCIPLEX_LINE not in lines:
            checks.append(
                skipped(
                    names[1],
                    f"{SCIPLEX_LINE} is not one of this run's {len(lines)} lines, so the design's "
                    "two pairs cannot be in this grid",
                )
            )
        else:
            stripped = {(str(a), str(b).strip()) for a, b in record.get("excluded_pairs", [])}
            checks.append(
                Check(
                    names[1],
                    f"{GRID_JSON} excludes {sorted(stripped)}",
                    f"design section 7 removes {sorted(SCIPLEX_PAIRS)}",
                    stripped == SCIPLEX_PAIRS,
                )
            )

    checks.append(_check_leakage_records(task_dir, record, names[2]))
    return checks


def _check_leakage_records(task_dir: Path, record: dict[str, Any] | None, name: str) -> Check:
    """One record per Stack version (ruling 40), none of which saw Tahoe's treated cells, and the
    fine-tune's exposed pairs equal to the pairs the grid removed."""
    path = task_dir / LEAKAGE_JSON
    if not path.exists():
        return skipped(name, f"{path} does not exist")
    profiles = json.loads(path.read_text())
    versions = [str(profile.get("model_version")) for profile in profiles]
    fine_tuned = [p for p in profiles if bool(p.get("sciplex_fine_tuned"))]
    problems: list[str] = []
    if versions != list(STACK_VERSIONS):
        problems.append(f"versions {versions}")
    if any(bool(profile.get("saw_tahoe_treated_cells")) for profile in profiles):
        problems.append("a version is recorded as having seen Tahoe's treated cells")
    if [str(p.get("model_version")) for p in fine_tuned] != ["stack_drug"]:
        problems.append("the sci-Plex fine-tune is not the only fine-tuned version")
    for profile in profiles:
        pairs = [tuple(str(v) for v in pair) for pair in profile.get("exposed_pairs", [])]
        if len(pairs) != int(profile.get("n_exposed_pairs", -1)):
            problems.append(f"{profile.get('model_version')} miscounts its exposed pairs")
        if record is not None and bool(profile.get("sciplex_fine_tuned")):
            excluded = [tuple(str(v) for v in pair) for pair in record.get("excluded_pairs", [])]
            in_grid = SCIPLEX_LINE in {str(v) for v in record.get("lines", [])}
            if bool(profile.get("sciplex_line_in_grid")) != in_grid:
                problems.append("sciplex_line_in_grid disagrees with the grid")
            if in_grid and sorted(pairs) != sorted(excluded):
                problems.append("the exposed pairs are not the pairs the grid removed")
    return Check(
        name,
        f"{len(profiles)} records: {versions}",
        "one per Stack version, none saw treated cells, the fine-tune names what it saw"
        if not problems
        else "; ".join(problems),
        not problems and len(profiles) == len(STACK_VERSIONS),
    )


# ------------------------------------------------------------------------------------------
# the figures, the settings, and the provenance record
# ------------------------------------------------------------------------------------------


def check_figures(task_dir: Path) -> list[Check]:
    """Every figure design section 8 declares, and the tables each was drawn from."""
    names = [
        "figures: every figure design section 8 declares is a real PNG",
        "figures: every figure's source tables were written",
    ]
    directory = task_dir / "figures"
    if not directory.exists():
        if not (task_dir / MODEL_SUMMARY).exists():
            return [skipped(name, f"{task_dir} holds no run yet") for name in names]
        # The run wrote its tables, so this is a figure step that did not run -- which is
        # exactly the case this pair of claims exists to tell apart from a stage never started.
        return [
            Check(
                name,
                f"{directory} does not exist",
                "the run's tables are here, so the figures were not drawn",
                False,
            )
            for name in names
        ]

    real, missing = [], []
    for figure in FIGURES:
        path = directory / figure
        if not path.exists():
            missing.append(figure)
        elif path.stat().st_size > 5_000 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n":
            real.append(figure)
        else:
            missing.append(f"{figure} (not a PNG over 5 kB)")

    absent_sources = {
        figure: [table for table in tables if not (task_dir / table).exists()]
        for figure, tables in FIGURES.items()
    }
    absent = {figure: tables for figure, tables in absent_sources.items() if tables}
    return [
        Check(
            names[0],
            f"{len(FIGURES)} figures, each a PNG over 5 kB",
            f"{len(real)} present and non-trivial" + (f"; missing {missing}" if missing else ""),
            not missing,
        ),
        Check(
            names[1],
            f"{sum(len(t) for t in FIGURES.values())} source tables across {len(FIGURES)} figures",
            "every source table is in the task folder" if not absent else f"absent: {absent}",
            not absent,
        ),
    ]


def check_settings(task_dir: Path, cache: Path | None = None) -> list[Check]:
    """``rung1_settings.csv``: one row per model in every round of both schemes, every chosen
    component count drawn from the candidate set the run recorded, and nearest lines' neighbour
    count drawn from its own."""
    names = [
        "settings: every round of both schemes recorded one row per model",
        "settings: every chosen component count is one of the run's candidates",
        "settings: nearest lines chose from the design's neighbour counts",
        "settings: every chosen setting has a recorded candidate set to have come from",
    ]
    path = task_dir / SETTINGS
    record = grid_record(task_dir, cache)
    if not path.exists():
        return [skipped(name, f"{path} does not exist") for name in names]
    if record is None:
        return [skipped(name, f"{task_dir / GRID_JSON} does not exist") for name in names]

    table = read_table(path)
    rounds = {"lolo": len(record.get("lines", [])), "lodo": len(record.get("drugs", []))}
    problems: list[str] = []
    for scheme, n_rounds in rounds.items():
        rows = table.loc[table["scheme"] == scheme]
        seen = sorted({int(value) for value in rows["round"]})
        if seen != list(range(n_rounds)):
            problems.append(f"{scheme}: rounds {len(seen)} of {n_rounds}")
            continue
        per_round = rows.groupby("round")["model"].apply(lambda models: tuple(sorted(models)))
        if len(set(per_round)) != 1:
            problems.append(f"{scheme}: the models fitted differ between rounds")
    checks = [
        Check(
            names[0],
            f"{len(table)} settings rows",
            f"{rounds['lolo']} leave-one-line-out and {rounds['lodo']} leave-one-drug-out rounds"
            + ("" if not problems else f"; {problems}"),
            not problems,
        )
    ]

    params = _read_json(task_dir / PARAMS_JSON)
    candidates = {
        str(model): [int(k) for k in ks]
        for model, ks in dict((params or {}).get("component_ks", {})).items()
    }
    chosen = table.loc[table["k"].notna()]
    components = chosen.loc[chosen["model"].isin(list(candidates))]
    if not candidates:
        checks.append(skipped(names[1], f"{PARAMS_JSON} records no component_ks"))
    elif components.empty:
        checks.append(
            skipped(names[1], f"no row of {SETTINGS} chose a count for {sorted(candidates)}")
        )
    else:
        outside = sorted(
            {
                f"{row['model']} k={int(row['k'])}"
                for _, row in components.iterrows()
                if int(row["k"]) not in candidates[str(row["model"])]
            }
        )
        checks.append(
            Check(
                names[1],
                f"{len(components)} rows chose a component count",
                f"candidates {candidates}"
                + ("" if not outside else f"; chosen outside them: {outside[:5]}"),
                not outside,
            )
        )

    # Every other k belongs to nearest lines, whose candidates are neighbour counts, not
    # components. A k from a model in neither group is a row nothing declares a meaning for.
    neighbours = chosen.loc[chosen["model"] == NEAREST_LINES]
    if neighbours.empty:
        checks.append(skipped(names[2], f"no {NEAREST_LINES} row of {SETTINGS} chose a k"))
    else:
        wrong = sorted(
            {
                f"{NEAREST_LINES} k={int(row['k'])}"
                for _, row in neighbours.iterrows()
                if int(row["k"]) not in NEAREST_LINES_KS
            }
        )
        checks.append(
            Check(
                names[2],
                f"{len(neighbours)} rows chose a neighbour count",
                f"design section 5 offers {NEAREST_LINES_KS}"
                + ("" if not wrong else f"; chosen outside them: {wrong[:5]}"),
                not wrong,
            )
        )

    # A k belonging to neither family is a setting nothing declares a meaning for -- ruling 36's
    # hazard, since a chosen count is only interpretable against the set it was chosen from. It
    # gets its own claim rather than being folded into the neighbour-count one, whose text would
    # otherwise fail while naming models that are not nearest lines.
    unaccounted = sorted(
        {
            str(row["model"])
            for _, row in chosen.iterrows()
            if str(row["model"]) not in candidates and str(row["model"]) != NEAREST_LINES
        }
    )
    checks.append(
        Check(
            names[3],
            f"{len(chosen)} rows of {SETTINGS} chose a k",
            f"components from {PARAMS_JSON}'s component_ks, neighbours from design section 5"
            + (
                ""
                if not unaccounted
                else f"; models with a chosen k and no candidate set: {unaccounted}"
            ),
            not unaccounted,
        )
    )
    return checks


def check_params(task_dir: Path, cache: Path | None = None, repo: Path = REPO) -> list[Check]:
    """The parameter sidecar: the inputs it pins, the commit it names, the seeds and the
    candidate component counts it records (project rule 1, rulings 33 and 36)."""
    names = [
        "params: every pinned input matches its recorded sha256",
        "params: the recorded commit is a commit on this branch",
        "params: the candidate component counts are the design's",
        "params: the redraw base seed, draw and block counts are the declared ones",
        "params: the recorded ceiling is rung1_ceiling.csv's",
    ]
    path = task_dir / PARAMS_JSON
    params = _read_json(path)
    if params is None:
        return [skipped(name, f"{path} does not exist") for name in names]

    checks = [
        _check_pinned_inputs(params, task_dir, cache, names[0]),
        _check_recorded_commit(params, repo, names[1]),
        _check_component_ks(params, task_dir, cache, names[2]),
    ]

    seeds = dict(params.get("seeds", {}))
    recorded = {
        "redraw_base_seed": seeds.get("redraw_base_seed"),
        "n_draws": params.get("n_draws"),
        "n_blocks": params.get("n_blocks"),
        "draws_per_block": params.get("draws_per_block"),
    }
    declared = {
        "redraw_base_seed": REDRAW_BASE_SEED,
        "n_draws": N_DRAWS,
        "n_blocks": N_BLOCKS,
        "draws_per_block": DRAWS_PER_BLOCK,
    }
    checks.append(
        Check(
            names[3],
            str(recorded),
            f"ruling 33 and design section 7 declare {declared}",
            recorded == declared,
        )
    )

    ceiling = _ceiling_values(task_dir)
    recorded_ceiling = {str(k): float(v) for k, v in dict(params.get("ceiling", {})).items()}
    if not ceiling:
        checks.append(skipped(names[4], f"{task_dir / CEILING_CSV} does not exist"))
    else:
        agrees = set(recorded_ceiling) == set(ceiling) and all(
            _close(recorded_ceiling[key], ceiling[key]) for key in ceiling
        )
        checks.append(
            Check(
                names[4],
                str(recorded_ceiling),
                f"{CEILING_CSV} holds {ceiling}",
                agrees,
            )
        )
    return checks


def _check_pinned_inputs(
    params: dict[str, Any], task_dir: Path, cache: Path | None, name: str
) -> Check:
    """Every input the run recorded, against the file on disk where that file is in this tree."""
    inputs = dict(params.get("inputs", {}))
    if not inputs:
        return skipped(name, f"{PARAMS_JSON} pins no inputs")
    roots = {"out_dir": task_dir, "cache": cache}
    matched, moved, unresolved = [], [], []
    for key, digest in inputs.items():
        root_name, _, relative = str(key).partition("/")
        root = roots.get(root_name)
        path = None if root is None else root / relative
        if path is None or not path.exists():
            unresolved.append(str(key))
        elif sha256_of(path) == str(digest):
            matched.append(str(key))
        else:
            moved.append(str(key))
    if not matched and not moved:
        return skipped(
            name,
            f"none of the {len(inputs)} pinned inputs is in this tree (they live in the run's "
            "scratch cache; pass --cache)",
        )
    detail = f"{len(matched)} of {len(inputs)} recompute"
    if moved:
        detail += f"; CHANGED {sorted(moved)[:5]}"
    if unresolved:
        detail += f"; not in this tree {len(unresolved)}"
    return Check(name, f"{len(inputs)} input checksums in {PARAMS_JSON}", detail, not moved)


def _check_recorded_commit(params: dict[str, Any], repo: Path, name: str) -> Check:
    """The commit the run was made at: a real commit in this repository, and an ancestor of HEAD
    -- a run made on another branch is not evidence about this one."""
    sha = str(params.get("git_sha", ""))
    if not sha:
        return skipped(name, f"{PARAMS_JSON} records no git_sha")
    try:
        exists = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        ancestor = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, "HEAD"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return skipped(name, f"git could not be run here ({error})")
    known = exists.returncode == 0
    on_branch = known and ancestor.returncode == 0
    return Check(
        name,
        f"git_sha {sha[:12]}",
        f"a commit in this repository: {known}; an ancestor of HEAD: {on_branch}",
        on_branch,
    )


def _check_component_ks(
    params: dict[str, Any], task_dir: Path, cache: Path | None, name: str
) -> Check:
    """Ruling 36: the candidate component counts the run actually tuned over, against the
    design's {2, 5, 10, 15, 20}. A smaller grid legitimately offers fewer, and says so."""
    recorded = {
        str(model): [int(k) for k in ks] for model, ks in params.get("component_ks", {}).items()
    }
    # The size of the run decides whether this claim applies -- not whether the field the claim
    # reads happens to be there. Ruling 36 requires component_ks in the sidecar, so on the
    # design's grid its absence is the failure, and it is tested before the record is read.
    record = grid_record(task_dir, cache)
    if record is not None and not is_design_grid(record):
        return skipped(
            name,
            f"this run's grid is {grid_shape(record)}, smaller than the design's "
            f"{DESIGN_LINES} x {DESIGN_DRUGS}, where fewer candidates are legitimate "
            f"(recorded: {recorded})",
        )
    if not recorded:
        return Check(
            name,
            f"{PARAMS_JSON} records no component_ks",
            "ruling 36 requires it: the chosen k in the settings table means nothing without "
            "the candidate set it was chosen from",
            False,
        )
    models_missing = sorted(set(TUNED_MODELS) - set(recorded))
    wrong = sorted(model for model, ks in recorded.items() if ks != COMPONENT_KS)
    return Check(
        name,
        f"{len(recorded)} tuned models: {recorded}",
        f"design section 5 declares {COMPONENT_KS} for {sorted(TUNED_MODELS)}"
        + (f"; missing {models_missing}" if models_missing else "")
        + (f"; other candidates {wrong}" if wrong else ""),
        not wrong and not models_missing,
    )


# ------------------------------------------------------------------------------------------


def resolve_cache(task_dir: Path, cache: Path | None) -> Path | None:
    """Where the run's redraw blocks are: ``--cache`` if given, else the task folder (some runs
    keep them beside the tables), else the cache the parameter sidecar recorded."""
    if cache is not None:
        return cache
    if (task_dir / "redraws_0.parquet").exists():
        return task_dir
    params = _read_json(task_dir / PARAMS_JSON) or {}
    recorded = str(dict(params.get("args", {})).get("cache", ""))
    if recorded and Path(recorded).is_dir():
        return Path(recorded)
    return None


def run_all_checks(
    task_dir: Path = DEFAULT_TASK_DIR, cache: Path | None = None, repo: Path = REPO
) -> list[Check]:
    """Every claim in the battery, in the order the run makes them."""
    resolved = resolve_cache(task_dir, cache)
    return [
        *check_grid(task_dir, resolved),
        *check_grid_provenance(task_dir, resolved),
        *check_ceiling(task_dir, repo, resolved),
        *check_settings(task_dir, resolved),
        *check_model_summary(task_dir),
        *check_comparisons(task_dir, resolved),
        *check_leakage(task_dir, resolved),
        *check_figures(task_dir),
        *check_params(task_dir, resolved, repo),
    ]


NOT_CHECKABLE_LOCALLY = """
Not checkable here, stated rather than hidden:
  - The redraw blocks live in the run's scratch cache, not the task folder. Pass --cache and
    every interval, p-value, MDE and design effect above is recomputed from them; without it
    those checks say SKIP rather than passing on draws they never read.
  - The answers, descriptions and embeddings are cluster-side and far too large to commit.
    What is checked is the parameter sidecar's sha256 of each input that IS in this tree.
  - Anything marked SKIP is a claim this working tree cannot answer -- a fixture grid smaller
    than the design's, a stage that has not run -- never a check that was waived.
"""


def exit_status(checks: Sequence[Check], *, design_grid: bool) -> int:
    """The battery's exit code: 0 only when it actually verified this run.

    Three ways to earn a non-zero status, and the last two are the ones a note to a human would
    not have caught:

    * **a check failed** -- a document and an artifact disagree;
    * **nothing ran.** Every check skipped is green continuous integration over an unverified
      run: a task folder holding one table would otherwise print "0 / 0 checks pass" and exit 0;
    * **the redraw family was skipped on the design's own grid.** The intervals, p-values,
      minimum detectable effects and design effects are what a promotion rests on, and the
      realistic way to lose them is a purged scratch cache -- which must not pass silently. On a
      fixture grid the same skip is legitimate and does not fail the battery.
    """
    if any(not check.ok for check in checks):
        return 1
    if not any(not check.skipped for check in checks):
        return 1
    if design_grid and any(check.skipped and check.group == REDRAW_GROUP for check in checks):
        return 1
    return 0


def render(checks: list[Check]) -> str:
    lines: list[str] = []
    for check in checks:
        mark = "SKIP" if check.skipped else ("PASS" if check.ok else "FAIL")
        lines.append(f"[{mark}] {check.name}")
        lines.append(f"       claim:      {check.claim}")
        lines.append(f"       recomputed: {check.computed}")
    n_skipped = sum(check.skipped for check in checks)
    n_ok = sum(check.ok and not check.skipped for check in checks)
    n_run = len(checks) - n_skipped
    lines.append(f"\n{n_ok} / {n_run} checks pass ({n_skipped} skipped, {len(checks)} total)")
    lines.append(NOT_CHECKABLE_LOCALLY)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task-dir",
        type=Path,
        default=DEFAULT_TASK_DIR,
        help=f"directory holding one run's artifacts (default: docs/tasks/{TASK})",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="the run's scratch cache, holding redraws_{b}.parquet (default: the task folder, "
        "or the cache the parameter sidecar recorded)",
    )
    ns = ap.parse_args(argv)
    task_dir = ns.task_dir if ns.task_dir.is_absolute() else REPO / ns.task_dir
    if not (task_dir / MODEL_SUMMARY).exists():
        print(f"no run to verify: {task_dir / MODEL_SUMMARY} does not exist.")
        print("Run the fit, redraw and combine stages first, or pass --task-dir.")
        return 2
    resolved = resolve_cache(task_dir, ns.cache)
    record = grid_record(task_dir, resolved)
    checks = run_all_checks(task_dir, cache=ns.cache)
    print(render(checks))
    status = exit_status(checks, design_grid=record is None or is_design_grid(record))
    if status != 0 and all(check.ok for check in checks):
        print(
            "FAILING anyway: the battery did not verify what it was asked to. Every check above "
            "that says SKIP is a claim nothing here answered -- pass --cache if the run's "
            "redraw blocks are on scratch."
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
