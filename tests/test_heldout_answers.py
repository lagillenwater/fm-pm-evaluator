"""The rung 1 answer: mean log2FoldChange over a pair's plates, responder calls, and the
line crosswalk (task 3).

Every test runs the REAL functions from ``fmharness.heldout.answers`` and
``scripts/heldout_answers.py`` on a synthetic pool in the DE table's own shape, with planted,
known answers -- not reimplementations of their logic.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on. This file
# also chains boolean-mask filtering with column selection (``df[mask]["col"].iloc[0]``),
# which pyright's partial pandas inference resolves inconsistently (sometimes as ndarray, which
# has no ``.iloc``); reportAttributeAccessIssue is suppressed for the same reason.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmharness.heldout.answers import assemble_answers, scoreable
from fmharness.heldout.grid import DOSE_UM, Grid, load_grid

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# delta_reproducibility must already be in sys.modules under this name before heldout_answers
# is executed, so its `import delta_reproducibility` statement finds it there rather than
# failing to resolve scripts/ on sys.path -- exactly how tests/test_rung0_controls.py loads it.
dr = _load_module("delta_reproducibility", "scripts/delta_reproducibility.py")
ha = _load_module("heldout_answers", "scripts/heldout_answers.py")

pytestmark = pytest.mark.known_answer

GRID_LINES = ("L0", "L1", "NA")
GRID_DRUGS = ("Dax", "Selinexor ")
N_GENES = 60
GENES = tuple(f"G{i}" for i in range(N_GENES))
PLATES = ("P1", "P2")
OFF_DOSES = (1.0, 10.0)
OUT_LINE = "OUTLINE"
OUT_DRUG = "OutDrug "

CELLOSAURUS = {"L0": "CVCL_L0", "L1": "CVCL_L1", "NA": "CVCL_NA", OUT_LINE: "CVCL_OUT"}
CELL_NAME = {"L0": "Name_L0", "L1": "Name_L1", "NA": "Name_NA", OUT_LINE: "Name_OUT"}


def _row(
    line: str, drug: str, plate: str, dose: float, gene: str, lfc: float, padj: float
) -> dict[str, object]:
    return {
        "Cell_ID_DepMap": line,
        "Cell_ID_Cellosaur": CELLOSAURUS[line],
        "Cell_Name_Vevo": CELL_NAME[line],
        "drug": drug,
        "gene_name": gene,
        "log2FoldChange": lfc,
        "padj": padj,
        "plate": plate,
        "concentration": dose,
    }


def write_fixture_pool(tmp: Path) -> Path:
    """A synthetic replicate pool in the DE table's own shape, one parquet file.

    Six (line, drug) pairs cover every bullet in the brief:

    * ``(L0, "Dax")``: gene ``G0`` planted at 1.0 / 3.0 across the two plates (mean-exact),
      ``G1`` planted with ``padj`` significant on the second plate only (responding on either
      plate), plus off-dose rows (1.0, 10.0 uM) and out-of-grid line/drug rows attached to it.
    * ``(L0, "Selinexor ")``: gene ``G0`` untestable on the first plate (null log2FoldChange
      and padj there), tested on the second alone -- the mean and responder call must come
      from that plate only.
    * ``(L1, "Dax")``: exactly 50 of 60 genes responding (the scoreable boundary, true side).
    * ``(L1, "Selinexor ")``: exactly 49 of 60 genes responding (the boundary, false side).
    * ``(NA, "Dax")``: only genes G0-G39 tested, so G40-G59 are untested for this pair even
      though they are tested (by other pairs) and so belong to the genes universe --
      confirms both that the literal ``"NA"`` line is kept and that untested genes are NaN.
    * ``(NA, "Selinexor ")``: filler, all 60 genes tested, nothing planted.
    """
    rows: list[dict[str, object]] = []

    def fill_pair(
        line: str,
        drug: str,
        responders: int = 0,
        untestable_gene: str | None = None,
        missing_genes: tuple[str, ...] = (),
    ) -> None:
        for idx, gene in enumerate(GENES):
            if gene in missing_genes:
                continue
            if gene == untestable_gene:
                rows.append(_row(line, drug, "P1", DOSE_UM, gene, float("nan"), float("nan")))
                rows.append(_row(line, drug, "P2", DOSE_UM, gene, 5.0, 0.01))
                continue
            if line == "L0" and drug == "Dax" and gene == "G0":
                rows.append(_row(line, drug, "P1", DOSE_UM, gene, 1.0, 0.9))
                rows.append(_row(line, drug, "P2", DOSE_UM, gene, 3.0, 0.9))
                continue
            if line == "L0" and drug == "Dax" and gene == "G1":
                rows.append(_row(line, drug, "P1", DOSE_UM, gene, 0.2, 0.5))
                rows.append(_row(line, drug, "P2", DOSE_UM, gene, 0.2, 0.01))
                continue
            padj = 0.01 if idx < responders else 0.9
            rows.append(_row(line, drug, "P1", DOSE_UM, gene, 0.3, padj))
            rows.append(_row(line, drug, "P2", DOSE_UM, gene, 0.3, 0.9))

    fill_pair("L0", "Dax")
    fill_pair("L0", "Selinexor ", untestable_gene="G0")
    fill_pair("L1", "Dax", responders=50)
    fill_pair("L1", "Selinexor ", responders=49)
    fill_pair("NA", "Dax", missing_genes=GENES[40:])
    fill_pair("NA", "Selinexor ")

    # off-dose rows for a grid pair -- must not reach the 5 uM answer.
    for dose in OFF_DOSES:
        for plate in PLATES:
            rows.append(_row("L0", "Dax", plate, dose, "G0", 999.0, 0.0001))

    # an out-of-grid drug, at the grid dose, for a grid line -- must not reach the answer.
    for plate in PLATES:
        rows.append(_row("L0", OUT_DRUG, plate, DOSE_UM, "G0", 777.0, 0.0001))

    # an out-of-grid line, at the grid dose, for a grid drug -- must not reach the answer.
    for plate in PLATES:
        rows.append(_row(OUT_LINE, "Dax", plate, DOSE_UM, "G0", 888.0, 0.0001))

    pool_dir = tmp / "pseudobulk_differential_expression"
    pool_dir.mkdir(parents=True, exist_ok=True)
    path = pool_dir / "answers-00000-of-00001.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _grid() -> Grid:
    return Grid(lines=GRID_LINES, drugs=GRID_DRUGS, metadata_name={}, excluded_pairs=())


def _one_pass(path: Path, tmp_path: Path) -> pd.DataFrame:
    grid = _grid()
    return ha.slice_answers(
        [str(path)],
        grid.drugs,
        grid.lines,
        DOSE_UM,
        tmp_path / "duck",
        n_parts=1,
        part=0,
        memory_limit="2GB",
    )


def _sorted_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` sorted by (line, drug, gene) as plain strings, not by categorical code order --
    the columns come back dictionary-encoded (``_compact_df``), whose category order is
    first-appearance, not lexicographic."""
    keyed = df.assign(
        line=df["line"].astype(str), drug=df["drug"].astype(str), gene=df["gene"].astype(str)
    )
    return keyed.sort_values(["line", "drug", "gene"]).reset_index(drop=True)


@pytest.mark.step_build
def test_mean_over_plates_is_exact(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    rows = _one_pass(path, tmp_path)
    row = rows[(rows["line"] == "L0") & (rows["drug"] == "Dax") & (rows["gene"] == "G0")]
    assert len(row) == 1
    assert row["mean_lfc"].iloc[0] == pytest.approx(2.0)
    assert row["n_lfc"].iloc[0] == 2
    assert row["n_plates"].iloc[0] == 2


@pytest.mark.step_build
def test_one_untestable_plate_gives_the_others_value(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    rows = _one_pass(path, tmp_path)
    row = rows[(rows["line"] == "L0") & (rows["drug"] == "Selinexor ") & (rows["gene"] == "G0")]
    assert len(row) == 1
    assert row["mean_lfc"].iloc[0] == pytest.approx(5.0)
    assert row["min_padj"].iloc[0] == pytest.approx(0.01)
    assert row["n_lfc"].iloc[0] == 1, "only the testable plate's fold change is counted"
    assert row["n_plates"].iloc[0] == 2, "both plates are present, one just untestable"


@pytest.mark.step_build
def test_responding_when_either_plate_is_significant(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    rows = _one_pass(path, tmp_path)
    responding_gene = rows[
        (rows["line"] == "L0") & (rows["drug"] == "Dax") & (rows["gene"] == "G1")
    ]
    assert responding_gene["min_padj"].iloc[0] == pytest.approx(0.01)

    not_responding_gene = rows[
        (rows["line"] == "L0") & (rows["drug"] == "Dax") & (rows["gene"] == "G0")
    ]
    assert not_responding_gene["min_padj"].iloc[0] == pytest.approx(0.9)


@pytest.mark.step_build
def test_other_doses_drugs_and_lines_are_excluded(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    rows = _one_pass(path, tmp_path)

    # the off-dose rows must not have moved (L0, Dax, G0)'s planted mean of 2.0.
    row = rows[(rows["line"] == "L0") & (rows["drug"] == "Dax") & (rows["gene"] == "G0")]
    assert row["mean_lfc"].iloc[0] == pytest.approx(2.0)

    assert OUT_DRUG not in set(rows["drug"])
    assert OUT_LINE not in set(rows["line"])


@pytest.mark.step_build
def test_na_line_is_kept(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    rows = _one_pass(path, tmp_path)
    assert "NA" in set(rows["line"])
    na_dax = rows[(rows["line"] == "NA") & (rows["drug"] == "Dax")]
    assert set(na_dax["gene"]) == set(GENES[:40])


@pytest.mark.step_build
def test_answer_slices_equal_one_pass(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    grid = _grid()
    one_pass = _sorted_by_key(_one_pass(path, tmp_path))

    n_parts = 4
    slices = [
        ha.slice_answers(
            [str(path)],
            grid.drugs,
            grid.lines,
            DOSE_UM,
            tmp_path / f"duck_{part}",
            n_parts=n_parts,
            part=part,
            memory_limit="2GB",
        )
        for part in range(n_parts)
    ]
    combined = _sorted_by_key(pd.concat(slices, ignore_index=True))

    # check_exact: PROCESS section 2 requires the slice-versus-one-pass comparison to be exact.
    # assert_frame_equal's default rtol of 1e-5 would pass a slicing scheme that shifted a pair's
    # mean by 1e-6, and this is the test that certifies the 8-way answers scan.
    pd.testing.assert_frame_equal(
        combined[list(one_pass.columns)], one_pass, check_dtype=False, check_exact=True
    )


@pytest.mark.step_build
def test_untested_genes_are_nan(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    rows = _one_pass(path, tmp_path)
    grid = _grid()
    answers = assemble_answers(rows, grid)

    na_idx = answers.lines.index("NA")
    dax_idx = answers.drugs.index("Dax")
    tested_genes = set(GENES[:40])
    tested_idx = [answers.genes.index(g) for g in tested_genes]
    untested_idx = [i for i in range(len(answers.genes)) if i not in set(tested_idx)]

    assert np.all(np.isfinite(answers.delta[na_idx, dax_idx, tested_idx]))
    assert np.all(np.isnan(answers.delta[na_idx, dax_idx, untested_idx]))


@pytest.mark.step_build
def test_scoreable_boundary_49_versus_50_responding_genes(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    rows = _one_pass(path, tmp_path)
    grid = _grid()
    answers = assemble_answers(rows, grid)
    scores = scoreable(answers, excluded_pairs=(), min_genes=50)

    l1_idx = answers.lines.index("L1")
    dax_idx = answers.drugs.index("Dax")
    selinexor_idx = answers.drugs.index("Selinexor ")

    assert answers.responding[l1_idx, dax_idx, :].sum() == 50
    assert scores["responding"][l1_idx, dax_idx], "50 responding genes clears min_genes=50"
    assert scores["all"][l1_idx, dax_idx]

    assert answers.responding[l1_idx, selinexor_idx, :].sum() == 49
    assert not scores["responding"][l1_idx, selinexor_idx], (
        "49 responding genes must not clear min_genes=50"
    )
    assert scores["all"][l1_idx, selinexor_idx], "60 tested genes still clears 'all'"


@pytest.mark.step_build
def test_excluded_pairs_are_false_even_when_otherwise_scoreable(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    rows = _one_pass(path, tmp_path)
    grid = _grid()
    answers = assemble_answers(rows, grid)

    scores = scoreable(answers, excluded_pairs=(("L1", "Dax"),), min_genes=50)
    l1_idx = answers.lines.index("L1")
    dax_idx = answers.drugs.index("Dax")
    assert not scores["responding"][l1_idx, dax_idx]
    assert not scores["all"][l1_idx, dax_idx]

    # every other otherwise-scoreable pair is unaffected.
    unfiltered = scoreable(answers, excluded_pairs=(), min_genes=50)
    unaffected = unfiltered["all"].copy()
    unaffected[l1_idx, dax_idx] = False
    affected = scores["all"].copy()
    affected[l1_idx, dax_idx] = False
    assert np.array_equal(unaffected, affected)


@pytest.mark.step_build
def test_combine_refuses_with_a_missing_part(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    grid = _grid()
    cache = tmp_path / "cache"
    cache.mkdir()

    n_parts = 2
    for part in range(n_parts):
        rows = ha.slice_answers(
            [str(path)],
            grid.drugs,
            grid.lines,
            DOSE_UM,
            tmp_path / f"duck_part_{part}",
            n_parts=n_parts,
            part=part,
            memory_limit="2GB",
        )
        rows.to_parquet(cache / f"answers_{part}.parquet", index=False)
    # deliberately do not write answers_0.parquet's completion record: combine must refuse.
    from fmharness.heldout.records import write_record

    write_record(cache / "answers_1.parquet")

    with pytest.raises(SystemExit, match=r"answers_0\.parquet"):
        ha.combine_answers(cache, grid, n_parts, tmp_path / "out")


@pytest.mark.step_build
def test_combine_writes_npz_and_counts_when_every_part_is_done(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    grid = _grid()
    cache = tmp_path / "cache"
    cache.mkdir()

    from fmharness.heldout.records import write_record

    n_parts = 3
    for part in range(n_parts):
        rows = ha.slice_answers(
            [str(path)],
            grid.drugs,
            grid.lines,
            DOSE_UM,
            tmp_path / f"duck_full_{part}",
            n_parts=n_parts,
            part=part,
            memory_limit="2GB",
        )
        out = cache / f"answers_{part}.parquet"
        rows.to_parquet(out, index=False)
        write_record(out)

    out_dir = tmp_path / "out"
    ha.combine_answers(cache, grid, n_parts, out_dir)

    npz_path = cache / "answers.npz"
    assert npz_path.exists()
    assert (cache / "answers.npz.done.json").exists()
    loaded = np.load(npz_path, allow_pickle=False)
    assert set(loaded["lines"]) == set(GRID_LINES)
    assert set(loaded["drugs"]) == set(GRID_DRUGS)
    assert loaded["delta"].shape == (3, 2, loaded["genes"].size)
    assert loaded["responding"].shape == loaded["delta"].shape

    counts_path = out_dir / "rung1_answer_counts.csv"
    assert counts_path.exists()
    assert (counts_path.with_name(counts_path.name + ".done.json")).exists()
    counts = pd.read_csv(counts_path)
    assert len(counts) == len(GRID_LINES) * len(GRID_DRUGS)
    row = counts[(counts["line"] == "L1") & (counts["drug"] == "Dax")]
    assert row["n_responding"].iloc[0] == 50
    assert row["n_tested"].iloc[0] == 60


@pytest.mark.step_build
def test_crosswalk_has_one_row_per_grid_line(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    grid = _grid()
    rows = ha.crosswalk_rows(
        [str(path)], grid.lines, tmp_path / "duck", memory_limit="2GB", threads=None
    )
    table = ha.build_crosswalk(rows, grid.lines)
    assert list(table["line"]) == list(grid.lines)
    assert set(table.columns) == {"line", "cellosaurus", "cell_name"}
    for line in grid.lines:
        assert table.loc[table["line"] == line, "cellosaurus"].iloc[0] == CELLOSAURUS[line]


@pytest.mark.step_build
def test_crosswalk_raises_on_a_line_with_two_cellosaurus_ids(tmp_path: Path) -> None:
    path = write_fixture_pool(tmp_path)
    df = pd.read_parquet(path)
    # give one of L0's rows a second, conflicting Cellosaurus id.
    l0_mask = df["Cell_ID_DepMap"] == "L0"
    first_l0_index = df.index[l0_mask][0]
    df.loc[first_l0_index, "Cell_ID_Cellosaur"] = "CVCL_CONFLICT"
    df.to_parquet(path, index=False)

    grid = _grid()
    rows = ha.crosswalk_rows(
        [str(path)], grid.lines, tmp_path / "duck", memory_limit="2GB", threads=None
    )
    with pytest.raises(ValueError, match="more than one Cellosaurus id"):
        ha.build_crosswalk(rows, grid.lines)


@pytest.mark.step_build
def test_load_grid_round_trips(tmp_path: Path) -> None:
    grid = Grid(
        lines=("NA", "ACH-1"),
        drugs=("Selinexor ", "Trametinib"),
        metadata_name={"Selinexor ": "Selinexor", "Trametinib": "Trametinib"},
        excluded_pairs=(("ACH-1", "Trametinib"),),
    )
    # the same shape scripts/heldout_grid.py writes: metadata_name plus the restriction record.
    grid_json = {
        "metadata_name": grid.metadata_name,
        "dose": DOSE_UM,
        "lines": list(grid.lines),
        "drugs": list(grid.drugs),
        "excluded_pairs": [list(pair) for pair in grid.excluded_pairs],
        "sha256_lines": "irrelevant-for-this-test",
        "sha256_drugs": "irrelevant-for-this-test",
        "source_sha256": {},
    }
    path = tmp_path / "rung1_grid.json"
    path.write_text(json.dumps(grid_json))

    loaded = load_grid(path)
    assert loaded.lines == grid.lines
    assert loaded.drugs == grid.drugs
    assert loaded.metadata_name == grid.metadata_name
    assert loaded.excluded_pairs == grid.excluded_pairs
