"""The rung 1 grid, drug-name crosswalk, leakage pairs, ceiling table, and restriction record.

Design.md sections 2, 6 and 7 fix these exactly: 107 drugs x 50 lines = 5,350 pairs at 5 uM,
two pairs removed for sci-Plex leakage, and a ceiling read (not recomputed) from rung 0's
promoted per-pair and dose-strata tables. Tests below run the real functions against those
committed tables and the offline drug-metadata fixture, so the numbers here are the numbers
a reader can reproduce without any cluster access.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from fmharness.heldout.grid import (
    Grid,
    attach_drug_metadata,
    ceiling_table,
    grid_from_rung0,
    load_drug_metadata,
    restriction_record,
    sciplex_exposed_pairs,
)
from fmharness.heldout.records import sha256_file

REPO = Path(__file__).resolve().parents[1]
PER_PAIR_PATH = REPO / "results/rung0-assay-reliability/rung0_per_pair_r.csv"
DOSE_STRATA_PATH = REPO / "results/rung0-assay-reliability/rung0_dose_strata.csv"
DRUG_METADATA_PATH = REPO / "tests/fixtures/tahoe_drug_metadata.csv"


def _read_per_pair(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _read_dose_strata(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def per_pair() -> pd.DataFrame:
    return _read_per_pair(PER_PAIR_PATH)


@pytest.fixture(scope="module")
def dose_strata() -> pd.DataFrame:
    return _read_dose_strata(DOSE_STRATA_PATH)


@pytest.fixture(scope="module")
def drug_metadata() -> pd.DataFrame:
    return load_drug_metadata(DRUG_METADATA_PATH)


@pytest.fixture(scope="module")
def grid(per_pair: pd.DataFrame) -> Grid:
    return grid_from_rung0(per_pair)


@pytest.mark.step_build
def test_grid_from_rung0_matches_the_promoted_grid(grid: Grid) -> None:
    assert len(grid.lines) == 50
    assert len(grid.drugs) == 107
    assert len(grid.lines) * len(grid.drugs) == 5350
    assert "NA" in grid.lines
    assert "ACH-000681" in grid.lines
    assert "L-Thyroxine (sodium salt pentahydrate)" not in grid.drugs


@pytest.mark.step_build
def test_grid_from_rung0_raises_on_a_synthetic_gap() -> None:
    # Line "A" gets one extra (duplicate) row for drug "Y" -- its set of covered lines still
    # equals the full line set (duplicates don't change *which* lines are covered), so the
    # nunique-based drug filter alone would not catch this; the row-count check must.
    per_pair = pd.DataFrame(
        {
            "patient": ["A", "B", "C", "A", "A", "B", "C"],
            "drug": ["X", "X", "X", "Y", "Y", "Y", "Y"],
            "dose": ["5.0"] * 7,
        }
    )
    with pytest.raises(ValueError, match="not complete"):
        grid_from_rung0(per_pair)


@pytest.mark.step_build
def test_attach_drug_metadata_maps_selinexor_despite_trailing_space(
    grid: Grid, drug_metadata: pd.DataFrame
) -> None:
    assert "Selinexor " in grid.drugs
    attached = attach_drug_metadata(grid, drug_metadata)
    assert attached.metadata_name["Selinexor "] == "Selinexor"
    # every grid drug got a match
    assert set(attached.metadata_name) == set(grid.drugs)


@pytest.mark.step_build
def test_attach_drug_metadata_raises_on_an_unmatched_drug(drug_metadata: pd.DataFrame) -> None:
    grid = Grid(lines=("A",), drugs=("Not A Real Drug",), metadata_name={}, excluded_pairs=())
    with pytest.raises(ValueError, match="Not A Real Drug"):
        attach_drug_metadata(grid, drug_metadata)


@pytest.mark.step_build
def test_sciplex_exposed_pairs_are_exactly_the_two_a549_pairs(
    grid: Grid, drug_metadata: pd.DataFrame
) -> None:
    exposed = sciplex_exposed_pairs(grid, drug_metadata)
    assert exposed == (
        ("ACH-000681", "Temsirolimus"),
        ("ACH-000681", "Trametinib"),
    )


@pytest.mark.step_build
def test_sciplex_exposed_pairs_empty_when_line_not_in_grid(drug_metadata: pd.DataFrame) -> None:
    grid = Grid(lines=("some-other-line",), drugs=(), metadata_name={}, excluded_pairs=())
    assert sciplex_exposed_pairs(grid, drug_metadata) == ()


@pytest.mark.known_answer
@pytest.mark.step_build
def test_ceiling_table_matches_design(
    per_pair: pd.DataFrame, grid: Grid, dose_strata: pd.DataFrame
) -> None:
    table = ceiling_table(per_pair, grid, dose_strata).set_index("gene_set")
    tol = 5e-5

    responding = table.loc["responding"]
    assert responding["pairs_scored_rung0"] == 4593
    assert responding["split_half_r"] == pytest.approx(0.5815, abs=tol)
    assert responding["sb"] == pytest.approx(0.7353, abs=tol)
    assert responding["sqrt_sb"] == pytest.approx(0.8575, abs=tol)
    assert responding["promoted_r"] == pytest.approx(0.5768, abs=tol)
    assert responding["promoted_sb"] == pytest.approx(0.7316, abs=tol)
    assert responding["promoted_sqrt_sb"] == pytest.approx(0.8553, abs=tol)

    all_genes = table.loc["all"]
    assert all_genes["pairs_scored_rung0"] == 5350
    assert all_genes["split_half_r"] == pytest.approx(0.0812, abs=tol)
    assert all_genes["sb"] == pytest.approx(0.1503, abs=tol)
    assert all_genes["sqrt_sb"] == pytest.approx(0.3876, abs=tol)
    assert all_genes["promoted_r"] == pytest.approx(0.0808, abs=tol)
    assert all_genes["promoted_sb"] == pytest.approx(0.1494, abs=tol)
    assert all_genes["promoted_sqrt_sb"] == pytest.approx(0.3865, abs=tol)


@pytest.mark.step_build
def test_restriction_record_hashes_and_sources(
    grid: Grid, drug_metadata: pd.DataFrame, tmp_path: Path
) -> None:
    attached = attach_drug_metadata(grid, drug_metadata)
    exposed = sciplex_exposed_pairs(attached, drug_metadata)
    from dataclasses import replace

    final_grid = replace(attached, excluded_pairs=exposed)

    sources = {"per_pair": PER_PAIR_PATH, "dose_strata": DOSE_STRATA_PATH}
    record = restriction_record(final_grid, sources)

    assert record["dose"] == 5.0
    assert record["lines"] == sorted(final_grid.lines)
    assert record["drugs"] == sorted(final_grid.drugs)
    assert record["excluded_pairs"] == [
        ["ACH-000681", "Temsirolimus"],
        ["ACH-000681", "Trametinib"],
    ]
    source_sha256 = cast("dict[str, str]", record["source_sha256"])
    assert source_sha256["per_pair"] == sha256_file(PER_PAIR_PATH)
    assert source_sha256["dose_strata"] == sha256_file(DOSE_STRATA_PATH)

    import hashlib

    expected_sha256_lines = hashlib.sha256(
        "\n".join(sorted(final_grid.lines)).encode("utf-8")
    ).hexdigest()
    assert record["sha256_lines"] == expected_sha256_lines


@pytest.mark.step_build
def test_cli_writes_grid_and_ceiling_and_skips_on_rerun(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(REPO / "scripts/heldout_grid.py"),
        "--per-pair",
        str(PER_PAIR_PATH),
        "--dose-strata",
        str(DOSE_STRATA_PATH),
        "--drug-metadata",
        str(DRUG_METADATA_PATH),
        "--out-dir",
        str(out_dir),
    ]
    first = subprocess.run(cmd, capture_output=True, text=True, check=True)
    grid_path = out_dir / "rung1_grid.json"
    ceiling_path = out_dir / "rung1_ceiling.csv"
    assert grid_path.exists()
    assert ceiling_path.exists()
    assert (out_dir / "rung1_grid.json.done.json").exists()
    assert (out_dir / "rung1_ceiling.csv.done.json").exists()

    grid_json = json.loads(grid_path.read_text())
    assert len(grid_json["lines"]) == 50
    assert len(grid_json["drugs"]) == 107
    assert grid_json["excluded_pairs"] == [
        ["ACH-000681", "Temsirolimus"],
        ["ACH-000681", "Trametinib"],
    ]
    assert grid_json["metadata_name"]["Selinexor "] == "Selinexor"

    ceiling = pd.read_csv(ceiling_path).set_index("gene_set")
    assert ceiling.loc["responding", "sqrt_sb"] == pytest.approx(0.8575, abs=5e-5)
    assert ceiling.loc["all", "sqrt_sb"] == pytest.approx(0.3876, abs=5e-5)

    grid_mtime_before = grid_path.stat().st_mtime_ns
    second = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "skipping" in second.stdout
    assert grid_path.stat().st_mtime_ns == grid_mtime_before
    assert first.returncode == 0
    assert second.returncode == 0
