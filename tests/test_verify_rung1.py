"""Task 12: rung 1's verification battery, exercised on a run the test session makes itself.

The battery (``scripts/verify_rung1.py``) recomputes every promoted claim from the run's written
artifacts alone. These tests run it on the 6-line x 8-drug staged run task 10's pipeline test
builds -- the same fixture cache, the same three command lines -- so the battery's arithmetic is
checked on every checkout, and they run it again on artifacts a test has deliberately altered,
because a battery that reports PASS on evidence somebody moved is measuring nothing.

The distinction this file exists to pin: a check that cannot run must SKIP, naming what was
absent, and must never come back PASS. Task 10's two reviews both found numbers manufactured
from silence (a dictionary default, a figure panel that could not show its hypothesis), and a
verification battery is the last place that class of defect can hide.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# One fixture builder, not two: these are the helpers task 10's pipeline test uses to write the
# miniature cache and the build-stage tables, and to run the three CLIs one command line at a
# time. A second copy here would drift from the run the battery is supposed to be checking.
from tests.test_heldout_pipeline import (
    EXCLUDED_PAIR,
    N_DRUGS,
    N_LINES,
    stage_by_stage,
    write_build_tables,
    write_fixture,
)

pytestmark = pytest.mark.step_document

REPO = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location("verify_rung1", REPO / "scripts" / "verify_rung1.py")
assert _SPEC is not None and _SPEC.loader is not None
vr: Any = importlib.util.module_from_spec(_SPEC)
sys.modules["verify_rung1"] = vr
_SPEC.loader.exec_module(vr)

#: Rung 0's promoted per-pair table, the one the ceiling is read from (design.md section 6).
RUNG0_PER_PAIR = REPO / "results" / "rung0-assay-reliability" / "rung0_per_pair_r.csv"

#: Design section 6's ceilings, and the pair counts behind them.
DESIGN_CEILING = {"responding": 0.8575, "all": 0.3876}
DESIGN_CEILING_PAIRS = {"responding": 4593, "all": 5350}

#: A value moved this far from the one the artifacts imply is unmistakably a different number,
#: not a rounding difference: every reported statistic here is a correlation in [-1, 1].
PERTURBATION = 0.123


def _failures(checks: list[Any]) -> list[str]:
    return [f"{c.name}: claim {c.claim!r} vs recomputed {c.computed!r}" for c in checks if not c.ok]


def _named(checks: list[Any], fragment: str) -> list[Any]:
    return [c for c in checks if fragment in str(c.name)]


@pytest.fixture(scope="session")
def fixture_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One staged run of the real CLIs on the fixture grid: ``(out_dir, cache)``.

    Rounds, redraw blocks and the combine, each through its own command line -- the same path
    the cluster takes, so what the battery reads is what a run writes.
    """
    root = tmp_path_factory.mktemp("rung1_verify")
    cache, out_dir = root / "cache", root / "out"
    grid_path = write_fixture(cache)
    write_build_tables(out_dir)
    stage_by_stage(cache, grid_path, out_dir)
    return out_dir, cache


@pytest.fixture
def artifacts(tmp_path: Path, fixture_run: tuple[Path, Path]) -> tuple[Path, Path]:
    """A private copy of that run, so a test may perturb it without harming the others."""
    out_dir, cache = fixture_run
    private_out, private_cache = tmp_path / "out", tmp_path / "cache"
    shutil.copytree(out_dir, private_out)
    shutil.copytree(cache, private_cache)
    return private_out, private_cache


def test_every_promoted_claim_recomputes_from_the_run(artifacts: tuple[Path, Path]) -> None:
    out_dir, cache = artifacts
    checks = vr.run_all_checks(out_dir, cache=cache)
    assert not _failures(checks), "the run and its documents disagree:\n" + "\n".join(
        _failures(checks)
    )
    assert len(checks) > 20, "the battery ran almost nothing"


def test_the_battery_covers_every_layer(artifacts: tuple[Path, Path]) -> None:
    """At least one check per layer the design names: the grid it scored, the ceiling it divides
    by, each model's score, every comparison's estimate, interval, p-value, MDE and Holm
    adjustment, the redraws behind them, the removed pairs, the leakage records, the figures,
    the settings, and the provenance the audit pins bytes with."""
    names = " | ".join(str(c.name) for c in vr.run_all_checks(*artifacts[:1], cache=artifacts[1]))
    for fragment in (
        "grid: the dose",
        "grid: sha256_lines",
        "ceiling: sb is",
        "ceiling: the declared",
        "ceiling: recomputes from rung 0",
        "model summary: one row per",
        "mean score recomputes",
        "fraction of the ceiling",
        "recompute from the redrawn units",
        "estimates recompute",
        "confidence intervals",
        "p-values recompute",
        "minimum detectable effects",
        "design effect",
        "Holm",
        "redraws: every contrast",
        "leakage: the excluded pairs",
        "leakage: one record per Stack version",
        "figures:",
        "params: every pinned input",
        "params: the recorded commit",
        "component counts",
        "settings:",
    ):
        assert fragment in names, f"no check covers {fragment!r}"


def _rewrite(path: Path, column: str, value: float) -> float:
    """Move one number in a committed table, returning what it was."""
    table = pd.read_csv(path, float_precision="round_trip")
    was = float(table.loc[0, column])
    table.loc[0, column] = value
    table.to_csv(path, index=False)
    return was


def test_a_perturbed_mean_score_fails_the_battery(artifacts: tuple[Path, Path]) -> None:
    """The `document` step's negative control: move a reported mean away from the per-pair
    scores it is the mean of, and the battery must say so by name."""
    out_dir, cache = artifacts
    was = _rewrite(out_dir / "rung1_model_summary.csv", "mean_r", PERTURBATION)
    failed = [str(c.name) for c in vr.run_all_checks(out_dir, cache=cache) if not c.ok]
    assert any("mean score recomputes" in name for name in failed), (
        f"the battery passed on a mean moved from {was} to {PERTURBATION}; failures: {failed}"
    )


def test_a_widened_model_summary_interval_fails_the_battery(artifacts: tuple[Path, Path]) -> None:
    """The model summary's interval, sd and MDE are the one reported family with no written
    draws behind them -- the combine redraws them in its own process and keeps nothing.

    The only check that ever read them was "the interval contains the mean", which a standard
    deviation ten times too large, and an interval an order of magnitude too wide, both pass.
    This moves exactly those numbers and requires the battery to say so; the final assertion
    shows the containment check still passing on the same file, which is why the values are
    recomputed rather than merely bracketed.
    """
    out_dir, cache = artifacts
    path = out_dir / "rung1_model_summary.csv"
    table = pd.read_csv(path, float_precision="round_trip")
    mean = float(table.loc[0, "mean_r"])
    table.loc[0, "sd"] = float(table.loc[0, "sd"]) * 10.0
    table.loc[0, "mde"] = float(table.loc[0, "mde"]) * 10.0
    table.loc[0, "ci_lo"] = mean - 10.0 * (mean - float(table.loc[0, "ci_lo"]))
    table.loc[0, "ci_hi"] = mean + 10.0 * (float(table.loc[0, "ci_hi"]) - mean)
    table.to_csv(path, index=False)

    checks = vr.run_all_checks(out_dir, cache=cache)
    failed = [str(c.name) for c in checks if not c.ok]
    assert any("recompute from the redrawn units" in name for name in failed), failed

    contains = _named(checks, "the interval contains the mean")
    assert contains and all(c.ok for c in contains), (
        "the containment check caught this, so it is not the gap these recomputed values fill"
    )


def test_a_perturbed_comparison_estimate_fails_the_battery(artifacts: tuple[Path, Path]) -> None:
    out_dir, cache = artifacts
    was = _rewrite(out_dir / "rung1_comparisons.csv", "estimate", PERTURBATION)
    failed = [str(c.name) for c in vr.run_all_checks(out_dir, cache=cache) if not c.ok]
    assert any("estimates recompute" in name for name in failed), (
        f"the battery passed on an estimate moved from {was} to {PERTURBATION}; failures: {failed}"
    )


def test_a_perturbed_holm_adjustment_fails_the_battery(artifacts: tuple[Path, Path]) -> None:
    out_dir, cache = artifacts
    table = pd.read_csv(out_dir / "rung1_comparisons.csv", float_precision="round_trip")
    adjusted = table.index[table["p_holm"].notna()]
    assert len(adjusted) > 0, "the fixture run adjusted nothing, so this control is vacuous"
    table.loc[adjusted[0], "p_holm"] = PERTURBATION
    table.to_csv(out_dir / "rung1_comparisons.csv", index=False)
    failed = [str(c.name) for c in vr.run_all_checks(out_dir, cache=cache) if not c.ok]
    assert any("Holm" in name for name in failed), f"failures: {failed}"


def test_a_leaked_pair_in_the_scores_fails_the_battery(artifacts: tuple[Path, Path]) -> None:
    """The pairs the grid removed must appear in no score table. A pair that crept back in is a
    model scored on data it saw, and it has to fail here rather than be averaged in."""
    out_dir, cache = artifacts
    path = out_dir / "rung1_pair_scores.csv.gz"
    scores = pd.read_csv(path, float_precision="round_trip")
    leaked = scores.iloc[[0]].copy()
    leaked["line"], leaked["drug"] = EXCLUDED_PAIR
    pd.concat([scores, leaked], ignore_index=True).to_csv(
        path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    failed = [str(c.name) for c in vr.run_all_checks(out_dir, cache=cache) if not c.ok]
    assert any("excluded pairs" in name for name in failed), f"failures: {failed}"


def test_an_altered_input_fails_its_recorded_checksum(artifacts: tuple[Path, Path]) -> None:
    """The parameter sidecar pins every input's sha256; that record is what ties what the audit
    read to what a reviewer later pulls, so an edited input must move a hash and fail."""
    out_dir, cache = artifacts
    ceiling = out_dir / "rung1_ceiling.csv"
    ceiling.write_text(ceiling.read_text() + "\n")
    failed = [str(c.name) for c in vr.run_all_checks(out_dir, cache=cache) if not c.ok]
    assert any("pinned input" in name for name in failed), f"failures: {failed}"


def test_a_missing_stand_in_contrast_fails_the_battery(artifacts: tuple[Path, Path]) -> None:
    """Design section 7's control for H1(b) is each description against its own random stand-in:
    a description that beats its stand-in gained from what it describes, not from the method. A
    run that reported none of those twelve contrasts must fail, not pass on the eleven
    hypothesis rows."""
    out_dir, cache = artifacts
    path = out_dir / "rung1_comparisons.csv"
    table = pd.read_csv(path, float_precision="round_trip")
    dropped = "lolo_expression_vs_random_expression"
    assert dropped in set(table["comparison"].astype(str)), "the fixture never ran that contrast"
    table.loc[table["comparison"] != dropped].to_csv(path, index=False)

    failed = [str(c.name) for c in vr.run_all_checks(out_dir, cache=cache) if not c.ok]
    assert any("stand-in contrast" in name for name in failed), failed


def test_a_perturbed_interval_or_mde_fails_the_battery(artifacts: tuple[Path, Path]) -> None:
    """The four statistics read off the redraws are what a promotion rests on, so each has to
    bite: move an interval bound and a minimum detectable effect away from the draws they came
    from, and the battery must name both."""
    out_dir, cache = artifacts
    path = out_dir / "rung1_comparisons.csv"
    table = pd.read_csv(path, float_precision="round_trip")
    table.loc[0, "ci_lo"] = float(table.loc[0, "ci_lo"]) - PERTURBATION
    table.loc[0, "mde"] = float(table.loc[0, "mde"]) + PERTURBATION
    table.to_csv(path, index=False)

    failed = [str(c.name) for c in vr.run_all_checks(out_dir, cache=cache) if not c.ok]
    assert any("confidence intervals" in name for name in failed), failed
    assert any("minimum detectable effects" in name for name in failed), failed


def test_a_missing_figures_directory_fails_when_the_run_is_present(
    artifacts: tuple[Path, Path],
) -> None:
    """A figure whose source table was never written is a stage that did not run; a missing
    figures directory beside a finished run is a broken figure step. Skipping says neither."""
    out_dir, cache = artifacts
    shutil.rmtree(out_dir / "figures")
    checks = vr.run_all_checks(out_dir, cache=cache)
    figures = _named(checks, "figures:")
    assert figures and not any(c.skipped for c in figures), [(c.name, c.skipped) for c in figures]
    assert any(not c.ok for c in figures), [(c.name, c.computed) for c in figures]


def test_a_missing_ceiling_table_does_not_weaken_the_fraction_claim(
    artifacts: tuple[Path, Path],
) -> None:
    """Without rung1_ceiling.csv the fraction of the ceiling can only be checked against the
    summary's own denominator, which is self-consistency, not verification. It must fail rather
    than pass on a number nothing corroborates."""
    out_dir, cache = artifacts
    (out_dir / "rung1_ceiling.csv").unlink()
    fractions = _named(vr.run_all_checks(out_dir, cache=cache), "fraction of the ceiling")
    assert fractions, "no check covers the fraction of the ceiling"
    assert not any(c.ok and not c.skipped for c in fractions), [
        (c.name, c.ok, c.skipped, c.computed) for c in fractions
    ]


def test_checks_that_cannot_run_skip_rather_than_pass(artifacts: tuple[Path, Path]) -> None:
    """With the redraw blocks gone, everything read off them -- the intervals, the p-values, the
    MDEs, the design effect -- must come back SKIP, naming what was absent. A check that passed
    here would be reporting on an input it never read."""
    out_dir, cache = artifacts
    for block in cache.glob("redraws_*.parquet*"):
        block.unlink()
    checks = vr.run_all_checks(out_dir, cache=cache)
    assert not _failures(checks), _failures(checks)

    for fragment in ("confidence intervals", "p-values recompute", "minimum detectable effects"):
        matching = _named(checks, fragment)
        assert matching, f"no check named {fragment!r}"
        assert all(c.skipped for c in matching), (
            f"{fragment!r} did not skip with its redraws absent: "
            f"{[(c.name, c.ok, c.skipped) for c in matching]}"
        )
        assert all("redraw" in str(c.computed) for c in matching), (
            f"{fragment!r} skipped without naming what was missing"
        )
    # and what does not depend on the redraws still ran
    assert any(not c.skipped and c.ok for c in _named(checks, "mean score recomputes"))


def test_a_clean_checkout_with_no_run_skips_everything(tmp_path: Path) -> None:
    """The suite has to pass on a fresh clone, where no cluster artifact exists: every check
    skips, none fails, and none silently passes."""
    checks = vr.run_all_checks(tmp_path, cache=tmp_path)
    assert checks, "an empty task directory produced no checks at all"
    assert all(c.skipped for c in checks), [
        (c.name, c.ok, c.skipped) for c in checks if not c.skipped
    ]
    assert not _failures(checks)


def test_the_committed_run_verifies_once_there_is_one() -> None:
    """When a run has been pulled into the task folder, it must pass the battery. Until then
    this skips by name rather than passing on an absence."""
    task_dir = REPO / "docs" / "tasks" / "rung1-held-out-prediction"
    if not (task_dir / "rung1_model_summary.csv").exists():
        pytest.skip("no run in the task folder yet")
    checks = vr.run_all_checks(task_dir)
    assert not _failures(checks), "the committed run and its documents disagree:\n" + "\n".join(
        _failures(checks)
    )


def test_main_exits_zero_on_a_good_run_and_non_zero_on_a_bad_one(
    artifacts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """The form continuous integration runs: claim / recomputed / verdict on stdout, and the
    exit status that makes a disagreement stop a build."""
    out_dir, cache = artifacts
    assert vr.main(["--task-dir", str(out_dir), "--cache", str(cache)]) == 0
    printed = capsys.readouterr().out
    assert "PASS" in printed and "claim" in printed and "recomputed" in printed

    _rewrite(out_dir / "rung1_model_summary.csv", "mean_r", PERTURBATION)
    assert vr.main(["--task-dir", str(out_dir), "--cache", str(cache)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_exits_two_when_there_is_no_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert vr.main(["--task-dir", str(tmp_path)]) == 2
    assert "no run to verify" in capsys.readouterr().out


def test_main_does_not_exit_zero_when_it_verified_nothing(
    tmp_path: Path, artifacts: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A folder holding only the model summary passes main's gate and then skips every check.
    Exiting 0 there is green continuous integration over an unverified run -- the realistic path
    being a purged scratch cache, where the intervals, p-values and MDEs a promotion rests on all
    skip while the exit status says pass."""
    out_dir, _ = artifacts
    shutil.copy(out_dir / "rung1_model_summary.csv", tmp_path / "rung1_model_summary.csv")
    status = vr.main(["--task-dir", str(tmp_path)])
    printed = capsys.readouterr().out
    assert status != 0, printed
    assert "0 /" in printed or "verified nothing" in printed


def _check(name: str, *, ok: bool = True, skipped: bool = False, group: str = "") -> Any:
    return vr.Check(name, "claim", "recomputed", ok, skipped=skipped, group=group)


def test_the_exit_status_refuses_a_clean_pass_it_did_not_earn() -> None:
    """The rule in code rather than in a note to a reader: a battery that ran nothing, or that
    skipped the redraw statistics on the design's own grid, has not verified the run."""
    ran = [_check("a"), _check("b", group="redraws")]
    assert vr.exit_status(ran, design_grid=True) == 0
    assert vr.exit_status([*ran, _check("c", ok=False)], design_grid=True) == 1

    nothing_ran = [_check("a", skipped=True), _check("b", skipped=True, group="redraws")]
    assert vr.exit_status(nothing_ran, design_grid=False) == 1
    assert vr.exit_status(nothing_ran, design_grid=True) == 1

    redraws_skipped = [_check("a"), _check("b", skipped=True, group="redraws")]
    assert vr.exit_status(redraws_skipped, design_grid=True) == 1
    # on a fixture grid the same skip is legitimate and does not fail the battery
    assert vr.exit_status(redraws_skipped, design_grid=False) == 0


# ==============================================================================================
# The claims the fixture's 6 x 8 grid cannot exercise: the design's own grid, and its ceiling


def _design_size_grid(directory: Path, drugs: int = 107, lines: int = 50) -> Path:
    """A grid record of the design's shape, with the hashes ``heldout_grid.py`` writes."""
    grid = {
        "dose": 5.0,
        "lines": [f"ACH-{i:06d}" for i in range(lines)],
        "drugs": [f"Drug-{j}" for j in range(drugs)],
        "metadata_name": {f"Drug-{j}": f"Drug-{j}" for j in range(drugs)},
        "excluded_pairs": [],
        "sha256_lines": "",
        "sha256_drugs": "",
        "source_sha256": {},
    }
    for key, items in (("sha256_lines", grid["lines"]), ("sha256_drugs", grid["drugs"])):
        assert isinstance(items, list)
        text = "\n".join(sorted(str(item) for item in items))
        grid[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = directory / "rung1_grid.json"
    path.write_text(json.dumps(grid, indent=2) + "\n")
    return path


def test_the_design_size_grid_claims_run_and_bite(tmp_path: Path) -> None:
    """On a grid of the design's shape the size and hash claims are checked, not skipped -- and
    a moved hash, or a dropped drug, fails them."""
    _design_size_grid(tmp_path)
    checks = vr.check_grid(tmp_path)
    size = _named(checks, "5,350 pairs")
    hashes = _named(checks, "sha256_lines") + _named(checks, "sha256_drugs")
    assert size and all(c.ok and not c.skipped for c in size), [(c.name, c.computed) for c in size]
    assert hashes and all(c.ok and not c.skipped for c in hashes)

    record = json.loads((tmp_path / "rung1_grid.json").read_text())
    record["sha256_drugs"] = "0" * 64
    record["drugs"] = record["drugs"][:-1]
    (tmp_path / "rung1_grid.json").write_text(json.dumps(record, indent=2) + "\n")
    failed = [str(c.name) for c in vr.check_grid(tmp_path) if not c.ok]
    assert any("5,350 pairs" in name for name in failed), failed
    assert any("sha256_drugs" in name for name in failed), failed


def test_the_grid_hash_claims_skip_when_the_record_holds_no_hashes(
    artifacts: tuple[Path, Path],
) -> None:
    """The fixture's grid record carries no hashes. On a grid that small that has to skip,
    naming the key, rather than pass on a comparison it never made."""
    _, cache = artifacts
    checks = vr.check_grid(cache)
    hashes = _named(checks, "sha256_lines")
    assert hashes and all(c.skipped for c in hashes)
    assert all("sha256_lines" in str(c.computed) for c in hashes)


@pytest.mark.parametrize("key", ["sha256_lines", "sha256_drugs", "source_sha256"])
def test_a_design_size_grid_missing_a_mandatory_field_fails(tmp_path: Path, key: str) -> None:
    """Those fields are mandatory in the data contract, so their absence is a defect on the
    design's own grid -- skipping there would be a check keyed on a missing input rather than on
    the size of the run."""
    _design_size_grid(tmp_path)
    record = json.loads((tmp_path / "rung1_grid.json").read_text())
    record[key] = {} if key == "source_sha256" else ""
    (tmp_path / "rung1_grid.json").write_text(json.dumps(record, indent=2) + "\n")

    checks = [c for c in vr.check_grid(tmp_path) if key in str(c.name) or key in str(c.computed)]
    assert checks, f"no check covers {key}"
    assert not any(c.skipped for c in checks), [(c.name, c.computed) for c in checks]
    assert any(not c.ok for c in checks), [(c.name, c.computed) for c in checks]


def test_a_design_size_run_without_component_ks_fails(tmp_path: Path) -> None:
    """Ruling 36 requires the realized candidate counts in the sidecar: the chosen k in the
    settings table means nothing without the set it was chosen from, so an absent record is a
    failure on the design's grid, not a skip."""
    _design_size_grid(tmp_path)
    (tmp_path / "rung1_run.params.json").write_text(
        json.dumps({"git_sha": "0" * 40, "inputs": {}, "seeds": {}}, indent=2) + "\n"
    )
    checks = _named(vr.check_params(tmp_path), "component counts")
    assert checks, "no check covers the candidate component counts"
    assert not any(c.skipped for c in checks), [(c.name, c.computed) for c in checks]
    assert any(not c.ok for c in checks), [(c.name, c.computed) for c in checks]


@pytest.mark.known_answer
def test_the_ceiling_recomputes_from_rung0s_promoted_table(tmp_path: Path) -> None:
    """Design section 6's ceilings -- responding 0.8575, all genes 0.3876 -- recomputed here
    from rung 0's promoted per-pair table restricted to rung 1's grid, through the battery's own
    check. This is the one claim rung 1 inherits rather than measures, so it is checked against
    the real table, not a fixture."""
    if not RUNG0_PER_PAIR.exists():
        pytest.skip("rung 0's promoted per-pair table is not in this checkout")
    grid = vr.grid_from_rung0_table(RUNG0_PER_PAIR)
    (tmp_path / "rung1_grid.json").write_text(json.dumps(grid, indent=2) + "\n")
    pd.DataFrame(
        [
            {
                "gene_set": gene_set,
                "pairs_scored_rung0": DESIGN_CEILING_PAIRS[gene_set],
                "split_half_r": r,
                "sb": sb,
                "sqrt_sb": DESIGN_CEILING[gene_set],
                "promoted_r": r,
                "promoted_sb": sb,
                "promoted_sqrt_sb": DESIGN_CEILING[gene_set],
            }
            for gene_set, r, sb in (("responding", 0.5815, 0.7353), ("all", 0.0812, 0.1503))
        ]
    ).to_csv(tmp_path / "rung1_ceiling.csv", index=False)

    checks = vr.check_ceiling(tmp_path, REPO)
    recomputed = _named(checks, "recomputes from rung 0")
    assert recomputed, "no check recomputes the ceiling from rung 0's table"
    assert all(c.ok and not c.skipped for c in recomputed), [
        (c.name, c.claim, c.computed) for c in recomputed
    ]
    assert all(c.ok for c in _named(checks, "the declared"))

    # and it bites: a ceiling moved off the value rung 0's table gives must fail.
    ceiling = pd.read_csv(tmp_path / "rung1_ceiling.csv")
    ceiling.loc[0, "sqrt_sb"] = PERTURBATION
    ceiling.to_csv(tmp_path / "rung1_ceiling.csv", index=False)
    failed = [str(c.name) for c in vr.check_ceiling(tmp_path, REPO) if not c.ok]
    assert any("recomputes from rung 0" in name for name in failed), failed


def _settings_only_run(directory: Path, neighbour_k: int) -> None:
    """The smallest task folder ``check_settings`` reads: a grid, a settings table, a sidecar.

    The settings table's ``k`` column carries two different quantities -- a component count for
    the tuned ridge models, a neighbour count for nearest lines -- and reading one against the
    other's candidates is exactly the confusion this fixture is here to catch.
    """
    (directory / "rung1_grid.json").write_text(
        json.dumps(
            {
                "dose": 5.0,
                "lines": ["ACH-000001"],
                "drugs": ["Drug-0"],
                "metadata_name": {"Drug-0": "Drug-0"},
                "excluded_pairs": [],
            },
            indent=2,
        )
        + "\n"
    )
    pd.DataFrame(
        [
            {"scheme": scheme, "round": 0, "model": model, "lambda": 1.0, "k": k, "loss_min": 0.1}
            for scheme in ("lolo", "lodo")
            for model, k in (("nmf", 2), ("nearest_lines", neighbour_k))
        ]
    ).assign(lambda_at_edge=False).to_csv(directory / "rung1_settings.csv", index=False)
    (directory / "rung1_run.params.json").write_text(
        json.dumps({"component_ks": {"nmf": [2, 5]}}, indent=2) + "\n"
    )


@pytest.mark.parametrize(("neighbour_k", "expected"), [(5, True), (7, False)])
def test_the_neighbour_count_claim_is_checked_against_its_own_candidates(
    tmp_path: Path, neighbour_k: int, expected: bool
) -> None:
    """Nearest lines averages k training lines, k from {3, 5, 10, 20} -- not a component count.
    A k of 7 is outside that set and must fail; a k of 5 is inside it and must pass, and neither
    may be read against the PCA/NMF candidates."""
    _settings_only_run(tmp_path, neighbour_k)
    checks = _named(vr.check_settings(tmp_path), "neighbour counts")
    assert checks, "no check covers the neighbour count"
    assert all(not c.skipped for c in checks)
    assert all(c.ok for c in checks) is expected, [(c.name, c.computed) for c in checks]

    components = _named(vr.check_settings(tmp_path), "component count is one of")
    assert components and all(c.ok and not c.skipped for c in components), [
        (c.name, c.computed) for c in components
    ]


def test_a_chosen_setting_with_no_candidate_set_fails_under_its_own_claim(tmp_path: Path) -> None:
    """A k belonging to neither family -- no recorded component candidates, and not nearest
    lines -- is a setting nothing declares a meaning for. It must fail under a claim about
    candidate sets, not under the neighbour-count claim, whose text would then be failing while
    naming models that are not nearest lines."""
    _settings_only_run(tmp_path, 5)
    (tmp_path / "rung1_run.params.json").write_text(json.dumps({"component_ks": {}}) + "\n")
    checks = vr.check_settings(tmp_path)

    unaccounted = _named(checks, "recorded candidate set")
    assert unaccounted and any(not c.ok for c in unaccounted), [
        (c.name, c.computed) for c in unaccounted
    ]
    assert any("nmf" in str(c.computed) for c in unaccounted)

    neighbours = _named(checks, "neighbour counts")
    assert neighbours and all(c.ok and not c.skipped for c in neighbours), [
        (c.name, c.computed) for c in neighbours
    ]


def test_the_fixture_run_is_the_shape_the_battery_expects(fixture_run: tuple[Path, Path]) -> None:
    """A guard on the fixture itself: the staged run really wrote the tables the battery reads,
    at the grid this file assumes, so a battery passing here is passing on a real run."""
    out_dir, cache = fixture_run
    for name in (
        "rung1_pair_scores.csv.gz",
        "rung1_model_summary.csv",
        "rung1_comparisons.csv",
        "rung1_settings.csv",
        "rung1_run.params.json",
        "rung1_leakage_profiles.json",
    ):
        assert (out_dir / name).exists(), f"the fixture run did not write {name}"
    assert len(list(cache.glob("redraws_*.parquet"))) == 8
    grid = json.loads((cache / "rung1_grid.json").read_text())
    assert (len(grid["lines"]), len(grid["drugs"])) == (N_LINES, N_DRUGS)
