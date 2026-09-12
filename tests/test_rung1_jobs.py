"""Task 11: rung 1's job scripts and chain submitter, checked as text.

No cluster involved -- every check reads the sbatch and shell files as text and cross-checks
them against rung1_env.sh's array-size constants, the design's own grid and block constants,
PROCESS §2's memory-per-core rule, and each other. The one exception is
``test_smoke_import_runs_clean``, which actually runs ``scripts/heldout_smoke_import.py``
locally.

Part A wrote the data stages (grid, answers, DMSO cells, embeddings, descriptions); part B adds
the fit stages (the 157-round fit array, the 8 redraw blocks, the combine) and the chain's
``--stage fit`` half.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALPINE_DIR = REPO / "scripts" / "alpine"
ENV_SH = ALPINE_DIR / "rung1_env.sh"
CHAIN_SH = ALPINE_DIR / "submit_rung1_chain.sh"
SMOKE_IMPORT_PY = REPO / "scripts" / "heldout_smoke_import.py"
FIT_PY = REPO / "scripts" / "heldout_fit.py"
COMPARISONS_PY = REPO / "src" / "fmharness" / "heldout" / "comparisons.py"

DATA_SBATCH = {
    "rung1_grid.sbatch": {
        "cores": 20,
        "mem_mb": 75 * 1024,
        "partition": "acpu",
        "qos": "cpu-normal",
    },
    "rung1_answers.sbatch": {
        "cores": 20,
        "mem_mb": 75 * 1024,
        "partition": "acpu",
        "qos": "cpu-normal",
        "array": "0-7",
    },
    "rung1_answers_combine.sbatch": {
        "cores": 16,
        "mem_mb": 60 * 1024,
        "partition": "acpu",
        "qos": "cpu-normal",
    },
    "rung1_dmso_cells.sbatch": {
        "cores": 4,
        "mem_mb": 15 * 1024,
        "partition": "acpu",
        "qos": "cpu-normal",
        "array": "0-63%4",
    },
    "rung1_dmso_combine.sbatch": {
        "cores": 8,
        "mem_mb": 30 * 1024,
        "partition": "acpu",
        "qos": "cpu-normal",
    },
    "rung1_embed.sbatch": {
        "cores": 8,
        "partition": "ah200",
        "qos": "gpu-normal",
        "array": "0-2",
        "gpu": True,
    },
    "rung1_descriptions.sbatch": {
        "cores": 8,
        "mem_mb": 30 * 1024,
        "partition": "acpu",
        "qos": "cpu-normal",
    },
}

#: Task 11 part B's fit stages. The fit array runs one round per task (50 lines then 107 drugs),
#: the redraw array one block of draws per task, and the combine gathers the whole run.
FIT_SBATCH = {
    "rung1_fit.sbatch": {
        "cores": 4,
        "mem_mb": 15 * 1024,
        "partition": "acpu",
        "qos": "cpu-normal",
        "array": "0-156",
    },
    "rung1_redraws.sbatch": {
        "cores": 2,
        "mem_mb": 2 * 3840,
        "partition": "acpu",
        "qos": "cpu-normal",
        "array": "0-7",
    },
    "rung1_combine.sbatch": {
        "cores": 4,
        "mem_mb": 15 * 1024,
        "partition": "acpu",
        "qos": "cpu-normal",
    },
}

ALL_SBATCH = {**DATA_SBATCH, **FIT_SBATCH}

#: What one fit task and the combine must hold, from task 10's measurements. A round loads the
#: answers (delta float32 ~0.96 GB, delta0 another ~0.96 GB, tested ~0.24 GB: about 2.2 GB) and
#: sizes its working blocks by ``rung1_fit.sbatch``'s own ``FIT_MAX_BYTES``, derived from that
#: job's ``--mem`` rather than from ``models.MAX_BYTES`` (see the test below); task 11B's brief
#: sizes a task at 8 GB or more. The combine holds the fit control's 50 x 107 x 300 grid several
#: times over plus 2 GiB of ridge working arrays; task 10b measured it at 8 GB.
FIT_MIN_MEM_MB = 8 * 1024

# Alpine bills memory per core at 3,840 MB/core (PROCESS §2). Slurm's --mem takes a size suffix
# where "G" means GiB (1024 MB), not 1000 MB decimal -- a request written as "61G" on 16 cores
# (61,440 MB budget) is actually 62,464 MB and 1,024 MB over, even though 61 <= 16 * 3.84 in
# decimal arithmetic. So budget comparisons here always go through _mem_to_mb, in MB, never
# through the raw numeral before its unit.
MEM_PER_CORE_MB = 3840


def _mem_to_mb(mem: str) -> int:
    """Parse a Slurm ``--mem`` value (``<N>M`` or ``<N>G``, G = 1024 M) into MB."""
    match = re.fullmatch(r"(\d+)([MG])", mem)
    assert match is not None, f"unrecognized --mem value: {mem!r}"
    value, unit = match.groups()
    return int(value) * 1024 if unit == "G" else int(value)


# PROCESS §2: a DuckDB process against the 89 GB Tahoe DE table peaks roughly 35-40 GB above
# its own --memory-limit, whatever the slice size. Job 32422188 (rung1_grid.sbatch's crosswalk
# call) OOM'd at --mem=30G because it passed no --memory-limit/--threads at all and ran on
# DuckDB's unbounded defaults with no headroom for that overhead. Any job script that hands
# heldout_answers.py a --local-dir (i.e. actually scans the DE table, as opposed to --combine,
# which only reassembles already-cached parquet slices) must therefore pin --memory-limit
# explicitly and size --mem to at least that limit plus the measured overhead.
DUCKDB_OVERHEAD_MB = 35 * 1024


def _memory_limit_gb(call_text: str) -> int | None:
    """Parse the ``--memory-limit <N>GB`` value passed alongside a heldout_answers.py call."""
    match = re.search(r"--memory-limit\s+(\d+)GB", call_text)
    return int(match.group(1)) if match is not None else None


def _heldout_answers_call(text: str) -> str | None:
    """The full (possibly multi-line, backslash-continued) heldout_answers.py invocation."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "python scripts/heldout_answers.py" in line:
            block = [line]
            j = i
            while lines[j].rstrip().endswith("\\"):
                j += 1
                block.append(lines[j])
            return "\n".join(block)
    return None


def _read(name: str) -> str:
    path = ALPINE_DIR / name
    assert path.exists(), f"missing job script: {path}"
    return path.read_text()


def _sbatch_directive(text: str, key: str) -> str:
    match = re.search(rf"^#SBATCH\s+--{re.escape(key)}=(\S+)", text, re.MULTILINE)
    assert match is not None, f"no #SBATCH --{key}= directive found"
    return match.group(1)


def _script_calls(text: str) -> list[str]:
    """Every ``scripts/heldout_*.py`` (or ``scripts/*.py``) path this job script invokes."""
    return re.findall(r"scripts/[A-Za-z0-9_./]+\.py", text)


def test_env_file_exists_and_declares_array_constants() -> None:
    assert ENV_SH.exists()
    text = ENV_SH.read_text()
    assert re.search(r'N_ANSWER_PARTS="\$\{N_ANSWER_PARTS:-8\}"', text)
    assert re.search(r'N_DMSO_BLOCKS="\$\{N_DMSO_BLOCKS:-64\}"', text)
    assert re.search(r'DMSO_CONCURRENCY="\$\{DMSO_CONCURRENCY:-4\}"', text)
    assert "PYTHONPATH" in text
    assert "PYTHONUNBUFFERED" in text
    assert "RUNG1_CACHE" in text
    assert "OUT_DIR" in text


@pytest.mark.parametrize("name", sorted(ALL_SBATCH))
def test_every_sbatch_exists_and_sources_env(name: str) -> None:
    text = _read(name)
    assert "source scripts/alpine/rung1_env.sh" in text
    assert 'REPO="${REPO:-$SLURM_SUBMIT_DIR}"' in text
    assert "set -euo pipefail" in text
    assert text.startswith("#!/bin/bash")


@pytest.mark.parametrize("name", sorted(ALL_SBATCH))
def test_every_sbatch_calls_a_script_that_exists(name: str) -> None:
    text = _read(name)
    calls = _script_calls(text)
    assert calls, f"{name} does not appear to call any scripts/*.py file"
    for call in calls:
        assert (REPO / call).exists(), f"{name} calls {call}, which does not exist"


@pytest.mark.parametrize("name", sorted(ALL_SBATCH))
def test_partition_and_qos(name: str) -> None:
    text = _read(name)
    spec = ALL_SBATCH[name]
    assert _sbatch_directive(text, "partition") == spec["partition"]
    assert _sbatch_directive(text, "qos") == spec["qos"]


@pytest.mark.parametrize("name", sorted(n for n, spec in ALL_SBATCH.items() if not spec.get("gpu")))
def test_cpu_job_memory_is_within_its_core_budget(name: str) -> None:
    text = _read(name)
    spec = ALL_SBATCH[name]
    cores = int(_sbatch_directive(text, "cpus-per-task"))
    assert cores == spec["cores"]
    mem = _sbatch_directive(text, "mem")
    mem_mb = _mem_to_mb(mem)
    assert mem_mb == spec["mem_mb"]
    assert mem_mb <= cores * MEM_PER_CORE_MB, (
        f"{name}: --mem={mem} is {mem_mb} MB, over {cores} cores x {MEM_PER_CORE_MB} MB/core"
    )


@pytest.mark.parametrize("name", sorted(ALL_SBATCH))
def test_heldout_answers_table_scan_pins_memory_limit_with_overhead_headroom(name: str) -> None:
    """Job 32422188 (rung1_grid.sbatch's crosswalk call) OOM'd at --mem=30G, MaxRSS 31.5 GB,
    because it scanned the DE table via --local-dir with no --memory-limit/--threads at all, so
    DuckDB ran on its unbounded defaults. Any job that scans the table (passes --local-dir to
    heldout_answers.py) must pin --memory-limit explicitly and size --mem to at least that limit
    plus PROCESS §2's measured ~35-40 GB overhead. --combine calls (no --local-dir, no table
    scan) are exempt.
    """
    text = _read(name)
    call = _heldout_answers_call(text)
    if call is None or "--local-dir" not in call:
        pytest.skip(f"{name} does not scan the DE table (no --local-dir)")
    limit_gb = _memory_limit_gb(call)
    assert limit_gb is not None, (
        f"{name}: heldout_answers.py scans the DE table (--local-dir) but passes no "
        "--memory-limit, so DuckDB falls back to its unbounded defaults (job 32422188's "
        "failure mode)"
    )
    mem_mb = _mem_to_mb(_sbatch_directive(text, "mem"))
    required_mb = limit_gb * 1024 + DUCKDB_OVERHEAD_MB
    assert mem_mb >= required_mb, (
        f"{name}: --mem is {mem_mb} MB, under the {limit_gb}GB engine limit plus PROCESS §2's "
        f"{DUCKDB_OVERHEAD_MB} MB overhead ({required_mb} MB required)"
    )


def test_mem_to_mb_parses_the_gib_suffix_correctly() -> None:
    # The bug this guards against: 16 cores x 3,840 MB = 61,440 MB, which is exactly 60G (GiB).
    # "61G" looks fine under decimal arithmetic (61 <= 16 * 3.84) but is actually 62,464 MB --
    # 1,024 MB over budget -- because Slurm's G suffix is GiB, not 1000 MB.
    cores = 16
    budget_mb = cores * MEM_PER_CORE_MB
    assert _mem_to_mb("61G") > budget_mb
    assert _mem_to_mb("60G") <= budget_mb
    assert _mem_to_mb("61440M") == budget_mb


def test_the_fit_job_passes_its_memory_budget_to_the_fitter() -> None:
    """A round's working-block budget must come from what THIS job asked for.

    ``scripts/heldout_fit.py`` passed nothing, so every fit ran at ``models.MAX_BYTES`` (2 GiB)
    however much memory the job held: ``ridge_lolo_k``'s tuning pass needs
    ``(5 x 13 + 2) x 49 x G x 8`` bytes per drug and raises above roughly 81,800 genes, which
    would kill all 157 array tasks at once and be fixable only by editing a constant and
    redeploying. The budget is derived here from the job's own ``--mem``, and the two numbers it
    is derived from are checked against that directive.
    """
    text = _read("rung1_fit.sbatch")
    mem_g = _mem_to_mb(_sbatch_directive(text, "mem")) // 1024
    declared = {
        key: int(value)
        for key, value in re.findall(r"^(FIT_MEM_G|FIT_ANSWERS_G)=(\d+)", text, re.MULTILINE)
    }
    assert declared.get("FIT_MEM_G") == mem_g, (
        f"FIT_MEM_G={declared.get('FIT_MEM_G')} must be this job's own --mem of {mem_g}G"
    )
    assert 0 < declared.get("FIT_ANSWERS_G", 0) < mem_g, (
        "the budget must leave the answer arrays room and still be positive"
    )
    assert re.search(
        r"FIT_MAX_BYTES=\$\(\(\s*\(FIT_MEM_G - FIT_ANSWERS_G\) \* 1024 \* 1024 \* 1024\s*\)\)",
        text,
    ), "FIT_MAX_BYTES must be derived from the two declared numbers, not written out"
    assert "scripts/heldout_fit.py" in _script_calls(text)
    assert re.search(r'--max-bytes\s+"\$FIT_MAX_BYTES"', text), (
        "heldout_fit.py must be GIVEN the budget; without the flag the fit silently uses "
        "models.MAX_BYTES and the job's --mem buys it nothing"
    )


@pytest.mark.parametrize("name", sorted(n for n, spec in ALL_SBATCH.items() if "array" in spec))
def test_array_ranges_match_env_constants(name: str) -> None:
    text = _read(name)
    spec = ALL_SBATCH[name]
    assert _sbatch_directive(text, "array") == spec["array"]


def test_gpu_job_has_gres_and_expected_cores() -> None:
    text = _read("rung1_embed.sbatch")
    assert _sbatch_directive(text, "gres") == "gpu:h200:1"
    cores = int(_sbatch_directive(text, "cpus-per-task"))
    assert cores == ALL_SBATCH["rung1_embed.sbatch"]["cores"]


def test_embed_job_sets_all_three_stack_versions() -> None:
    text = _read("rung1_embed.sbatch")
    for version in ("base", "cytokine", "drug"):
        assert f"VERSION={version}" in text
    assert "--base-checkpoint" in text


def test_logs_use_expected_output_patterns() -> None:
    for name, spec in ALL_SBATCH.items():
        text = _read(name)
        if "array" in spec:
            assert "logs/%x-%A_%a.out" in text
        else:
            assert "logs/%x-%j.out" in text


def test_every_stage_prints_a_resolved_line() -> None:
    for name in ALL_SBATCH:
        text = _read(name)
        assert 'echo "Resolved:' in text
        assert 'git -C "$REPO" rev-parse --short HEAD' in text


def test_chain_script_exists_and_submits_every_sbatch() -> None:
    assert CHAIN_SH.exists()
    text = CHAIN_SH.read_text()
    for name in ALL_SBATCH:
        assert f"scripts/alpine/{name}" in text, f"chain script never submits {name}"


def test_chain_script_checks_a_dependency_for_every_submission_after_the_first() -> None:
    text = CHAIN_SH.read_text()
    assert text.count('"$RALPINE" submit') >= len(ALL_SBATCH)
    assert text.count("check_dependency") >= len(ALL_SBATCH) - 1


def test_chain_script_takes_both_stages_and_an_optional_from() -> None:
    text = CHAIN_SH.read_text()
    assert "--stage" in text
    assert "--from" in text
    for stage in ("data", "fit"):
        assert re.search(rf"^\s+{stage}\)", text, re.MULTILINE), (
            f"the chain script's --stage case has no {stage} arm"
        )
    for start in ("grid", "answers", "dmso", "embed", "fit", "redraws", "combine"):
        assert re.search(rf"\b{start}\b", text), f"the chain script cannot resume from {start}"


def test_chain_script_is_executable() -> None:
    """It is run as ``scripts/alpine/submit_rung1_chain.sh``, so the mode bit has to be set --
    it was committed non-executable by part A."""
    assert os.access(CHAIN_SH, os.X_OK), f"{CHAIN_SH} is not executable"


def test_smoke_import_runs_clean() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "heldout_smoke_import.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_the_smoke_import_names_every_heldout_module_and_script() -> None:
    """The check every long stage runs first only covers what it names. Modules added after it
    was written (scoring, comparisons, controls, figures, leakage) would otherwise first be
    imported hours into a cluster job, under Alpine's Python 3.10."""
    text = SMOKE_IMPORT_PY.read_text()
    named = set(re.findall(r'"(fmharness\.heldout[.\w]*)"', text))
    expected = {"fmharness.heldout"} | {
        f"fmharness.heldout.{path.stem}"
        for path in (REPO / "src" / "fmharness" / "heldout").glob("*.py")
        if path.stem != "__init__"
    }
    assert named == expected, f"the smoke import does not name {sorted(expected - named)}"

    # The runner scripts are covered by the glob rather than by name, so the glob has to be the
    # one that matches all of them -- including task 10's three.
    assert 'glob("heldout_*.py")' in text
    for script in ("heldout_fit.py", "heldout_redraws.py", "heldout_combine.py"):
        assert (REPO / "scripts" / script).exists()
        assert (REPO / "scripts" / script).match("heldout_*.py")


# ==============================================================================================
# Part B: the fit stages -- one round per array task, one block of redraws, one combine


def _int_constant(path: Path, name: str) -> int:
    """One ``NAME = <int>`` constant, read out of a Python source file as text.

    Read rather than imported: this file checks job scripts against the constants the run
    itself uses, and importing the run's modules would pull numpy and pandas into a text-only
    test for the sake of two integers.
    """
    match = re.search(rf"^{re.escape(name)}\s*=\s*(\d+)", path.read_text(), re.MULTILINE)
    assert match is not None, f"no {name} constant in {path}"
    return int(match.group(1))


def _design_grid() -> tuple[int, int]:
    """The design's grid, from ``heldout_fit.py``: 50 lines and 107 drugs, so 157 rounds."""
    match = re.search(
        r"^FULL_GRID_LINES,\s*FULL_GRID_DRUGS\s*=\s*(\d+),\s*(\d+)",
        FIT_PY.read_text(),
        re.MULTILINE,
    )
    assert match is not None, f"no FULL_GRID_LINES/FULL_GRID_DRUGS in {FIT_PY}"
    return int(match.group(1)), int(match.group(2))


def _round_for_index(index: int) -> subprocess.CompletedProcess[str]:
    """Run the fit job's OWN index-to-round mapping, cut out of the job script.

    The one piece of logic in that script is which round an array index is, so it is exercised
    rather than pattern-matched: the two constants and the function are cut out of the file with
    ``sed`` and evaluated under bash, exactly as written. ``eval``, not ``source <(...)``:
    macOS ships bash 3.2, which does not pick up a sourced process substitution under
    ``bash -c`` (measured here -- the function came back "command not found").
    """
    extract = f"sed -n '/^N_LOLO_ROUNDS=/,/^}}/p' {ALPINE_DIR / 'rung1_fit.sbatch'}"
    return subprocess.run(
        ["bash", "-c", f'eval "$({extract})"; round_for_index {index}'],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_fit_array_runs_one_task_per_round_of_the_design_grid() -> None:
    n_lines, n_drugs = _design_grid()
    text = _read("rung1_fit.sbatch")
    assert _sbatch_directive(text, "array") == f"0-{n_lines + n_drugs - 1}"
    assert f"N_LOLO_ROUNDS={n_lines}" in text
    assert f"N_LODO_ROUNDS={n_drugs}" in text


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "lolo 0"), (49, "lolo 49"), (50, "lodo 0"), (156, "lodo 106")],
)
def test_a_fit_array_index_names_its_scheme_and_round(index: int, expected: str) -> None:
    result = _round_for_index(index)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_a_fit_array_index_outside_the_grid_fails_loudly() -> None:
    """An index past the last round must stop the task. Falling through to round 0 would rewrite
    a finished round's outputs under another index's name."""
    n_lines, n_drugs = _design_grid()
    result = _round_for_index(n_lines + n_drugs)
    assert result.returncode != 0
    assert "outside" in result.stderr


def test_the_redraw_array_matches_the_declared_block_count() -> None:
    n_blocks = _int_constant(COMPARISONS_PY, "N_BLOCKS")
    assert _sbatch_directive(_read("rung1_redraws.sbatch"), "array") == f"0-{n_blocks - 1}"


@pytest.mark.parametrize("name", ["rung1_fit.sbatch", "rung1_combine.sbatch"])
def test_the_fit_and_combine_jobs_clear_their_measured_working_set(name: str) -> None:
    """Task 10a measured a round at about 2.2 GB of answer arrays plus models.MAX_BYTES (2 GiB)
    of working blocks; task 10b measured the combine at 8 GB. Both are sized above that floor,
    in whole cores."""
    mem_mb = _mem_to_mb(_sbatch_directive(_read(name), "mem"))
    assert mem_mb >= FIT_MIN_MEM_MB, f"{name}: --mem is {mem_mb} MB, under {FIT_MIN_MEM_MB} MB"


@pytest.mark.parametrize(
    ("name", "script", "flags"),
    [
        ("rung1_fit.sbatch", "heldout_fit.py", ("--scheme", "--round", "--grid", "--cache")),
        ("rung1_redraws.sbatch", "heldout_redraws.py", ("--block", "--grid", "--cache")),
        (
            "rung1_combine.sbatch",
            "heldout_combine.py",
            ("--grid", "--cache", "--drug-metadata", "--out-dir"),
        ),
    ],
)
def test_each_fit_stage_calls_its_cli_with_every_required_flag(
    name: str, script: str, flags: tuple[str, ...]
) -> None:
    text = _read(name)
    assert f"scripts/{script}" in text
    for flag in flags:
        assert flag in text, f"{name} calls {script} without {flag}"


def test_the_fit_array_can_be_told_to_wait_for_the_data_stage() -> None:
    """``--stage data`` and ``--stage fit`` are separate chains, so the fit array depends on the
    answers, the embeddings and the descriptions only when it is given a job id to wait on: 157
    tasks started against a missing answers.npz would fail 157 times in seconds. The script has
    to offer that dependency, confirm it, and say plainly that it is not automatic."""
    text = CHAIN_SH.read_text()
    assert "--after)" in text, "the chain script takes no --after job id"
    assert 'AFTER_DEP="--dependency=afterok:$AFTER"' in text
    assert 'check_dependency "$FIT" "afterok:$AFTER"' in text
    assert "does NOT wait" in text, "the script does not say the two stages are separate chains"


def test_the_redraws_and_the_combine_wait_for_the_whole_array_before_them() -> None:
    """``load_pair_scores`` refuses until all 157 rounds are done, and the combine until all 8
    blocks are: each must depend on the whole array job id, never on one task of it."""
    text = CHAIN_SH.read_text()
    assert '--dependency=afterok:$FIT"' in text
    assert '--dependency=afterok:$REDRAWS"' in text
    assert "afterok:${FIT}_" not in text and "afterok:$FIT_" not in text
    assert 'check_dependency "$REDRAWS" "afterok:$FIT"' in text
    assert 'check_dependency "$COMBINE" "afterok:$REDRAWS"' in text
