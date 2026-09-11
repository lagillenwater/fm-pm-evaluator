"""Task 11 part A: rung 1's data-stage job scripts and chain submitter, checked as text.

No cluster involved -- every check reads the sbatch and shell files as text and cross-checks
them against rung1_env.sh's array-size constants, PROCESS §2's memory-per-core rule, and each
other. The one exception is ``test_smoke_import_runs_clean``, which actually runs
``scripts/heldout_smoke_import.py`` locally.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALPINE_DIR = REPO / "scripts" / "alpine"
ENV_SH = ALPINE_DIR / "rung1_env.sh"
CHAIN_SH = ALPINE_DIR / "submit_rung1_chain.sh"

DATA_SBATCH = {
    "rung1_grid.sbatch": {"cores": 8, "mem_gb": 30, "partition": "acpu", "qos": "cpu-normal"},
    "rung1_answers.sbatch": {
        "cores": 16,
        "mem_gb": 61,
        "partition": "acpu",
        "qos": "cpu-normal",
        "array": "0-7",
    },
    "rung1_answers_combine.sbatch": {
        "cores": 16,
        "mem_gb": 61,
        "partition": "acpu",
        "qos": "cpu-normal",
    },
    "rung1_dmso_cells.sbatch": {
        "cores": 4,
        "mem_gb": 15,
        "partition": "acpu",
        "qos": "cpu-normal",
        "array": "0-63%4",
    },
    "rung1_dmso_combine.sbatch": {
        "cores": 8,
        "mem_gb": 30,
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
        "mem_gb": 30,
        "partition": "acpu",
        "qos": "cpu-normal",
    },
}

MEM_PER_CORE_GB = 3.84


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


@pytest.mark.parametrize("name", sorted(DATA_SBATCH))
def test_data_stage_sbatch_exists_and_sources_env(name: str) -> None:
    text = _read(name)
    assert "source scripts/alpine/rung1_env.sh" in text
    assert 'REPO="${REPO:-$SLURM_SUBMIT_DIR}"' in text
    assert "set -euo pipefail" in text
    assert text.startswith("#!/bin/bash")


@pytest.mark.parametrize("name", sorted(DATA_SBATCH))
def test_data_stage_calls_a_script_that_exists(name: str) -> None:
    text = _read(name)
    calls = _script_calls(text)
    assert calls, f"{name} does not appear to call any scripts/*.py file"
    for call in calls:
        assert (REPO / call).exists(), f"{name} calls {call}, which does not exist"


@pytest.mark.parametrize("name", sorted(DATA_SBATCH))
def test_partition_and_qos(name: str) -> None:
    text = _read(name)
    spec = DATA_SBATCH[name]
    assert _sbatch_directive(text, "partition") == spec["partition"]
    assert _sbatch_directive(text, "qos") == spec["qos"]


@pytest.mark.parametrize(
    "name", sorted(n for n, spec in DATA_SBATCH.items() if not spec.get("gpu"))
)
def test_cpu_job_memory_is_within_its_core_budget(name: str) -> None:
    text = _read(name)
    spec = DATA_SBATCH[name]
    cores = int(_sbatch_directive(text, "cpus-per-task"))
    assert cores == spec["cores"]
    mem = _sbatch_directive(text, "mem")
    assert mem is not None and mem.endswith("G")
    mem_gb = int(mem[:-1])
    assert mem_gb == spec["mem_gb"]
    assert mem_gb <= cores * MEM_PER_CORE_GB, (
        f"{name}: --mem={mem_gb}G exceeds {cores} cores x {MEM_PER_CORE_GB}G/core"
    )


@pytest.mark.parametrize("name", sorted(n for n, spec in DATA_SBATCH.items() if "array" in spec))
def test_array_ranges_match_env_constants(name: str) -> None:
    text = _read(name)
    spec = DATA_SBATCH[name]
    assert _sbatch_directive(text, "array") == spec["array"]


def test_gpu_job_has_gres_and_expected_cores() -> None:
    text = _read("rung1_embed.sbatch")
    assert _sbatch_directive(text, "gres") == "gpu:h200:1"
    cores = int(_sbatch_directive(text, "cpus-per-task"))
    assert cores == DATA_SBATCH["rung1_embed.sbatch"]["cores"]


def test_embed_job_sets_all_three_stack_versions() -> None:
    text = _read("rung1_embed.sbatch")
    for version in ("base", "cytokine", "drug"):
        assert f"VERSION={version}" in text
    assert "--base-checkpoint" in text


def test_logs_use_expected_output_patterns() -> None:
    for name, spec in DATA_SBATCH.items():
        text = _read(name)
        if "array" in spec:
            assert "logs/%x-%A_%a.out" in text
        else:
            assert "logs/%x-%j.out" in text


def test_every_data_stage_prints_a_resolved_line() -> None:
    for name in DATA_SBATCH:
        text = _read(name)
        assert 'echo "Resolved:' in text
        assert 'git -C "$REPO" rev-parse --short HEAD' in text


def test_chain_script_exists_and_submits_every_data_stage_sbatch() -> None:
    assert CHAIN_SH.exists()
    text = CHAIN_SH.read_text()
    for name in DATA_SBATCH:
        assert f"scripts/alpine/{name}" in text, f"chain script never submits {name}"


def test_chain_script_checks_a_dependency_for_every_submission_after_the_first() -> None:
    text = CHAIN_SH.read_text()
    assert text.count('"$RALPINE" submit') >= len(DATA_SBATCH)
    assert text.count("check_dependency") >= len(DATA_SBATCH) - 1


def test_chain_script_takes_stage_data_and_an_optional_from() -> None:
    text = CHAIN_SH.read_text()
    assert "--stage" in text
    assert '"$STAGE" != "data"' in text
    assert "--from" in text


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
