"""Security-boundary regression tests for scripts/alpine/ralpine.

These are pure TEXT-SCAN tests over the script's source -- no ssh, no execution, nothing
remote. That makes them the weaker-but-cheap half of enforcement: they can only catch a
regression that is visible in the script's literal text (an escape-capable command readmitted
to READ_ONLY, a validation line moved after the remote call it was supposed to guard), not a
logic bug that keeps the text looking right while behaving wrong. They exist because the READ_
ONLY allowlist and the log/jobinfo fixed-command paths are exactly the kind of thing that drifts
silently in a script nobody runs through a test suite -- `find` was readmitted once already
(closed 2026-08-28) and this file exists to make the next regression fail loudly instead.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

RALPINE = (Path(__file__).resolve().parents[1] / "scripts" / "alpine" / "ralpine").read_text()

# Commands that would let `ralpine run <cmd>` escape the read-only boundary: `find`/`scontrol`
# write or destroy through predicates/subcommands (closed 2026-08-28); the rest are shells or
# interpreters that would admit arbitrary code by construction if ever added to READ_ONLY.
ESCAPE_CAPABLE = (
    "find",
    "scontrol",
    "file",
    "nvidia-smi",
    "git",
    "bash",
    "sh",
    "xargs",
    "python",
    "perl",
)


def _read_only_tokens() -> list[str]:
    match = re.search(r"READ_ONLY=\(\n(.*?)\n\)", RALPINE, re.S)
    assert match is not None, "could not find the READ_ONLY array literal in ralpine"
    return match.group(1).split()


def _case_block(verb: str) -> str:
    """The body of one `case` arm, from `  <verb>)` to its closing `    ;;`.

    Only matches ralpine's multi-line case-arm style (`log`, `jobinfo`, `switch`, ...), not the
    single-line arms (`ls)      remote ls ... ;;`) -- the verbs this file inspects are all
    multi-line.
    """
    match = re.search(rf"\n  {re.escape(verb)}\)\n(.*?)\n    ;;\n", RALPINE, re.S)
    assert match is not None, f"could not find a multi-line '{verb})' case block in ralpine"
    return match.group(1)


def test_read_only_allowlist_excludes_every_escape_capable_command() -> None:
    tokens = _read_only_tokens()
    present = [cmd for cmd in ESCAPE_CAPABLE if cmd in tokens]
    assert not present, (
        f"escape-capable command(s) {present} are in READ_ONLY; remote()'s pass-through would "
        "let `ralpine run <cmd> ...` execute arbitrary remote commands through them"
    )


# `remote_fixed "` (space, then an opening quote) targets the actual CALL SITE -- every
# invocation in this script passes a double-quoted string argument -- not the bare word
# "remote_fixed" as it also appears in this file's own prose comments, which would give a
# false-early index and defeat the ordering check below.
_REMOTE_FIXED_CALL = 'remote_fixed "'


def test_log_verb_validates_the_pattern_before_the_remote_fixed_call() -> None:
    block = _case_block("log")
    validation_idx = block.find(r"^[A-Za-z0-9._-]*$")
    remote_fixed_idx = block.find(_REMOTE_FIXED_CALL)
    assert validation_idx != -1, "log's pattern-validation regex is missing from the case block"
    assert remote_fixed_idx != -1, "log no longer calls remote_fixed"
    assert validation_idx < remote_fixed_idx, (
        "log's pattern validation must run BEFORE the remote_fixed call it guards -- an "
        "unvalidated pattern would be spliced into a remote find invocation"
    )


def test_jobinfo_verb_validates_the_job_id_before_the_remote_fixed_call() -> None:
    block = _case_block("jobinfo")
    validation_idx = block.find(r"^[0-9]+$")
    remote_fixed_idx = block.find(_REMOTE_FIXED_CALL)
    assert validation_idx != -1, "jobinfo's numeric-job-id validation regex is missing"
    assert remote_fixed_idx != -1, "jobinfo no longer calls remote_fixed"
    assert validation_idx < remote_fixed_idx, (
        "jobinfo's job-id validation must run BEFORE the remote_fixed call it guards -- an "
        "unvalidated job id would be spliced into a remote scontrol invocation"
    )
    assert "scontrol show job" in block, "jobinfo must invoke the read-only 'scontrol show job'"


def test_remote_enforces_the_allowlist_and_metacharacter_rejection() -> None:
    match = re.search(r"\nremote\(\) \{(.*?)\n\}\n", RALPINE, re.S)
    assert match is not None, "could not find the remote() function body in ralpine"
    body = match.group(1)
    assert "is_read_only" in body, "remote() must check its command against READ_ONLY"
    assert "reject_metacharacters" in body, "remote() must reject shell metacharacters"


def _move_aside_cmd_body() -> str:
    """The body of the shared `move_aside_cmd` helper `update` and `switch` both call."""
    match = re.search(r"\nmove_aside_cmd\(\) \{(.*?)\n\}\n", RALPINE, re.S)
    assert match is not None, "could not find the move_aside_cmd() function body in ralpine"
    return match.group(1)


def _rendered_move_aside(range_arg: str) -> str:
    """Render move_aside_cmd's output for a given diff range, without touching the network,
    the filesystem outside /tmp, or any git state -- this only evaluates the function body
    (a single `echo` of a string) captured from the script's own source above.
    """
    body = _move_aside_cmd_body()
    script = f"move_aside_cmd() {{{body}\n}}\nmove_aside_cmd {range_arg!r}\n"
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True, timeout=5
    )
    return result.stdout.strip()


def test_move_aside_cmd_never_deletes_and_only_moves() -> None:
    rendered = _rendered_move_aside("HEAD..@{u}")
    assert "rm " not in rendered and not rendered.startswith("rm"), (
        "move_aside_cmd must never delete -- only mv into _moved_aside/"
    )
    assert "mv --" in rendered, "move_aside_cmd must move files aside with `mv --`"
    assert "_moved_aside/" in rendered, "move_aside_cmd must move into _moved_aside/<UTC time>/"
    assert "git ls-files --error-unmatch" in rendered, (
        "move_aside_cmd must test 'untracked' the same way update always has"
    )


def test_update_move_aside_uses_the_head_dotdot_upstream_range() -> None:
    block = _case_block("update")
    assert 'move_aside_cmd "HEAD..@{u}"' in block, (
        "update must call move_aside_cmd with the HEAD..@{u} range -- its behaviour must not "
        "change when the move-aside logic is shared with switch"
    )
    rendered = _rendered_move_aside("HEAD..@{u}")
    assert "git diff --name-only --diff-filter=A HEAD..@{u}" in rendered


def test_switch_move_aside_runs_before_switch_uses_head_origin_branch_range_and_validates() -> None:
    block = _case_block("switch")

    # Branch-name validation must still run, and must still run before anything is spliced
    # into a remote command.
    validation_idx = block.find(r"^[A-Za-z0-9._/-]+$")
    assert validation_idx != -1, "switch's branch-name validation regex is missing"

    move_aside_idx = block.find('move_aside_cmd "HEAD origin/')
    switch_idx = block.find("git switch --guess")
    assert move_aside_idx != -1, "switch must call move_aside_cmd before `git switch --guess`"
    assert switch_idx != -1, "switch must still call `git switch --guess`"
    assert validation_idx < move_aside_idx < switch_idx, (
        "switch must validate the branch name, then move aside, then switch -- in that order"
    )

    # The range passed to move_aside_cmd must compare HEAD directly against origin/<branch>,
    # not rely on @{u} (which names switch's target, not its source, and the two branches may
    # have diverged histories).
    assert 'move_aside_cmd "HEAD origin/' in block, (
        "switch must call move_aside_cmd with a HEAD origin/<branch> range, not @{u}"
    )
    rendered = _rendered_move_aside("HEAD origin/some-branch")
    assert "git diff --name-only --diff-filter=A HEAD origin/some-branch" in rendered

    assert "rm " not in block and "rm -" not in block, "switch must never delete, only move aside"

    # The branch is still validated and still spliced in only via printf %q.
    assert "printf '%q'" in block, "switch must still quote the branch name with printf %q"


def test_switch_and_update_share_one_move_aside_helper_not_a_duplicated_loop() -> None:
    update_block = _case_block("update")
    switch_block = _case_block("switch")
    assert "move_aside_cmd" in update_block
    assert "move_aside_cmd" in switch_block
    # Neither case arm should re-implement the loop inline -- that logic lives once, in
    # move_aside_cmd, and is only invoked (not duplicated) from each case arm.
    for block, verb in ((update_block, "update"), (switch_block, "switch")):
        assert "git ls-files --error-unmatch" not in block, (
            f"{verb} must call move_aside_cmd rather than duplicate its loop inline"
        )


def test_branch_name_with_shell_metacharacter_or_space_is_refused() -> None:
    block = _case_block("switch")
    validation_match = re.search(r'\[\[ "\$1" =~ (\S+) \]\]', block)
    assert validation_match is not None, "could not find switch's branch-name validation regex"
    # The regex as written (^...$) already anchors the whole string.
    anchored = re.compile(validation_match.group(1))
    for bad in ("main; rm -rf /", "main`whoami`", "main && echo pwned", "has space", "$(id)"):
        assert not anchored.fullmatch(bad), (
            f"switch's branch-name regex must reject {bad!r} (shell metacharacter or space)"
        )
    for good in ("main", "rung1-held-out-prediction-work", "feature/foo.bar"):
        assert anchored.fullmatch(good), f"switch's branch-name regex must accept {good!r}"
