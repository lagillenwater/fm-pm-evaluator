"""Security-boundary regression tests for scripts/alpine/ralpine.

Most of these are pure TEXT-SCAN tests over the script's source -- no ssh, nothing remote.
That makes them the weaker-but-cheap half of enforcement: they can only catch a regression
that is visible in the script's literal text (an escape-capable command readmitted to
READ_ONLY, a validation line moved after the remote call it was supposed to guard), not a
logic bug that keeps the text looking right while behaving wrong. They exist because the
READ_ONLY allowlist and the log/jobinfo fixed-command paths are exactly the kind of thing
that drifts silently in a script nobody runs through a test suite -- `find` was readmitted
once already (closed 2026-08-28) and this file exists to make the next regression fail loudly
instead.

A second group of tests DOES execute things, but never touches git, the network, or real
ssh:
  - a few extract one helper function's body from the script's source and run it standalone
    via `bash -c` (it only `echo`s a string; nothing it does can reach outside that subshell),
    always with `cwd` pinned to a pytest `tmp_path`, never the repository;
  - a few run `scripts/alpine/ralpine` itself as a real subprocess, but with a stub `ssh`
    (a tiny script this file writes to `tmp_path`) placed first on PATH, so no real network
    connection is ever attempted; these also run with `cwd=tmp_path`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

RALPINE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "alpine" / "ralpine"
RALPINE = RALPINE_PATH.read_text()

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


def _function_body(name: str) -> str:
    """The body of a top-level `name() { ... }` function definition in ralpine's source."""
    match = re.search(rf"\n{re.escape(name)}\(\) \{{(.*?)\n\}}\n", RALPINE, re.S)
    assert match is not None, f"could not find the {name}() function body in ralpine"
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


def _remote_fixed_arg(block: str) -> str:
    """The text of the argument passed to `remote_fixed`, from its call site to the end of the
    case arm -- i.e. the actual remote command string, as opposed to the assignment lines
    (`move_aside="$(move_aside_cmd ...)"`) that build pieces of it above the call. Ordering
    checks against this substring, not the whole case block, so that deleting the `$move_aside
    &&` splice from the remote_fixed string (while leaving the assignment line untouched above
    it) is caught: `move_aside_cmd` would still appear in the case block via the assignment,
    but `$move_aside` would no longer appear in this substring.
    """
    idx = block.find(_REMOTE_FIXED_CALL)
    assert idx != -1, "case block never calls remote_fixed"
    return block[idx + len(_REMOTE_FIXED_CALL) :]


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
    body = _function_body("remote")
    assert "is_read_only" in body, "remote() must check its command against READ_ONLY"
    assert "reject_metacharacters" in body, "remote() must reject shell metacharacters"


# ---------------------------------------------------------------------------------------------
# move_aside_cmd (update + switch) and copy_aside_cmd (switch only)
# ---------------------------------------------------------------------------------------------


def _render_function(name: str, arg: str, tmp_path: Path) -> str:
    """Run one extracted `name() { ... }` body standalone via `bash -c name <arg>`, in
    `tmp_path` (never the repository), and return its stdout. `move_aside_cmd` and
    `copy_aside_cmd` only ever `echo` a string built from their argument and literal text --
    they never touch git, the filesystem, or the network themselves (the commands they
    *produce* do, but those are never executed here) -- so this is safe to run directly.
    """
    body = _function_body(name)
    script = f"{name}() {{{body}\n}}\n{name} {arg!r}\n"
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
        cwd=tmp_path,
    )
    return result.stdout.strip()


def test_move_aside_cmd_never_deletes_only_moves_and_ignores_renames(tmp_path: Path) -> None:
    rendered = _render_function("move_aside_cmd", "HEAD..@{u}", tmp_path)
    assert "rm " not in rendered and not rendered.startswith("rm"), (
        "move_aside_cmd must never delete -- only mv into _moved_aside/"
    )
    assert "mv --" in rendered, "move_aside_cmd must move files aside with `mv --`"
    assert "$aside" in rendered, (
        "move_aside_cmd must move into the shared $aside directory (set once by ASIDE_INIT, "
        "not by move_aside_cmd itself, so switch's move- and copy-aside steps share one run)"
    )
    assert "git ls-files --error-unmatch" in rendered, (
        "move_aside_cmd must test 'untracked' the same way update always has"
    )
    assert "--no-renames" in rendered, (
        "move_aside_cmd must pass --no-renames to its diff -A -- otherwise a path git detects "
        "as a rename is reported as R, not A, and silently skipped (Ruling 20)"
    )
    assert "--diff-filter=A" in rendered


def test_move_aside_cmd_loop_failure_is_loud() -> None:
    body = _function_body("move_aside_cmd")
    assert "|| exit 1" in body, (
        "a mkdir/mv failure inside move_aside_cmd's loop must abort the remote chain (Ruling "
        "5) rather than being swallowed by the while loop"
    )


def test_aside_init_shares_one_timestamp_and_is_unexpanded_locally() -> None:
    match = re.search(r"^ASIDE_INIT='([^']*)'$", RALPINE, re.M)
    assert match is not None, "ASIDE_INIT must be a single-quoted top-level assignment"
    value = match.group(1)
    assert value.startswith("aside=_moved_aside/"), "ASIDE_INIT must set $aside"
    assert "$(date" in value, "ASIDE_INIT must build the aside dir from date -u, unexpanded"


def test_copy_aside_cmd_copies_before_checkout_never_deletes(tmp_path: Path) -> None:
    rendered = _render_function("copy_aside_cmd", "origin/some-branch", tmp_path)

    assert "rm " not in rendered and not rendered.startswith("rm"), (
        "copy_aside_cmd must never delete"
    )
    assert 'cp -p -- "$f"' in rendered, "copy_aside_cmd must copy with `cp -p` before touching $f"
    assert "git checkout HEAD -- " in rendered, (
        "copy_aside_cmd must restore the tracked file to HEAD's content"
    )
    assert 'git diff --quiet HEAD "origin/some-branch" -- "$f"' in rendered, (
        "copy_aside_cmd must compare HEAD against the given target ref for that path"
    )

    mkdir_idx = rendered.find("mkdir -p")
    cp_idx = rendered.find("cp -p")
    checkout_idx = rendered.find("git checkout HEAD")
    echo_idx = rendered.find("echo ")
    assert -1 not in (mkdir_idx, cp_idx, checkout_idx, echo_idx)
    assert mkdir_idx < cp_idx < checkout_idx < echo_idx, (
        "copy_aside_cmd must mkdir, then cp, then checkout, then echo, in that order -- a "
        "failed cp must never be followed by a checkout that discards the only copy"
    )


def test_copy_aside_cmd_loop_failure_is_loud() -> None:
    body = _function_body("copy_aside_cmd")
    assert "|| exit 1" in body, (
        "a cp/mkdir/checkout failure inside copy_aside_cmd's loop must abort the remote chain "
        "(Ruling 5), and cp specifically must be chained with && so a failed cp is never "
        "followed by `git checkout HEAD --` (which would discard the only copy)"
    )
    assert re.search(r"cp -p -- \S+ \S+ &&", body), (
        "cp must be && before the following git checkout, not run unconditionally"
    )


def test_update_remote_command_moves_aside_before_merge() -> None:
    block = _case_block("update")
    assert 'move_aside_cmd "HEAD..@{u}"' in block, (
        "update must call move_aside_cmd with the HEAD..@{u} range -- its behaviour must not "
        "change when the move-aside logic is shared with switch"
    )
    arg = _remote_fixed_arg(block)
    aside_init_idx = arg.find("$ASIDE_INIT")
    move_aside_idx = arg.find("$move_aside")
    merge_idx = arg.find("git merge --ff-only")
    assert -1 not in (aside_init_idx, move_aside_idx, merge_idx), (
        "update's remote_fixed argument must splice in $ASIDE_INIT, $move_aside, and the merge"
    )
    assert aside_init_idx < move_aside_idx < merge_idx, (
        "update must init the aside dir, then move aside, then merge -- inside the actual "
        "remote command, not just somewhere in the case block"
    )
    assert "rm " not in arg and "git clean" not in arg and "reset --hard" not in arg


def test_switch_remote_command_moves_and_copies_aside_before_switch_and_merge() -> None:
    block = _case_block("switch")

    # Branch-name validation (both the charset regex and the leading-dash guard) must still
    # run, before anything is spliced into a remote command.
    validation_idx = block.find(r"^[A-Za-z0-9._/-]+$")
    leading_dash_idx = block.find("!= -*")
    remote_fixed_idx = block.find(_REMOTE_FIXED_CALL)
    assert validation_idx != -1, "switch's branch-name charset regex is missing"
    assert leading_dash_idx != -1, "switch must still reject a branch name beginning with '-'"
    assert remote_fixed_idx != -1, "switch no longer calls remote_fixed"
    assert validation_idx < remote_fixed_idx and leading_dash_idx < remote_fixed_idx, (
        "branch validation (both checks) must precede the remote_fixed call"
    )

    arg = _remote_fixed_arg(block)
    aside_init_idx = arg.find("$ASIDE_INIT")
    move_aside_idx = arg.find("$move_aside")
    copy_aside_idx = arg.find("$copy_aside")
    switch_idx = arg.find("git switch --guess")
    merge_idx = arg.find("git merge --ff-only")
    assert -1 not in (aside_init_idx, move_aside_idx, copy_aside_idx, switch_idx, merge_idx), (
        "switch's remote_fixed argument must splice in $ASIDE_INIT, $move_aside, $copy_aside, "
        "the switch, and the merge -- deleting any one of these splices must fail this test"
    )
    assert aside_init_idx < move_aside_idx < copy_aside_idx < switch_idx < merge_idx, (
        "switch must: init the aside dir, move untracked additions aside, copy tracked "
        "modifications aside, THEN switch, THEN merge -- in that order, inside the actual "
        "remote command"
    )

    # The range passed to move_aside_cmd (and the target passed to copy_aside_cmd) must compare
    # HEAD directly against origin/<branch>. Before the switch runs, @{u} still names the
    # CURRENT branch's upstream, not the target's -- it cannot stand in for the target here,
    # and HEAD/the switch target may have diverged histories.
    assert 'move_aside_cmd "HEAD origin/' in block, (
        "switch must call move_aside_cmd with a HEAD origin/<branch> range, not @{u}"
    )
    assert 'copy_aside_cmd "origin/' in block, (
        "switch must call copy_aside_cmd with an origin/<branch> target, not @{u}"
    )

    assert "rm " not in arg and "git clean" not in arg and "reset --hard" not in arg
    assert "--discard-changes" not in arg

    # The branch is still validated and still spliced in only via printf %q.
    assert "printf '%q'" in block, "switch must still quote the branch name with printf %q"


def test_switch_and_update_share_helpers_not_a_duplicated_loop() -> None:
    update_block = _case_block("update")
    switch_block = _case_block("switch")
    assert "move_aside_cmd" in update_block
    assert "move_aside_cmd" in switch_block
    assert "copy_aside_cmd" in switch_block
    assert "copy_aside_cmd" not in update_block, (
        "update never had the tracked-modified-file case; it must not call copy_aside_cmd"
    )
    # Neither case arm should re-implement either loop inline -- that logic lives once in each
    # helper, and is only invoked (not duplicated) from the case arms.
    for block, verb in ((update_block, "update"), (switch_block, "switch")):
        assert "git ls-files --error-unmatch" not in block, (
            f"{verb} must call move_aside_cmd rather than duplicate its loop inline"
        )
        assert "git checkout HEAD --" not in block, (
            f"{verb} must call copy_aside_cmd rather than duplicate its loop inline"
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


def test_branch_name_beginning_with_dash_is_refused_even_though_charset_allows_it() -> None:
    block = _case_block("switch")
    validation_match = re.search(r'\[\[ "\$1" =~ (\S+) \]\]', block)
    assert validation_match is not None
    charset = re.compile(validation_match.group(1))
    # "-fd" (an sbatch/scancel-like flag shape) passes the charset regex -- letters and a dash
    # are both permitted -- so a SEPARATE guard must catch a leading '-'.
    assert charset.fullmatch("-fd"), "the charset regex is expected to accept '-fd' on its own"
    assert re.search(r'\[\[ "\$1" != -\* \]\]', block), (
        "switch must separately reject a branch name beginning with '-' (Ruling 21) -- the "
        "charset regex alone would accept it, which could be read as an option by git switch"
    )


# ---------------------------------------------------------------------------------------------
# Stub-ssh subprocess tests: run scripts/alpine/ralpine for real, but with a fake `ssh` on PATH
# that only records its arguments and never contacts any host. Never touches the network.
# ---------------------------------------------------------------------------------------------

_STUB_SSH = """#!/usr/bin/env bash
dir="$(cd "$(dirname "$0")" && pwd)"
printf '%s\\0' "$@" > "$dir/capture.bin"
exit 0
"""


def _stub_ssh_bin(tmp_path: Path) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    ssh_stub = bindir / "ssh"
    ssh_stub.write_text(_STUB_SSH)
    ssh_stub.chmod(0o755)
    return bindir


def _run_ralpine(
    args: list[str], tmp_path: Path, path_prefix: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{path_prefix}:{env.get('PATH', '')}"
    env["ALPINE_HOST"] = "stub.invalid"
    return subprocess.run(
        ["bash", str(RALPINE_PATH), *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=10,
    )


def test_switch_refuses_bad_branch_names_without_ever_calling_ssh(tmp_path: Path) -> None:
    bindir = _stub_ssh_bin(tmp_path)
    capture = bindir / "capture.bin"
    # ralpine's only preliminary ssh call is `status`'s `ssh -O check`; `switch` and `update`
    # never do a ControlMaster check first, so there is nothing to bypass here -- the stub
    # only needs to stand in for remote_fixed's own ssh call, which refusal must never reach.
    for bad in ("main; rm -rf /", "-fd", "has space"):
        result = _run_ralpine(["switch", bad], tmp_path, str(bindir))
        assert result.returncode != 0, f"switch must refuse {bad!r}"
        assert "refusing" in result.stderr, f"switch must explain the refusal of {bad!r}"
        assert not capture.exists(), f"ssh must never be invoked for a refused branch {bad!r}"


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _captured_remote_command(bindir: Path) -> str:
    capture = bindir / "capture.bin"
    assert capture.exists(), "stub ssh was never invoked"
    parts = [p for p in capture.read_bytes().split(b"\0") if p]
    return parts[-1].decode()


def test_switch_captured_remote_command_is_safe_and_ordered(tmp_path: Path) -> None:
    bindir = _stub_ssh_bin(tmp_path)
    result = _run_ralpine(["switch", "task-branch"], tmp_path, str(bindir))
    assert result.returncode == 0, result.stderr
    rendered = _normalize_ws(_captured_remote_command(bindir))

    assert "--no-renames" in rendered
    assert "$(date -u +%Y%m%dT%H%M%SZ)" in rendered, (
        "the aside-dir timestamp must survive UNEXPANDED in the command actually sent -- it is "
        "meant to run on the remote shell, not locally"
    )
    for forbidden in ("rm ", "git clean", "reset --hard", "--discard-changes"):
        assert forbidden not in rendered, (
            f"switch's remote command must never contain {forbidden!r}"
        )

    aside_idx = rendered.find("aside=_moved_aside")
    move_idx = rendered.find("moved aside (untracked")
    copy_idx = rendered.find("copied aside (tracked")
    switch_idx = rendered.find("git switch --guess task-branch")
    merge_idx = rendered.find("git merge --ff-only")
    assert -1 not in (aside_idx, move_idx, copy_idx, switch_idx, merge_idx)
    assert aside_idx < move_idx < copy_idx < switch_idx < merge_idx


def test_update_captured_remote_command_is_safe_and_ordered(tmp_path: Path) -> None:
    bindir = _stub_ssh_bin(tmp_path)
    result = _run_ralpine(["update"], tmp_path, str(bindir))
    assert result.returncode == 0, result.stderr
    rendered = _normalize_ws(_captured_remote_command(bindir))

    assert "--no-renames" in rendered
    assert "$(date -u +%Y%m%dT%H%M%SZ)" in rendered
    for forbidden in ("rm ", "git clean", "reset --hard", "--discard-changes"):
        assert forbidden not in rendered, (
            f"update's remote command must never contain {forbidden!r}"
        )
    # update never had the tracked-modified-file case; its command must not copy anything aside.
    assert "copied aside" not in rendered

    aside_idx = rendered.find("aside=_moved_aside")
    move_idx = rendered.find("moved aside (untracked")
    merge_idx = rendered.find("git merge --ff-only")
    assert -1 not in (aside_idx, move_idx, merge_idx)
    assert aside_idx < move_idx < merge_idx
