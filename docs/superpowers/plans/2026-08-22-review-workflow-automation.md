# Review Workflow Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a three-layer, version-controlled review workflow (agent self-review, CodeRabbit + Codecov automated PR review, unchanged human approval) in fm-pdo-evaluator, and produce a discussion-ready proposal for extending it to greenelab/onboarding.

**Architecture:** Five new/modified files — a CodeRabbit config, a Codecov config plus CI wiring plus README badges, a `CLAUDE.md` self-review instruction, and an onboarding-facing proposal document — validated end-to-end via one real PR against this repo, once the account-level CodeRabbit/Codecov installs are done.

**Tech Stack:** CodeRabbit (GitHub App + `.coderabbit.yaml`), Codecov (GitHub App + `codecov/codecov-action@v7` + `codecov.yml`), existing `uv`/ruff/pyright/pytest CI, GitHub CLI (`gh`) for PR operations.

**Spec:** `docs/superpowers/specs/2026-08-22-review-workflow-automation-design.md`

## Global Constraints

- Do not run `git commit` or `git push` without first stopping and getting Lucas's explicit go-ahead at that point — he commits in VS Code himself; this holds even if a step below shows the exact command, and even in an automated/subagent execution context.
- Do not open a PR against `greenelab/onboarding`. The onboarding-facing deliverable (Task 4) is a proposal document living in this repo, not a change to that repo.
- Do not port any of CU-DBMI's `.agents/skills` files into this repo.
- Do not modify `extras/linter_install_tutorial.md` in greenelab/onboarding (referenced only as a finding in Task 4's doc, not acted on).
- CodeRabbit config must keep `request_changes_workflow: false` and `poem: false` — comment-only, never blocks merge.
- Codecov config must keep `coverage.status.project.default.informational: true` and the same for `patch` — visibility, never a merge gate.
- The Codecov upload step in CI must read `token: ${{ secrets.CODECOV_TOKEN }}` unconditionally, so the same workflow file works whether or not a token turns out to be required.
- `CLAUDE.md` stays scoped to review workflow only — not a general project-context file.
- Do not merge the Task 5 validation PR without an explicit go-ahead from Lucas.

---

## Task 1: CodeRabbit configuration

**Files:**
- Create: `.coderabbit.yaml`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `.coderabbit.yaml` at repo root — read automatically by the CodeRabbit GitHub App once it has access to this repo (Task 5 depends on this file existing on the PR branch).

- [ ] **Step 1: Create `.coderabbit.yaml`**

Create the file with exactly this content:

```yaml
# yaml-language-server: $schema=https://coderabbit.ai/integrations/schema.v2.json
language: en-US
early_access: false

reviews:
  profile: chill
  request_changes_workflow: false
  high_level_summary: true
  poem: false
  review_status: true
  auto_review:
    enabled: true
    base_branches:
      - main
  path_filters:
    - "!archive/**"
    - "!docs/figures/**"
    - "!**/*.h5ad"
    - "!**/*.csv"
    - "!**/*.zip"
    - "!**/*.pptx"
  path_instructions:
    - path: "src/fmharness/**"
      instructions: >
        This is the core evaluation-harness library: data loaders, model
        adapters, splitters, evaluation metrics. Apply strict scrutiny to
        correctness — train/test leakage across splits, off-by-one
        indexing into biological identifiers (drug/sample IDs), silent
        NaN propagation, and statistical methodology (permutation nulls,
        cross-validation folds).
    - path: "scripts/**"
      instructions: >
        One-off analysis/pipeline scripts, not library code. Favor
        correctness and reproducibility (fixed random seeds, documented
        inputs/outputs). Do not flag lack of abstraction or code reuse —
        that is expected here.
    - path: "tests/**"
      instructions: >
        Check that tests assert on actual values or behavior, not just
        "runs without error." Flag missing edge cases for data loaders
        (empty input, mismatched IDs, all-NaN columns).

chat:
  auto_reply: true
```

- [ ] **Step 2: Validate YAML syntax**

Run: `uv run pre-commit run check-yaml --files .coderabbit.yaml`
Expected: `Passed` (this reuses the `check-yaml` hook already configured in `.pre-commit-config.yaml` — no new dependency needed).

- [ ] **Step 3: Stage the change and check in before committing**

```bash
git add .coderabbit.yaml
```

Stop here and confirm with Lucas before running `git commit` — do not commit autonomously (see Global Constraints).

---

## Task 2: Codecov integration (config, CI wiring, badges)

**Files:**
- Create: `codecov.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `codecov.yml`, the updated `ci.yml` coverage-upload step, and the two README badges — read/exercised by Task 5's validation PR.

- [ ] **Step 1: Create `codecov.yml`**

Create the file with exactly this content:

```yaml
codecov:
  require_ci_to_pass: true

coverage:
  status:
    project:
      default:
        informational: true
    patch:
      default:
        informational: true

comment:
  layout: "diff, flags, files"
  behavior: default
  require_changes: false
```

- [ ] **Step 2: Modify `.github/workflows/ci.yml` to emit and upload coverage**

The current last two lines of the file are:

```yaml
      - name: Pytest
        run: uv run pytest --cov=src/fmharness --cov-report=term-missing
```

Replace them with:

```yaml
      - name: Pytest
        run: uv run pytest --cov=src/fmharness --cov-report=term-missing --cov-report=xml
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v7
        with:
          files: ./coverage.xml
          fail_ci_if_error: true
          token: ${{ secrets.CODECOV_TOKEN }}
```

- [ ] **Step 3: Add badges to `README.md`**

The current start of the file is:

```markdown
# fm-pdo-evaluator

Foundation-model evaluation harness for patient-derived tumor organoid (PDTO) drug-response prediction. Realizing the benefits of foundation models requires careful evaluations that map the boundaries of generalization — and that test a model in the mode it was actually designed for.
```

Replace it with:

```markdown
# fm-pdo-evaluator

[![CI](https://github.com/lagillenwater/fm-pdo-evaluator/actions/workflows/ci.yml/badge.svg)](https://github.com/lagillenwater/fm-pdo-evaluator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lagillenwater/fm-pdo-evaluator/graph/badge.svg)](https://codecov.io/gh/lagillenwater/fm-pdo-evaluator)

Foundation-model evaluation harness for patient-derived tumor organoid (PDTO) drug-response prediction. Realizing the benefits of foundation models requires careful evaluations that map the boundaries of generalization — and that test a model in the mode it was actually designed for.
```

- [ ] **Step 4: Validate YAML syntax**

Run: `uv run pre-commit run check-yaml --files codecov.yml .github/workflows/ci.yml`
Expected: `Passed` for both files.

- [ ] **Step 5: Confirm the local suite still passes with the new coverage flag**

Run: `uv run ruff check .`
Expected: no errors.

Run: `uv run ruff format --check .`
Expected: no errors.

Run: `uv run pyright`
Expected: 0 errors.

Run: `uv run pytest --cov=src/fmharness --cov-report=term-missing --cov-report=xml`
Expected: existing test suite passes, and a `coverage.xml` file is created in the repo root.

Run: `test -f coverage.xml && echo "coverage.xml created"`
Expected: `coverage.xml created` (note: this file is already covered by `.gitignore` — do not add it to git).

- [ ] **Step 6: Stage the changes and check in before committing**

```bash
git add codecov.yml .github/workflows/ci.yml README.md
```

Stop here and confirm with Lucas before running `git commit` — do not commit autonomously (see Global Constraints).

---

## Task 3: Agent self-review instructions

**Files:**
- Create: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `CLAUDE.md` at repo root — read automatically by Claude Code in future sessions in this repo.

- [ ] **Step 1: Create `CLAUDE.md`**

Create the file with exactly this content:

```markdown
# fm-pdo-evaluator — agent instructions

## Before opening a PR

Run `/code-review` on your diff before pushing. Use the five-axis lens —
correctness, readability, architecture, security, performance — and fix
anything flagged with high confidence. This is the same bar CodeRabbit
and a human reviewer will apply; catching it here saves a review round
trip.

This file intentionally stays scoped to review workflow. Broader project
context lives in `docs/` (see `docs/fm-pdo-evaluator-plan.md` and
`docs/models.md`).
```

- [ ] **Step 2: Verify the file content**

Run: `grep -q "code-review" CLAUDE.md && echo "found"`
Expected: `found`

- [ ] **Step 3: Stage the change and check in before committing**

```bash
git add CLAUDE.md
```

Stop here and confirm with Lucas before running `git commit` — do not commit autonomously (see Global Constraints).

---

## Task 4: Onboarding-facing proposal document

**Files:**
- Create: `docs/onboarding_proposal.md`

**Interfaces:**
- Consumes: the concrete config choices from Tasks 1–2 (referenced by value, not by import — this is a prose document).
- Produces: `docs/onboarding_proposal.md` — the artifact Lucas brings to the lab discussion. No later task consumes it programmatically.

- [ ] **Step 1: Create `docs/onboarding_proposal.md`**

Create the file with exactly this content:

```markdown
# Proposal: Automated Review Tooling for greenelab/onboarding

Status: draft, for lab discussion. Not yet proposed as a PR against
greenelab/onboarding — see the parent spec,
`docs/superpowers/specs/2026-08-22-review-workflow-automation-design.md`,
for why.

## Why

greenelab/onboarding documents a fully human code-review workflow: named
reviewers, a manual checklist, one lab-member approval. It has no
automated tooling anywhere — no CI-based linting gate visible to
reviewers, no bot review, no coverage tracking. This was tested first in
fm-pdo-evaluator (a single-author repo, so a good low-risk testbed)
before being written up here for the wider lab.

## What changes, and where

Three files in greenelab/onboarding would change:

### 1. New: `extras/automated_review_tooling.md`

```
# Automated Review Tooling (Optional)

This is an optional layer on top of the review process described in the
main onboarding document — it does not change the requirement that every
PR gets at least one lab-member approval before merging.

## What it adds

- **An automated PR review bot** (e.g. CodeRabbit) comments on pull
  requests automatically when they're opened, flagging likely bugs,
  security issues, and (depending on configuration) style concerns.
  Comment-only by default — it does not block merges.
- **Coverage reporting** (e.g. Codecov) tracks what fraction of a
  repository's code is exercised by its test suite, and comments on each
  PR with how much of the *changed* code is covered. Informational by
  default — a low number is a prompt to add tests, not a blocker.
- **A pre-PR self-review step**, for repos where contributors use an AI
  coding agent: an instruction (in `CLAUDE.md`, `AGENTS.md`, or
  equivalent) to run the agent's own review capability against the diff
  before opening the PR.

These three are complementary, not redundant: the self-review step
happens before a PR exists and leaves no external trace; the bot review
happens automatically on every PR regardless of who or what wrote it,
and leaves a permanent, visible comment; coverage reporting does
something neither of the others can — track a number over time across
commits.

## Setting this up for a new repository

1. **Automated PR review.** Install the CodeRabbit GitHub App (or an
   equivalent) on your personal account (for a fork/testbed) or request
   it be installed on the `greenelab` GitHub organization (for a shared
   repository — this requires an org admin, since GitHub App
   installation is a browser-based, per-namespace action that can't be
   scripted). Commit a `.coderabbit.yaml` to the repo root rather than
   relying on dashboard-only settings, so the configuration is
   version-controlled and reviewable like any other code. Recommended
   starting defaults: `profile: chill` (flags bugs/security/logic
   issues, skips style nitpicks your linter already catches),
   `request_changes_workflow: false` (comments only, never blocks
   merge — the human-approval rule stays the actual gate).

2. **Coverage reporting.** Add a coverage-report step to your existing
   CI (e.g. `pytest --cov=<package> --cov-report=xml` for Python), then
   upload it with `codecov/codecov-action`. Commit a `codecov.yml` with
   `coverage.status.project.default.informational: true` and the same
   for `patch`, so coverage is visible without being a merge gate.
   Connect the repository at codecov.io — for a personal fork this is
   self-service; for a `greenelab`-owned repository, an org admin needs
   to install Codecov's GitHub App there too, and greenelab's org-level
   "require token for public repos" setting will likely need a
   `CODECOV_TOKEN` secret added to that specific repository (existing
   organizations default to requiring one).

3. **Pre-PR self-review.** Add a short section to the repository's
   `CLAUDE.md` (or equivalent) telling the agent to run its review
   capability before opening a PR, and what to check for (e.g. the
   five-axis lens below).

## What this costs

Both CodeRabbit and Codecov are free for public repositories. Neither
requires ongoing maintenance beyond the initial setup — configuration is
version-controlled and changes rarely. The one recurring cost is
judgment: each repo's `.coderabbit.yaml` `path_instructions` should
reflect what's actually worth flagging in that codebase (e.g., don't
apply library-code scrutiny to one-off analysis scripts).
```

### 2. Addition to `extras/code_review_checklist.md`

Insert before the existing "Pride" bullet, as a short framing note:

```
When reviewing a PR, it helps to check it against five axes:
**correctness** (does it do what it claims, including edge cases),
**readability** (would a lab member unfamiliar with this code
understand it), **architecture** (does it fit the rest of the
codebase, or bolt on awkwardly), **security** (credentials, injection,
unsafe deserialization, unvalidated external input), and
**performance** (obviously wasteful loops or memory use, for code that
runs at any scale). The checklist below expands on specific items
within these axes.
```

### 3. Addition to `onboarding.md`, § "Source Code, Data, and Reproducibility"

Insert after the existing "Reviewing Pull Requests" bullet:

```
**Automated tooling (optional).** Repositories may additionally run
automated PR review (e.g. CodeRabbit) and coverage reporting (e.g.
Codecov) — see `extras/automated_review_tooling.md` for what these add
and how to set them up. These are additive to, not a replacement for,
the ≥1-lab-member-approval requirement above.
```

## What this does NOT change

The existing human-approval requirement (named reviewer, ≥1 lab-member
approval before merge) is untouched. Nothing here is mandatory — it's
opt-in tooling a repo maintainer can adopt, not a new onboarding rule.

## Prerequisites for org-wide adoption

GitHub Apps (both CodeRabbit and Codecov) are installed per GitHub
account or organization via a browser-based consent screen — this can't
be scripted, and a personal fork's install doesn't cover the `greenelab`
org. Rolling this out org-wide needs someone with app-install rights on
`greenelab` (per onboarding.md, likely Casey Greene or another org
admin) to install both apps once, ideally scoped to "all repositories"
so it covers future repos automatically rather than needing to be
repeated per repo. Budget for a `CODECOV_TOKEN` secret on any
`greenelab`-owned repository that adopts this, since Codecov's
"require token for public repos" org setting defaults to required for
established organizations.

## Known adjacent gap (not addressed by this proposal)

`extras/linter_install_tutorial.md` still documents `black` + `flake8`
for local linting. fm-pdo-evaluator (and other recent lab repos) have
moved to `ruff`, which replaces both. Worth a separate, focused update —
flagged here so it isn't lost, not bundled into this proposal since it's
an unrelated tooling change.

## Reference implementation

This proposal was validated end-to-end in fm-pdo-evaluator before being
written up here — see that repo's `.coderabbit.yaml`, `codecov.yml`, and
`.github/workflows/ci.yml` for the working configuration, and `CLAUDE.md`
for the self-review instruction.
```

- [ ] **Step 2: Verify the file content**

Run: `grep -c "automated_review_tooling.md\|code_review_checklist.md\|onboarding.md" docs/onboarding_proposal.md`
Expected: a count of at least 3 (confirms all three target files are referenced).

- [ ] **Step 3: Stage the change and check in before committing**

```bash
git add docs/onboarding_proposal.md
```

Stop here and confirm with Lucas before running `git commit` — do not commit autonomously (see Global Constraints).

---

## Task 5: Manual prerequisites and end-to-end validation

**Files:** none created or modified — this task exercises Tasks 1–2's output against the real CodeRabbit and Codecov services.

**Interfaces:**
- Consumes: `.coderabbit.yaml` (Task 1), `codecov.yml` + `ci.yml` + `README.md` (Task 2) — all must be committed to the branch before Step 2 below.
- Produces: a validated, merged PR demonstrating the whole workflow works.

This task cannot be fully automated. Steps 1 and 6 require Lucas personally — a subagent executing this plan must stop and wait at those points rather than attempting to proceed.

- [ ] **Step 1: STOP — confirm Phase 1 prerequisites are complete**

Ask Lucas to confirm both of the following are done (both are browser-based, account-level actions with no CLI equivalent — see the spec's "Prerequisites / manual setup" section for why):

1. The CodeRabbit GitHub App is installed on the `lagillenwater` GitHub account with access to `fm-pdo-evaluator`.
2. `fm-pdo-evaluator` is connected on codecov.io.

Do not proceed to Step 2 until Lucas confirms both are done.

- [ ] **Step 2: Push the branch and open the validation PR**

Confirm with Lucas before pushing (this is the point where the work becomes visible outside the local machine). Once confirmed:

```bash
git push -u origin review-workflow-automation
gh pr create --title "Add review workflow automation: CodeRabbit, Codecov, agent self-review" --body "$(cat <<'EOF'
Implements the review-workflow-automation design in docs/superpowers/specs/2026-08-22-review-workflow-automation-design.md.

Adds:
- .coderabbit.yaml — automated PR review (comment-only, non-blocking)
- codecov.yml + CI coverage upload + README badges — coverage visibility (informational, non-blocking)
- CLAUDE.md — pre-PR agent self-review instruction

This PR is also the Phase-1 validation step from the spec: confirms CI still passes, Codecov posts a diff-coverage comment, and CodeRabbit posts a review respecting path_instructions.
EOF
)"
```

- [ ] **Step 3: Confirm CI passes**

Run: `gh pr checks --watch`
Expected: all checks pass, including the new coverage-upload step.

- [ ] **Step 4: Confirm CodeRabbit reviewed the PR**

Run: `gh pr view --json comments --jq '.comments[] | select(.author.login | test("coderabbit"; "i")) | .body[0:200]'`
Expected: at least one comment from a CodeRabbit bot account. This can take a few minutes to appear — if empty on the first check, wait and re-run rather than treating it as a failure immediately. If still empty after several minutes, re-check Step 1's prerequisite (the App may not actually have access to this repo).

- [ ] **Step 5: Confirm Codecov posted a diff-coverage comment**

Run: `gh pr view --json comments --jq '.comments[] | select(.author.login | test("codecov"; "i")) | .body[0:200]'`
Expected: at least one comment from a Codecov bot account, showing diff/patch coverage. Same note as Step 4 on timing.

- [ ] **Step 6: STOP — confirm with Lucas before merging**

Report the results of Steps 3–5 to Lucas and wait for his explicit go-ahead. Do not merge autonomously (see Global Constraints).

- [ ] **Step 7: Merge**

Only after Step 6's go-ahead:

```bash
gh pr merge --squash
```

- [ ] **Step 8: Confirm badges render**

Run: `curl -s -o /dev/null -w "%{http_code}" https://github.com/lagillenwater/fm-pdo-evaluator/actions/workflows/ci.yml/badge.svg`
Expected: `200`

Run: `curl -s -o /dev/null -w "%{http_code}" https://codecov.io/gh/lagillenwater/fm-pdo-evaluator/graph/badge.svg`
Expected: `200`

If either returns a non-200 status, wait a minute (badge services can lag the first upload) and retry before treating it as a real failure.
