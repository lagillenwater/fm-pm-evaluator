# Review Workflow Automation — Design

**Status:** Draft, pending review
**Date:** 2026-08-22

## Motivation

Greene Lab's onboarding process (`greenelab/onboarding`) documents a fully
human code-review workflow: named reviewers, a manual checklist, one
lab-member approval. It has no automated tooling anywhere — no CI-based
linting gate visible to reviewers, no bot review, no coverage tracking.
Meanwhile `fm-pdo-evaluator` already runs CI (ruff, pyright, pytest) but
never surfaces coverage, and is currently a single-author repo with no
second human in the loop by default.

`cytomining/pycytominer` runs CodeRabbit (automated PR review) and
Codecov (coverage tracking + PR diff comments) in production. This
design adapts that pattern for `fm-pdo-evaluator` as a testbed, with
explicit, version-controlled configuration meant to be portable into
`greenelab/onboarding` after a lab discussion — not merged there
directly by this work.

## Goals

- **Coverage visibility** — surface test-coverage trends and per-PR diff
  coverage, which `pytest --cov` already computes but nothing reports.
- **Reference implementation** — a working, documented example the lab
  can evaluate before adopting anything lab-wide.
- **Solo-work safety net** — automated review feedback on PRs opened by
  a single author, without waiting on a second human reviewer.

## Non-goals

- **Porting CU-DBMI's `.agents/skills` files.** Their review-relevant
  skills (`code-review-and-quality`, `ci-cd-and-automation`) are
  prompt-only instructions that duplicate what this session already has
  via `superpowers:requesting-code-review`/`receiving-code-review` and
  Claude Code's built-in `/code-review`. Two ideas are borrowed (the
  five-axis review framing, an agent-facing pre-PR checklist) without
  importing the files or the skill mechanism.
- **Opening a PR against `greenelab/onboarding`.** That happens only
  after a lab discussion; this work produces the proposal content, not
  the PR.
- **Updating `extras/linter_install_tutorial.md`'s stale black/flake8
  guidance** (fm-pdo-evaluator's own pre-commit already uses ruff
  instead). Real gap, but unrelated to CodeRabbit/Codecov/skills —
  flagged in the onboarding proposal doc as a separate finding, not
  acted on here.
- **Language-specific tooling for R or JS.** The onboarding proposal's
  *structure* is written to be language-agnostic, but the only concrete,
  tested implementation is Python (this repo).

## Architecture: three review layers

1. **Agent self-review (pre-push, local).** Claude Code's existing
   `/code-review` skill, invoked before opening a PR. Zero new
   infrastructure — just an instruction, in a new repo-level `CLAUDE.md`,
   telling the agent to run it and what lens to use.
2. **Automated PR review (on PR open, external).** CodeRabbit posts a
   review comment; Codecov posts a diff-coverage comment. Both are
   comment-only — neither blocks merge.
3. **Human approval (existing, unchanged).** Greenelab's existing rule
   (named reviewer, ≥1 lab-member approval) is untouched. Layers 1–2 feed
   the human reviewer better information; they don't replace them.

## Components

### 1. `.coderabbit.yaml` (new, repo root)

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

`request_changes_workflow: false` and `poem: false` are the two defaults
most worth a second look: the first keeps CodeRabbit comment-only (layer
3 stays the actual gate), the second just cuts noise. `path_filters`
duplicates what `.gitignore` already excludes (no large binary/data files
are actually git-tracked — confirmed) as a second layer of defense in
case one is force-added.

### 2. `codecov.yml` (new, repo root)

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

Both project and patch status checks are `informational: true` —
Codecov reports the number, never blocks a merge on it. Matches the
"visibility, not gate" goal.

### 3. `.github/workflows/ci.yml` (modified)

Extend the existing pytest step to also emit XML, and add an upload
step immediately after:

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

`token` is read unconditionally so the same file works whether the
namespace it runs in requires a token or not (see Prerequisites below) —
`codecov-action` no-ops the token field when one isn't needed.

### 4. `CLAUDE.md` (new, repo root)

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

Deliberately minimal — this is not an attempt to write a full
project-context file, only the review-workflow instruction this design
is scoped to.

### 5. `README.md` (modified)

Add two badges under the title:

```markdown
[![CI](https://github.com/lagillenwater/fm-pdo-evaluator/actions/workflows/ci.yml/badge.svg)](https://github.com/lagillenwater/fm-pdo-evaluator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lagillenwater/fm-pdo-evaluator/graph/badge.svg)](https://codecov.io/gh/lagillenwater/fm-pdo-evaluator)
```

## Prerequisites / manual setup

GitHub Apps (both CodeRabbit, and Codecov in its current form) are
installed per GitHub account or organization via a browser-based consent
screen — this cannot be scripted through `gh`/the API; it's a GitHub
security boundary, not a tooling limitation. A fork
(`lagillenwater/fm-pdo-evaluator`) and its upstream
(`greenelab/fm-pm-evaluator`) are different namespaces for this purpose,
so each needs its own install — but it's a one-time per-namespace
decision, not per-repo: installing with "All repositories" (or an
explicit allowlist) covers every current and future repo in that
namespace.

**Phase 1 (now, this repo, no blockers):**
- Install the CodeRabbit GitHub App on the `lagillenwater` personal
  account, granting access to `fm-pdo-evaluator` (or all personal repos).
- Connect `fm-pdo-evaluator` on codecov.io (GitHub OAuth). Public repo,
  so likely tokenless — the CI step reads `${{ secrets.CODECOV_TOKEN }}`
  regardless, so it degrades gracefully if a token turns out to be
  required.

Both are actions only Lucas can take (account-level consent) — not
something this implementation can automate.

**Phase 2 (later, after the lab discussion, needs org-admin action):**
- CodeRabbit and Codecov both need a *separate* install on the
  `greenelab` GitHub org — requires whoever holds app-install rights
  there (per `onboarding.md`, likely Casey Greene or another org admin).
- Codecov's "require token for public repos" setting is an org-level
  toggle that defaults to *required* for established orgs (new orgs
  default to not-required — per Codecov's docs, greenelab almost
  certainly qualifies as established). Budget for a `CODECOV_TOKEN` repo
  secret on the greenelab side even if the fork itself goes tokenless.

This is exactly the kind of decision the lab discussion is meant to
resolve — not something to do unilaterally on shared org infrastructure.

## Validation plan

1. Implement components 1–5 on a feature branch.
2. Complete Phase 1 prerequisites (personal-account installs).
3. Open a real PR from that branch. Confirm:
   - CI still passes with the added `--cov-report=xml` + upload step.
   - Codecov posts a diff-coverage comment.
   - CodeRabbit posts a review that respects `path_instructions` (e.g.
     doesn't flag `scripts/**` for lacking abstraction).
4. Merge, confirm badges render on `README.md`.

## Onboarding-facing deliverable

`docs/onboarding_proposal.md` — written to be discussion- and PR-ready,
mapped onto `greenelab/onboarding`'s actual file structure (confirmed via
research, not assumed):

- New `extras/automated_review_tooling.md`: generalized (language-
  agnostic framing where possible) setup how-to for CodeRabbit + Codecov,
  including the org-install/token findings above as prerequisites.
- Small addition to `extras/code_review_checklist.md`: the five-axis
  framing (correctness/readability/architecture/security/performance) as
  a lens at the top — not a rewrite of the existing ~50-line checklist.
- Short addition to `onboarding.md` § "Source Code, Data, and
  Reproducibility": notes these tools exist and are additive to the
  existing ≥1-lab-member-approval requirement, not a replacement for it.

This document is not submitted anywhere by this work — it is an input to
the lab discussion.

## Open questions to resolve during implementation

- Exact current CodeRabbit YAML schema (field names for path filters/
  instructions, `poem`, `request_changes_workflow`) — pin to CodeRabbit's
  published schema URL (already in the file above) and validate against
  it rather than hand-verify from memory.
- Exact current Codecov config schema (`coverage.status.*.informational`,
  `comment.layout`, `comment.require_changes`) — verify against Codecov's
  current docs at implementation time for the same reason.
- Exact current Codecov tokenless behavior for a *personal* account (the
  org-level default is confirmed; the personal-account default isn't) —
  verify at Phase 1 implementation time.
- Whether `fail_ci_if_error` on the upload step should stay `true`
  (surface upload failures) or move to `false` (never let Codecov's own
  flakiness fail unrelated CI) — starting at `true` to match pycytominer;
  revisit if it causes noise.
