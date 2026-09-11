# Rung 1 — decision lineage

**As of** 2026-09-11. Dated entries amending this task's documents, oldest first, labeled by the
document they amend. `design.md` carries the current position.

## design.md

- **2026-09-11** — **Task opened on a branch cut from `rung0-assay-reliability` before rung 0
  merges**, at Lucas's direction ("start building rung 1 off of rung 0's code"). PROCESS §4 says to
  branch after the previous task merges; that is knowingly set aside so rung 1 reuses rung 0's
  measurement code as it stands.
- **2026-09-11** — **Rung 0 is not revisited.** Its promoted 5 uM ceilings are the ceiling, and the
  restriction to rung 1's grid is a filter of its promoted per-triple table, not a recomputation. At
  Lucas's direction: "Report what we have as a ceiling… After we complete every run we can return
  and make edits." Two rung 0 issues noticed while designing are recorded, not acted on:
  - Rung 0's responder noise decomposition selects responders on one of the two plates it compares,
    which likely inflates its between-plate share (0.74).
  - Its responder ceiling is selected from one fixed plate group.
- **2026-09-11** — **One dose, 5 uM**, the dose with the most replicated pairs, at Lucas's direction
  ("Rung 1 must control for dose. To streamline, focus on the dose with the highest N").
- **2026-09-11** — **Results are reported; nothing passes or fails**, at Lucas's direction ("it
  doesn't have to 'pass', we are reporting results"). The rung's hypotheses, H1 (Stack embeddings
  against the drug mean and expression baselines) and H2 (drug fine-tune against cytokine
  post-training), are Lucas's wording made into declared paired contrasts. The proposed `SPEC.md`
  text replaces rung 1's "Passing means" with "Reports".
- **2026-09-11** — **Stack is in this task.** This supersedes the choice made in conversation on
  2026-09-10 to build the harness with reference models first and bring Stack in as a later task:
  both hypotheses are about Stack, so the rung cannot report on them without it. Kernel ridge on a
  fixed representation keeps Stack's addition to a representation build plus one shared fit.
- **2026-09-11** — **Leave one line out and leave one drug out, never together**, at Lucas's
  direction ("within the same distribution, using Leave one line out or leave one drug out"),
  following the archive plan (`docs/tasks/arm2-harness-validation/design.md` on branch
  `docs-project-spec-tree`, archive): leave-one-line-out primary, leave-one-drug-out the stress test.
  Ruled out in conversation on 2026-09-10:
  - Scoring each prediction against a fixed half of rung 0's plate split. Rejected by Lucas for the
    bias in which half is scored.
  - A two-fold swap of the halves. Superseded by the full-data target, since leave-one-line-out
    already keeps the held-out line out of fitting.
- **2026-09-11** — **The ceiling is √SB, not SB.** Lucas asked why the square root. A prediction is
  scored against one noisy measurement, so a perfect prediction reaches √R, while rung 0's R is the
  agreement of two noisy measurements. The derivation, the archive lineage's unresolved "baselines
  exceed the ceiling" case it explains, and the known-answer test are in `design.md` §6 and §8.
- **2026-09-11** — **A complete grid, 107 drugs × 50 lines.** One drug replicated in only 46 lines
  is dropped.
  - **Why:** the archive lineage's line-blind predictor reached the base embedding's headline
    interaction (0.118 against 0.119), measured in-sample, attributed to 17 missing pairs that
    leaked each line's drug composition. A complete grid removes that route and makes the
    leave-one-drug-out kernel a Kronecker product with an exact solution.
- **2026-09-11** — **One estimator for every representation**: kernel ridge with a penalty tuned
  inside the fold. Also a kNN baseline under leave-one-line-out, and a random twin per
  representation.
  - **Why:** representations then compete at identical flexibility, and no embedding is truncated
    to a principal-component count (archive `docs/tasks/cross-check-fairness-and-capacity/design.md`).
  - **Dropped:** PCA and NMF variants of expression; kernel ridge on the full profile replaces them.
- **2026-09-11** — **Stack generation mode is out of scope.**
  - **Why:** a leakage-free leave-one-line-out context needs 5,350 generation runs per checkpoint,
    and the archive generation result was null (0.012 and 0.021) even with the held-out line in
    context.
- **2026-09-11** — **Run outputs stay out of git until the separate change relocating evidence
  lands.** Lucas is handling that change. Rung 0's evidence commit added about 205,000 lines of
  generated tables, which is the problem it addresses.

### Review round 1 (2026-09-11), at Lucas's direction

- **2026-09-11** — **REVERSAL: PCA and NMF baselines restored**, reversing the entry above that
  dropped them ("Add the PCA and NMF baselines").
  - **How they work:** each reduces expression to 2–20 components, with the count chosen inside the
    round, and goes through the same ridge regression as every other description.
  - **Where they appear:** in H1(b) under both tests. The main test now has 6 comparisons and the
    second test 5.
- **2026-09-11** — **Alpine instructions added** (design §9): the job-array stage plan, and which
  nodes were free during rung 0.
  - **acpu:** 1,353 jobs pending, 914 ahead of this account.
  - **amem:** eight nodes idle; rung 0's slices packed onto three of them, finishing in 2.5–2.8 h.
  - **GPU:** jobs moved from `aa100` to `ah200`.
  - **Disk:** scratch read slowly on 2026-09-10.
  - **Sources:** `docs/tasks/rung0-assay-reliability/decisions.md` (2026-09-10, packed slices),
    `verification.md`'s job table, rung 0's job logs for the node names, and the archive
    `scripts/alpine/06_stack_embed.sbatch` for the GPU partition.
- **2026-09-11** — **Rewritten in plain language:** at most 250 lines, at a 10th-grade reading
  level, with no jargon.
  - **Checked by script, not by eye:** 249 lines; Flesch-Kincaid grade about 7.9 over the prose.
  - **What changed:** the estimator is described as ridge regression on each line description. The
    solution method (a kernel form, exact under a complete grid) is implementation detail for
    `plan.md`. "Random twin" became "random stand-in".
  - **Moved here:** the proposed `docs/SPEC.md` text, below, to meet the line limit.
- **2026-09-11** — **Commits: work in a worktree, rebuild into topic commits for review**, at Lucas's
  direction ("consolidate commits just a few based on specific topics, as with the updates to
  rung 0 last night… work on a worktree").
  - **Where the work is:** the worktree `.claude/worktrees/rung1-held-out-prediction` on
    `rung1-held-out-prediction-work`, cut from `dff524f`. The design files moved there; the main
    checkout is back on `rung0-assay-reliability`.
  - **Before the pull request:** the final files are recommitted as five topic commits on
    `rung1-held-out-prediction`, the task branch, from the same base. It is checked to have no
    difference from the working branch, then pushed without force.
  - **Why not rewrite the working branch in place:** Alpine's checkout follows the working branch,
    and `ralpine update` refuses a rewritten history. Rung 0's rebuild needed a force-push and tags;
    this route needs neither.

**Proposed `docs/SPEC.md` text for rung 1** (applied on approval):

> ### Rung 1 — can a model predict a cell line or a drug it has not seen?
>
> **Question** Given a line or a drug left out of training, can a model predict the expression change?
> **Adds** One hidden thing at a time, a line or a drug. Same lab method, cells and dose as rung 0. What
> counts as unseen is set by each model's leakage record.
> **Measure** Rung 0's score (per-pair correlation across genes, averaged over pairs), at one dose, on a
> grid where every drug was measured in every line, for responding and all genes; also as a fraction of
> √SB, since a prediction meets measurement noise once, not twice.
> **Reports** Head-to-head comparisons with confidence intervals, p-values and MDEs, including each line
> description against the drug average and a random stand-in. No pass mark.
> **Why it matters** A model that cannot beat the drug average when only the line or drug is new will not
> do better when more changes.

- **2026-09-11** — **Design approved (gate 1)** by Lucas: "design looks good. Execute autonomously until
  you reach the summary." Applied in the same change: the proposed rung 1 text in `docs/SPEC.md`, its
  row and status in `README.md`, and its row in `docs/STATE.md`. Lucas's own edit to §1 (dropping the
  sentence "The lab method, cells and dose stay as in rung 0" and "fixed before any run") stands.

## plan.md

- **2026-09-11** — **Pre-flight scan of the plan, four rulings applied before Task 1.**
  - **Fit controls moved to Task 9.** Tasks 7 and 8's known-answer fit controls place their plant relative to
    an MDE, and the MDE machinery arrives in Task 9. Tasks 7–8 keep their exactness tests.
  - **Line crosswalk added to Task 3.** Task 4 needed each line's Cellosaurus id and no task produced it. The
    scan writes `rung1_line_crosswalk.csv` from the screen's own key columns, which also covers the line
    whose DepMap key is the literal `NA`.
  - **Drug-table fixture added to Task 2** (`tests/fixtures/tahoe_drug_metadata.csv`), so the grid's name
    crosswalk and exclusion tests run offline.
  - **Data jobs submitted after Task 6.** Task 11's data stages depend only on Tasks 1–6, so they are submitted
    as soon as those land, while the models are built.
- **2026-09-11** — **The answers are scanned, not read from rung 0's cache.** Rung 0's cache keeps only genes
  with a fold change in both plate halves (`frame_from_slice`). The design defines the answer over every gene
  measured on at least one plate. Reading the cache would have narrowed an approved definition; one filtered
  scan keeps it, at the cost of a few hours of cluster time.
