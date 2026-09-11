# Rung 0 — review findings and their dispositions

**As of** 2026-09-09. Per `docs/PROCESS.md` §1 (Review): each finding, what was checked, what was
done, and what remains.

## The review

One review so far, on pull request #9 (greenelab/fm-pm-evaluator), posted 2026-09-02 by the
lab's principal investigator as an AI-assisted review (Codex 5.6-Sol, findings read by the
account holder before posting). Eight findings, five at P1. Its recommendation: keep the pull
request in draft; fix the estimand and the specification first, then the two scientific controls
and the dose-key propagation; run the complete dose-fixed pipeline and the permutation stage;
regenerate run-time provenance; audit afresh before promotion.

Every finding was checked against the tree before anything was changed. Seven of eight are
confirmed; one could not be reproduced. The review also missed one thing, recorded at the end.

## Findings

### P1 — the promoted result records the wrong producing commit

**Confirmed.** The run's parameter sidecar records commit `5192606` (dose-pooled code); the
provenance record beside the promoted copy records `92407c1`, the commit promotion happened at,
whose code held dose fixed. The promotion script read `HEAD` at promotion time.

**Done (`ed7b800`).** `scripts/promote_result.py` records the commit the run was made at, read
from the `<result>.params.json` sidecar every run writes, and the promotion commit separately as
`promotion_commit`; it refuses when neither the sidecar nor `--code-commit` names one. The
verification battery checks that a promoted record's `code_commit` is the sidecar's `git_sha`,
which the existing record fails.

**Done (2026-09-11).** The pooled promotion is withdrawn, and this run's four tables are promoted with records naming the commit the run was made at (`8392557`, combine job 32379206).

### P1 — the noise decomposition is biased in the two-plate regime

**Confirmed, analytically.** At two plates the variance across plates has one degree of freedom.
Flooring each gene's `var - se^2` at zero before averaging has expectation
`0.25 * E[max(chi2_1 - 1, 0)] = 0.121` for a within-plate variance of 0.25 and no plate effect —
a share of 0.15, with 31.7% of genes positive — which is the review's simulation to three
decimals. The negative control at 24 plates could not see it. The dose-fixed noise run that
completed on 2026-09-02 (job 31998062) reported a share of 0.192, most of which is that bias.

**Done (`a1dcff5`).** The estimator pools first and floors once:
`max(mean(var) - mean(se^2), 0)` over gene-conditions, for all genes, for each condition's
responders and for its non-responders, and as the mean over conditions of each condition's own
pooled share. The per-gene column is kept signed for the figures. The slices' partial sums now
carry the responders' squared standard errors, which the earlier partials did not, so the
responder share pools too. The negative control is at two plates and shows what the per-gene
floor would have reported on the same pool; a positive control at two plates recovers a planted
component. The battery recomputes every reported share from the per-condition sums.

### P1 — the effect-size control is circular

**Confirmed.** The control ranked conditions by the mean of `|a + b| / 2`; a large sum is reached
most easily when the halves agree, so pure noise rose through the terciles and the verifier's
monotone criterion passed on nothing. This task's own 2026-09-01 leakage-control entry describes
the same mechanism for responder selection.

**Done (`a1dcff5`).** The per-condition table carries each half's mean absolute delta separately;
the control ranks by one half, scores the correlation of the pair, then swaps, and both rankings
must rise. A signal-free pool is pinned flat under the shipped ranking and shown to rise under the
old one. The battery recomputes both rankings from the committed table.

### P1 — the dose-fixed migration is incomplete

**Confirmed, all four parts.** The pool-arithmetic check merged on (line, drug); the summary
notebook's noise merge did the same; the parameter sidecar still wrote "pooled for the
reliabilities"; and the verifier demanded the full population's row count and mean from a
two-million-row sample, which would have failed the moment the noise stage completed.

**Done (`a1dcff5`, `ed7b800`).** Every merge in the battery is on the full (line, drug, dose)
key; the sidecar records the dose handling, the split rule and the weighting; the noise checks
read the per-condition sums the run commits for every condition and use the sample only for the
row-wise identity, the strata and the control.

**Done.** The summary notebook merges on the full key and reads the dose-fixed run's tables.

### P1 — the audit gate has not passed

**Confirmed.** The audit covers `c3e55b2`; commits after it addressed seventeen of its twenty-one
drift items without writing the dispositions back; four were never addressed; no `review.md` or
`verification.md` existed; the summary said the pooled design "passed a full drift audit", which
it had not.

**Done.** This document. The audit's fix-wave section now records every item's disposition
against the current tree. The summary's false sentence goes with the notebook rewrite.

**Done.** `verification.md` records the run; the re-audit passed on its third pass (`audit.md`, "Re-audit").

### P2 — the specification does not define the quantity later rungs divide by

**Confirmed.** `docs/SPEC.md` still scored a (line, drug) condition and its frame carried no dose
handling or weighting; the task design still said "doses pooled" in its promotion paragraph and
called a dose-resolved reliability out of scope, both stale after the 2026-09-01 reversal.

**Done.** The spec's scoring unit is the (line, drug, dose) triple; the frame includes the unit,
the dose handling and the weighting, and a change to any of them is a new frame; the rung reports
the equal-weight mean over triples beside each dose level and the per-pair weighting, promotes
the per-triple table those are aggregates of, and declares the ceiling in the task's decision
lineage. The design says the same, and its stale sentences are gone.

**Done (2026-09-11).** The ceilings are declared per dose, each against floors drawn at its own dose.

### P2 — definitions that do not match their names

**Confirmed, all three.** `n_plates_even` was the parity of the total plate count; the
reliabilities required a plate in each hash half while the decomposition required any two plates;
the same-drug null ignored dose.

**Done (`a1dcff5`).** The split alternates over the sorted plate ids within each triple, as a
committed table, so every replicated triple has a plate on each side, the equal-halves flag is
exactly an even plate count, and the two inclusion rules are one rule. The same-drug null holds
dose fixed. The battery checks the split against its rule and the flag against both definitions.

### P2/P3 — the advertised clean checks

**Partly confirmed.** Four `.DS_Store` files, trailing whitespace, and four archived-lineage files
under `results/` swept in by `c155f85` were real and are removed (`ed7b800`).

**Not reproduced.** With the locked pyright 1.1.409 in the project's strict configuration, both
the whole tree and `src/fmharness/statistics.py` alone report zero errors, before and after these
changes. The seven strict-typing errors the review reports could not be reproduced here; the
reviewer is asked for the command and environment that produced them.

## What the review did not raise

Under `hash(plate) % 2` a two-plate triple splits only when its two ids hash to different
parities, and 7,441 of the 7,641 replicated triples have exactly two plates. How many split was
never counted; the design's "one plate against one" was assumed. The alternating split removes
the question rather than answering it: every replicated triple splits.

## The run, and what follows it

The dose-fixed run completed on 2026-09-10 and 2026-09-11 (assign 32366450, the packed slices
32369084, the permutation check 32378577, the final combine 32379206), on code carrying every fix
above plus the per-dose floors. It is verified (`verification.md`: the battery 79 of 79), audited
(`audit.md`, re-audit passed), and promoted. The branch history was rebuilt into five commits by
area, with every earlier commit reachable under the tag `rung0-pre-rewrite-2026-09-11`.
