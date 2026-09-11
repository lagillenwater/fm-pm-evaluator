# Rung 1 — predicting a cell line or a drug the model has not seen

**Task** `rung1-held-out-prediction` · **Status** OPEN, design approved 2026-09-11 · **Branches** work on `rung1-held-out-prediction-work`, pull request from `rung1-held-out-prediction` (§10)
**Steps** build, split, fit, score, null, document, promote.
**Spec** [docs/SPEC.md](../../SPEC.md), rung 1 · **State** [docs/STATE.md](../../STATE.md) · **Ceiling from** [rung 0](../rung0-assay-reliability/design.md)
**As of** 2026-09-11. Why each choice was made: [`decisions.md`](decisions.md).

## 1. Question and hypotheses

Tahoe-100M measured how 379 drugs change gene expression in 50 cancer cell lines. Rung 1 asks whether a
model can predict that change for a line, or a drug, it has never seen. Two hypotheses:

- **H1.** Stack's summary of a cell line (its *embedding*) predicts how that line, in particular, responds to
  a drug. It beats (a) the drug's average effect in other lines, and (b) the same model given the line's
  gene expression instead.
- **H2.** Stack fine-tuned on drug experiments (sci-Plex) predicts better than the released Stack, which was
  tuned on immune-signal experiments (cytokines).

We report estimates, 95% confidence intervals, p-values, and the smallest effect each comparison could
reliably detect (the *minimum detectable effect*, MDE). Nothing passes or fails.

## 2. What stays fixed

| Item | Choice | Why |
|---|---|---|
| Data | Tahoe-100M's table of expression changes (tranche `tahoe100m-pseudobulk-de.v1`), as in rung 0 | same measurement |
| Dose | **5 micromolar (uM) only** | most repeated measurements: 5,396 of rung 0's 7,641 |
| Pairs | **107 drugs × 50 lines = 5,350 (line, drug) pairs**, each on at least two plates | every drug in every line, so no gaps skew averages; one drug measured in only 46 lines is dropped |
| Answer | each gene's change (`log2FoldChange`), averaged over the pair's plates | all of the pair's data |
| Score | correlation across genes between predicted and measured change (Pearson r), per pair, averaged over pairs | rung 0's score |
| Genes | **Responding (main):** called changed (adjusted p-value `padj` < 0.05) on at least one of the pair's plates. **All (second):** every measured gene. A pair needs 50 genes; all models use the same pairs | responding genes carry the repeatable signal (rung 0: 0.58 against 0.08) |
| Ceiling | rung 0's promoted 5 uM results, limited to these pairs (§6) | rung 0 is not re-run |

## 3. How a line or a drug is hidden

- **Leave one line out (main test), 50 rounds.** Each round hides one line. Models learn from the other 49
  lines and predict every drug for the hidden line, seeing only how it looks with no drug (§4).
- **Leave one drug out (second test), 107 rounds.** Each round hides one drug. Models learn from the other
  106 drugs and predict it in every line, seeing only its chemical structure.
- **Rules.** Settings are chosen inside each round from training data only. A line and a drug are never hidden together.

## 4. What models are told about lines and drugs

Each line is described once, from untreated cells given only the solvent dimethyl sulfoxide (DMSO).

| Line description | What it is |
|---|---|
| Expression | the line's average expression (log counts per million) over Stack's 15,012 genes |
| PCA | *principal component analysis*: the few directions in which the 50 lines' expression differs most |
| NMF | *non-negative matrix factorization*: expression rebuilt as a mix of a few gene programs |
| Stack base | the average Stack embedding of the line's cells, from the pretrained `bc_large.ckpt` |
| Stack cytokine | the same, from the released cytokine-tuned `bc_large_aligned.ckpt` |
| Stack drug | the same, from our sci-Plex fine-tune of `bc_large.ckpt` (archive file `epoch=5-val_loss=6.1078.ckpt`) |
| Random stand-in | random numbers of the same size, drawn once per line with a fixed seed |

- **Cells.** Every description uses the same cells: up to 1,000 DMSO cells per line, spread evenly over its
  plates, picked with a fixed seed. Stack gets raw counts, as it was trained, in groups from one line only.
- **Rescaling.** Descriptions are rescaled across all 50 lines, and PCA and NMF use all 50. Descriptions
  hold no drug responses, so this reveals nothing about a hidden answer.
- **Drugs (leave one drug out only).** A drug is described by the small chemical groups it contains (a
  Morgan fingerprint, from Tahoe's drug table). Two drugs' similarity is the share of groups they have in
  common (Tanimoto similarity).

## 5. The models

- **Drug average** (hiding lines): the drug's average change in the 49 training lines. No line description.
- **Chemistry only** (hiding drugs): each line's average change over training drugs, adjusted using
  chemically similar drugs. No line description.
- **Ridge regression**, a linear model held back from fitting noise, for every description: expression, PCA,
  NMF, three Stack versions, and each random stand-in. One method for all, so only the description changes.
  - Hiding a line: for each drug, it learns how the 49 lines' departures from the drug average follow their
    descriptions, predicts the hidden line's departure, and adds the average back.
  - Hiding a drug: it learns from all training pairs at once, letting chemically similar drugs share
    effects, on average and in lines with similar descriptions.
  - How strongly it is held back, and the number of PCA or NMF components (2, 5, 10, 15 or 20), are
    chosen by leaving out one training line (or drug) at a time.
- **Nearest lines** (hiding lines): the average change in the k training lines whose expression looks most
  like the hidden line; k (3, 5, 10 or 20) is chosen the same way.
- **Untested genes** count as zero change in training and are never scored.

**Why beating the drug average means predicting line-specific response.** The drug average already holds
everything about a pair that does not depend on the line. A model can beat it only by using what it was
told about the line. Each random stand-in goes through the same fit, so a description that beats its
stand-in gains from the description, not the method. For hidden drugs, chemistry only plays this role.

## 6. Scoring, and why the ceiling is a square root

Each model gets its average score over pairs with a confidence interval, also shown as a fraction of the
best possible score, √SB.

**Why the square root.**
- **Two measurements.** Rung 0 compared two halves of one experiment. Each half carries its own noise, so
  their agreement is lowered twice.
- **One prediction.** A prediction is not a second measurement, so only one side is noisy.
- **In symbols.** A measurement is X = T + e (true change plus noise), and R = var(T) / var(X). Two
  measurements correlate at R. The true change correlates with one measurement at √R.

Dividing by R would let a perfect model score over 100%. The archive rung 1 saw models at 0.28–0.32 beat a
"ceiling" of 0.109; against √0.197 = 0.44 they reach 62–72%.

Rung 1 predicts the average over all plates, so R is rung 0's full-data reliability: its split-half
agreement lifted by the Spearman-Brown formula (SB), the standard step from half data to full data. The
values come from rung 0's promoted per-pair table (`rung0_per_pair_r.csv`), filtered to our pairs:

| Genes | Pairs rung 0 scored | Split-half r | SB | Ceiling √SB | Rung 0's promoted 5 uM row (r / SB / √SB) |
|---|---|---|---|---|---|
| Responding | 4,593 | 0.5815 | 0.7353 | **0.8575** | 0.5768 / 0.7316 / 0.8553 |
| All | 5,350 | 0.0812 | 0.1503 | **0.3876** | 0.0808 / 0.1494 / 0.3865 |

**Known limits.**
1. **Averaging order.** Averaging over pairs before the square root makes the ceiling a little high, so
   the fraction reads a little low.
2. **Different responding genes.** Rung 0 picked responding genes from half the plates, and rung 1 from
   all of them. Rung 0's number is the closest ceiling that exists.
3. **Three-plate pairs.** 50 pairs have three plates. Their answer is more reliable than SB assumes, so
   the fraction reads a little high for them.
4. **Shared wells.** All 50 lines share wells, so a well's lab error reaches hidden and training lines
   alike. It cancels in comparisons, but can inflate the fraction by an amount this task does not measure.

## 7. Comparisons and statistics

Each hypothesis is a head-to-head comparison on the same pairs: the average of one model's score minus
another's, on responding genes.

| Test | Comparisons | Hypothesis |
|---|---|---|
| Leave one line out (main, 6) | Stack base − drug average | H1(a) |
| | Stack base − expression; − PCA; − NMF; − nearest lines | H1(b) |
| | Stack drug − Stack cytokine | H2 |
| Leave one drug out (5) | Stack base − chemistry only | H1(a) |
| | Stack base − expression; − PCA; − NMF | H1(b) |
| | Stack drug − Stack cytokine | H2 |

- **Uncertainty.** Redraw the hidden units with repeats (the 50 lines, or the 107 drugs) 2,000 times, and
  recompute each comparison.
  - **Confidence interval:** the middle 95% of redraws.
  - **p-value:** twice the share of redraws past zero.
  - **MDE:** 2.8 × the redraws' standard deviation (5% false positives, 80% power).
- **Several comparisons.** p-values are adjusted by Holm's method within each test.
- **Pairs share lines and drugs.** We also redraw lines and drugs together once and report how much wider intervals get.
- **Also reported, unadjusted:** the same comparisons on all genes, each description against its random
  stand-in, and each model's average score and fraction of the ceiling.
- **Models that saw the answer.**
  - **The sci-Plex fine-tune** saw the A549 line (DepMap ACH-000681) with five Tahoe drugs (PubChem 6918289,
    11626560, 104741, 11707110, 3385). Those pairs are removed for every model, and the count is reported.
  - **Stack's pretraining** may include untreated cells of these lines: input, not answer. No Stack saw Tahoe's treated cells.
  - Each model version's exposure is written to a `LeakageProfile` record.

## 8. Figures, controls and power (project rule 4)

Each step has a positive control (a planted answer the real code must find) and a negative control (no
signal, which must come out empty). Each figure is drawn from a table the run writes, with its control
beside it. Tolerances go in `plan.md`, from each test's spread by chance.

- **build** — line descriptions and answers.
  - positive: a description built from half of a line's cells best matches that line's other half, for
    more lines than the negative control's 99th percentile.
  - negative: with cells' line labels shuffled first, matches fall to chance (1 in 50). Check: the drug
    fine-tune's weights differ from `bc_large.ckpt`.
  - figures: cells per line and plate; half-versus-half match grids per description, shuffled grid beside.
- **split** — hiding a line or a drug.
  - positive: a signal planted only in one line's (or drug's) own responses is found by a broken split that
    leaves the hidden unit in training.
  - negative: the real splits score it at zero, and a test confirms no hidden value reaches training.
  - figures: the 50 × 107 grid of rounds; the planted signal under the broken and the real split.
- **fit** — ridge regression and nearest lines.
  - positive: on synthetic data of the screen's size and noise, a line-specific response planted at twice
    its MDE is recovered by the matching description and not by its random stand-in.
  - negative: with nothing planted, no model beats the drug average by more than its MDE.
  - figures: settings chosen per round; planted gain found, beside each model's real score.
- **score** — the score and the √SB ceiling.
  - positive: synthetic answers of known reliability R (the §6 values): the true change scores √R and a
    second noisy copy scores R, through the real scoring code.
  - negative: a prediction unrelated to the answer scores zero.
  - figures: each model's scores on real data with √SB and rung 0's r marked; the √R test beside it.
- **null** — the redraws and the random stand-ins.
  - positive: a comparison planted exactly at its MDE is detected about 80% of the time.
  - negative: with nothing planted, detections happen at most 5% of the time.
  - figures: every comparison with its interval and MDE, both gene sets; redraws for H1(a) and H2.

## 9. Running on Alpine (the university computing cluster)

**Access.** Heavy steps run on Alpine only through `scripts/alpine/ralpine`, never plain ssh or file copies.
Push the working branch, run `ralpine switch rung1-held-out-prediction-work` once, then `ralpine update`
and `ralpine submit`. First confirm inputs exist (`ralpine ls`, `ralpine du`), and pass the synthetic
tests locally.

**Run everything possible in parallel,** as *job arrays* (many copies of one job, each on its own slice).
- **Chaining.** Chain stages with `--dependency=afterok:<job id>`, and confirm each with `ralpine jobinfo <id>`.
- **Resubmitting.** A task whose output already exists skips, so a failed task reruns alone (`--array=<index>`).
- **Logs.** Pull every log into `results/rung1-held-out-prediction/logs/`.

| Stage | How it runs in parallel | Waits for |
|---|---|---|
| 1. DMSO cells and drug table (Hugging Face) | array over source files, at most 4 at once (`%4`); many readers get rate-limited | — |
| 2. Answers (5 uM, 107 drugs) | read rung 0's 32 cached gene slices; if scratch was purged, re-run rung 0's slice array with these filters | — |
| 3. Stack embeddings | 3-task GPU array: base, cytokine, drug | 1 |
| 4. Expression, PCA, NMF, random descriptions | one small job | 1 |
| 5. Model fits | 157-task array, one per round (50 lines + 107 drugs), every model per task | 2, 3, 4 |
| 6. Redraws | array: 8 tasks × 250 redraws, per test | 5 |
| 7. Tables and figures | one job | 6 |

**Which nodes were free during rung 0 (2026-09-10 and 11).**
- **`acpu`** (general CPU, `--qos=cpu-normal`; `amilan` was retired 2026-08-24). 1,353 jobs were waiting,
  914 ahead of this account.
- **`amem`** (1 TB high-memory). Eight nodes sat idle, and rung 0 ran there. These need `--qos=mem-normal`
  and at least 256 GB, so small tasks share a whole node: rung 0 put 11 slices on each (48 cores, 990 GB).
  Each finished in 2.5–2.8 hours (c3mem-a7-u38-3, c3mem-a7-u38-4, c3mem-a5-u36-2). A core costs 4 times an
  `acpu` core.
- **GPU.** August's Stack jobs moved from `aa100` (long backlog) to `ah200` (H200), with
  `--qos=gpu-normal --gres=gpu:h200:1`.
- **Disk.** Scratch reads were 3 times slower on 2026-09-10 than the day before; give time limits slack.

**Choosing a partition.** Before submitting, run `ralpine run squeue -p acpu -t PENDING` and
`ralpine run sinfo -p amem`. If `acpu` is deep and `amem` is idle, pack stage 5 onto `amem`. Alpine charges
memory per core (3,840 MB), so ask for memory in whole cores.

## 10. Commits and the pull request

- **Working branch.** Work in the worktree `.claude/worktrees/rung1-held-out-prediction` on
  `rung1-held-out-prediction-work`. Commit and push as needed, since pushing is how code reaches Alpine.
- **Rebuild for review.** Before the pull request, the final files are recommitted as a few topic commits on
  `rung1-held-out-prediction`, from the same base, as rung 0 was rebuilt on 2026-09-11:
  1. design, decisions, SPEC/STATE/README
  2. data: cells, drug structures, descriptions, their jobs and tests
  3. measurement: splits, models, scores, redraws, their jobs and tests
  4. verification script and notebooks
  5. review and audit records

  Results come later, once the new storage for results exists; until then no run output enters git.
- **Check before pushing.** `git diff rung1-held-out-prediction-work rung1-held-out-prediction` must be
  empty. The pull request comes from `rung1-held-out-prediction`, so nothing is force-pushed and Alpine's
  copy never goes out of sync. The working branch is deleted after merge, with Lucas's OK.

## 11. Earlier evidence (archive branch `worktree-modular-harness-core`)

- **Its rung 1** (doses pooled, 1,508 pairs, lines hidden): drug average 0.315, PCA 0.319, NMF 0.313, nearest
  lines 0.276. No expression model beat the drug average.
- **Stack generating responses** scored near zero (0.012 cytokine, 0.021 drug), even with the hidden line leaked in.
- **For cell survival**, only Stack base showed a line-specific signal (0.119), but a model ignoring lines
  reached 0.118, blamed on 17 missing pairs; hence the complete grid.

## 12. Not in this task, and what changes on approval

- **Not in this task:** Stack generating responses directly (5,350 generation runs per model version to
  hide lines properly); the 0.05 and 0.5 uM doses; re-running rung 0; hiding a line and a drug together.
- **On approval:** `docs/SPEC.md`'s rung 1 entry is rewritten to match §1–§7 (proposed text in
  `decisions.md`), and STATE and README are updated to match.
